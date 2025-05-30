#include "aux.hpp"

void signal_thread_done() {
    std::lock_guard<std::mutex> lock(threads_mutex);
    threads_done++;
    threads_cv.notify_one();
}

void wait_for_threads(int required) {
    std::unique_lock<std::mutex> lock(threads_mutex);
    threads_cv.wait(lock, [&] { return threads_done.load() >= required; });
}   

bool recvAll(int sock, uint8_t* buffer, size_t length) {
    size_t total = 0;
    while (total < length) {
        ssize_t bytes = recv(sock, buffer + total, length - total, 0);
        if (bytes <= 0) return false;
        total += bytes;
    }
    return true;
}

cv::Mat get_frame_from_tcp(int sock) {
    send(sock, CAPTURE_CMD, 4, 0);

    uint8_t sizeBytes[4];
    if (!recvAll(sock, sizeBytes, 4)) return cv::Mat();

    int imgSize = (sizeBytes[0] << 24) | (sizeBytes[1] << 16) | (sizeBytes[2] << 8) | sizeBytes[3];
    if (imgSize <= 0 || imgSize > 150000) return cv::Mat();

    std::vector<uint8_t> imgBuffer(imgSize);
    if (!recvAll(sock, imgBuffer.data(), imgSize)) return cv::Mat();

    return cv::imdecode(imgBuffer, cv::IMREAD_COLOR);
}

cv::Mat get_frame_from_camera(cv::VideoCapture& cap) {
    cv::Mat frame;
    cap >> frame;
    return frame;
}

void draw_mean_slope_arrow_sobel(cv::Mat& vis, const cv::Mat& depth_map) {
    if (depth_map.empty() || depth_map.type() != CV_32F) return;

    cv::Mat dx, dy;
    cv::Sobel(depth_map, dx, CV_32F, 1, 0, 3);  // Derivada en X
    cv::Sobel(depth_map, dy, CV_32F, 0, 1, 3);  // Derivada en Y

    // Invertimos los gradientes porque buscamos la bajada
    dx = -dx;
    dy = -dy;

    // Cálculo de dirección promedio
    cv::Scalar mean_dx = cv::mean(dx);
    cv::Scalar mean_dy = cv::mean(dy);
    cv::Mat grad_mag;
    cv::magnitude(dx, dy, grad_mag);

    cv::Scalar stddev;
    cv::meanStdDev(grad_mag, cv::noArray(), stddev);

    cv::Point2f dir(mean_dx[0], mean_dy[0]);
    float norm = std::sqrt(dir.x * dir.x + dir.y * dir.y);

    if (stddev[0] < DEV_TRHS || norm < MIN_TRHS ){
        norm = 0.0f;
        dir = cv::Point2f(0.f, 0.f);
    }

    // Escalado proporcional (opcional: ajustar factor_visual para visibilidad)
    const float factor_visual = 10.0f; // Aumentar si los vectores son muy pequeños
    dir *= factor_visual;

    // Dibujar si procede
    if (norm > 0.0f) {
        cv::Point2f center(depth_map.cols / 2.0f, depth_map.rows / 2.0f);
        cv::Point2f end = center + dir;
        cv::arrowedLine(vis, center, end, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }

    // Mostrar magnitud
    std::ostringstream text;
    text << "Mag: " << std::fixed << std::setprecision(2) << norm;
    cv::putText(vis, text.str(), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
}

cv::Mat estimate_depth_map(torch::jit::script::Module& model, const cv::Mat& frame,
                           const cv::Size& input_size,
                           const std::vector<float>& mean,
                           const std::vector<float>& std,
                           bool swapRB, bool crop) {
    // Preprocesado
    cv::Mat resized;
    cv::resize(frame, resized, input_size);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    if (swapRB)
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    for (int c = 0; c < 3; ++c) {
        resized.forEach<cv::Vec3f>([c, &mean, &std](cv::Vec3f& pixel, const int*) {
            pixel[c] = (pixel[c] - mean[c]) / std[c];
        });
    }

    // Convertir a tensor
    torch::Tensor input_tensor = torch::from_blob(resized.data, {1, input_size.height, input_size.width, 3}).permute({0, 3, 1, 2}).contiguous();

    // Inferencia
    std::vector<torch::jit::IValue> inputs = { input_tensor };
    torch::Tensor output = model.forward(inputs).toTensor();

    // Verificar forma
    // std::cout << "Output shape: " << output.sizes() << std::endl;

    // Aplanar posibles dimensiones [1, H, W] o [1, 1, H, W]
    if (output.dim() == 4 && output.size(0) == 1 && output.size(1) == 1) {
        output = output.squeeze(0).squeeze(0);  // [H, W]
    } else if (output.dim() == 3 && output.size(0) == 1) {
        output = output.squeeze(0);  // [H, W]
    } else {
        throw std::runtime_error("Unexpected output shape from depth model.");
    }

    // Convertir a cv::Mat
    output = output.detach().cpu();
    cv::Mat depth_map(output.size(0), output.size(1), CV_32F, output.data_ptr<float>());
    depth_map = depth_map.clone();

    cv::resize(depth_map, depth_map, frame.size(), 0, 0, cv::INTER_CUBIC);
    return depth_map;
}


void annotate_with_depth(cv::Mat& frame, const cv::Mat& depth_map, std::vector<ObjectBBox>& detections) {
    for (auto& bbox : detections) {
        int x1 = std::clamp(static_cast<int>(bbox.x1), 0, frame.cols - 1);
        int y1 = std::clamp(static_cast<int>(bbox.y1), 0, frame.rows - 1);
        int x2 = std::clamp(static_cast<int>(bbox.x2), 0, frame.cols - 1);
        int y2 = std::clamp(static_cast<int>(bbox.y2), 0, frame.rows - 1);

        // Asegurarse de que x2 > x1 y y2 > y1
        if (x2 <= x1 || y2 <= y1) continue;

        cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
        cv::Mat depth_roi = depth_map(roi);
        cv::resize(depth_map, depth_map, frame.size());
        CV_Assert(depth_map.size() == frame.size());
        float distance = static_cast<float>(cv::mean(depth_roi)[0]);

        std::ostringstream label;
        label << bbox.label << " " << std::fixed << std::setprecision(2)
              << bbox.conf << " D:" << distance;

        bbox.draw(frame);
        cv::putText(frame, label.str(), cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 2);
    }
}

cv::Mat estimate_midas_depth_v21(torch::jit::script::Module& model, const cv::Mat& frame) {
    return estimate_depth_map(model, frame,
                               cv::Size(256, 256),
                               {0.0f, 0.0f, 0.0f},
                               {1.0f, 1.0f, 1.0f},
                               true, false);
}

cv::Mat estimate_depth_anything_v2_outdoor(torch::jit::script::Module& model, const cv::Mat& frame) {
    return estimate_depth_map(model, frame,
                               cv::Size(518, 518),
                               {0.5f, 0.5f, 0.5f},
                               {0.5f, 0.5f, 0.5f},
                               true, false);
}
