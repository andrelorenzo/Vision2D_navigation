#include "helper.hpp"

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
#ifdef PROF
    InstrumentationTimer timer("get_frame_from_tcp");
#endif
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
#ifdef PROF
    InstrumentationTimer timer("get_frame_from_camera");
#endif
    cv::Mat frame;
    cap >> frame;
    return frame;
}

cv::Mat normalize_depth_with_percentile(const cv::Mat& input, float lower_percent, float upper_percent ) {
#ifdef PROF
    InstrumentationTimer timer("normalize_depth_with_percentile");
#endif
    CV_Assert(input.type() == CV_32F);

    // Aplanar y filtrar valores válidos
    std::vector<float> values;
    values.reserve(input.total());

    for (int y = 0; y < input.rows; ++y) {
        const float* row_ptr = input.ptr<float>(y);
        for (int x = 0; x < input.cols; ++x) {
            float val = row_ptr[x];
            if (std::isfinite(val))
                values.push_back(val);
        }
    }

    if (values.size() < 100) return cv::Mat::zeros(input.size(), CV_32F);

    std::sort(values.begin(), values.end());

    size_t idx_low = static_cast<size_t>(lower_percent / 100.0f * values.size());
    size_t idx_high = static_cast<size_t>(upper_percent / 100.0f * values.size()) - 1;

    float min_val = values[idx_low];
    float max_val = values[idx_high];
    float range = max_val - min_val;

    if (range < 1e-6f) return cv::Mat::zeros(input.size(), CV_32F);

    cv::Mat normalized;
    input.convertTo(normalized, CV_32F);
    normalized = (normalized - min_val) / range;
    cv::threshold(normalized, normalized, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(normalized, normalized, 0.0, 0.0, cv::THRESH_TOZERO);
    return normalized;
}
// void exponential_smoothing(const cv::Mat& current, cv::Mat& filtered, float alpha) {
//     CV_Assert(current.type() == CV_32F);

//     if (filtered.empty()) {
//         filtered = current.clone();
//     } else {
//         filtered = alpha * filtered + (1.0f - alpha) * current;
//     }
// }

cv::Mat exponential_smoothing(const cv::Mat current, cv::Mat future, float alpha) {
#ifdef PROF
    InstrumentationTimer timer("exponential_smoothing");
#endif
    CV_Assert(current.type() == CV_32F);

    if (future.empty()) {
        return current.clone();
    } else {
        return alpha * current + (1.0f - alpha) * future;
    }
}

void scale_depth_map(cv::Mat& depth_map, float depth_reference_m, float midas_value_reference) {
#ifdef PROF
    InstrumentationTimer timer("scale_depth_map");
#endif
    if (depth_map.empty() || depth_map.type() != CV_32F || midas_value_reference < 1e-6f)
        return;

    cv::Mat safe_map;
    cv::max(depth_map, 1e-6f, safe_map);  // evita dividir por cero o valores negativos

    cv::Mat inverted;
    cv::divide(1.0f, safe_map, inverted);  // 1 / valor

    float scale_factor = depth_reference_m * midas_value_reference;

    depth_map = inverted * scale_factor;
}
float estimate_distance_from_yolo(const ObjectBBox& det,
                                  const std::unordered_map<std::string, std::pair<float, float>>& sizes,
                                  float focal_px) {
#ifdef PROF
    InstrumentationTimer timer("estimate_distance_from_yolo");
#endif
    auto it = sizes.find(det.label);
    if (it == sizes.end()) return -1.0f;

    float height_real = it->second.first;  // Altura real en metros
    float height_px = std::abs(det.y2 - det.y1);
    if (height_px <= 0.0f) return - 1.0f;

    return focal_px * height_real / height_px;
}

std::unordered_map<std::string, std::pair<float, float>> load_object_sizes(const std::string& filename) {
#ifdef PROF
    InstrumentationTimer timer("load_object_sizes");
#endif
    std::unordered_map<std::string, std::pair<float, float>> object_sizes;
    std::ifstream infile(filename);
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string class_name;
        float height, width;
        if (!(iss >> class_name >> height >> width)) continue;
        object_sizes[class_name] = std::make_pair(height, width);
    }

    return object_sizes;
}
void draw_mean_slope_arrow_sobel(cv::Mat& vis, const cv::Mat& depth_map_input) {
#ifdef PROF
    InstrumentationTimer timer("draw_mean_slope_arrow_sobel");
#endif
    if (depth_map_input.empty()) {
        std::cout << "[DEBUG] Depth map is empty\n";
        return;
    }
    if (depth_map_input.type() != CV_32F) {
        std::cout << "[DEBUG] Depth map type is not CV_32F: " << depth_map_input.type() << "\n";
        return;
    }

    // Clonar y verificar rango
    cv::Mat depth_map = depth_map_input.clone();

    // Imprimir min/max reales
    double minVal, maxVal;
    cv::minMaxLoc(depth_map, &minVal, &maxVal);
    // std::cout << "[DEPTH RANGE] min: " << minVal << ", max: " << maxVal << "\n";

    // Preprocesado básico
    cv::patchNaNs(depth_map, 0.0);
    cv::threshold(depth_map, depth_map, 1e4, 0.0, cv::THRESH_TOZERO_INV);

    // Generar máscara válida
    cv::Mat valid_mask = (depth_map > 0) & (depth_map < 1e4);
    int num_valid = cv::countNonZero(valid_mask);
    if (num_valid < 100) {
        std::cout << "[DEBUG] Too few valid pixels.\n";
        return;
    }

    // Suavizado espacial
    cv::GaussianBlur(depth_map, depth_map, cv::Size(3, 3), 0.5);

    // Gradientes con Sobel
    cv::Mat dx, dy;
    cv::Sobel(depth_map, dx, CV_32F, 1, 0, 3);
    cv::Sobel(depth_map, dy, CV_32F, 0, 1, 3);

    dx = -dx;
    dy = -dy;

    // Enmascarar gradientes
    cv::Mat valid_dx, valid_dy;
    dx.copyTo(valid_dx, valid_mask);
    dy.copyTo(valid_dy, valid_mask);

    // Media de gradientes
    cv::Scalar mean_dx = cv::mean(valid_dx, valid_mask);
    cv::Scalar mean_dy = cv::mean(valid_dy, valid_mask);

    // Escalado fijo
    const float gradient_scale = 1000.0f;
    mean_dx *= gradient_scale;
    mean_dy *= gradient_scale;

    // Magnitud del gradiente
    cv::Mat grad_mag;
    cv::magnitude(valid_dx, valid_dy, grad_mag);

    cv::Scalar mean_mag, stddev_mag;
    cv::meanStdDev(grad_mag, mean_mag, stddev_mag, valid_mask);

    // Dirección y norma
    cv::Point2f dir(mean_dx[0], mean_dy[0]);
    float norm = std::sqrt(dir.x * dir.x + dir.y * dir.y);

    if (stddev_mag[0] < 1e-3 || norm < 1e-3) {
        std::cout << "[DEBUG] Norm or deviation too small, not drawing arrow\n";
        return;
    }

    // Dibujar flecha en centro
    const float factor_visual = 10.0f;
    dir *= factor_visual;

    cv::Point2f center(depth_map.cols / 2.0f, depth_map.rows / 2.0f);
    cv::Point2f end = center + dir;
    cv::arrowedLine(vis, center, end, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    // Mostrar magnitud numérica
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
#ifdef PROF
    InstrumentationTimer timer("estimate_depth_map");
#endif
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    torch::NoGradGuard no_grad;

    // Preprocesado
    cv::Mat resized;
    cv::resize(frame, resized, input_size);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    if (swapRB)
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    auto input_tensor = torch::from_blob(resized.data, {1, input_size.height, input_size.width, 3}, torch::kFloat)
                            .permute({0, 3, 1, 2})  // NCHW
                            .contiguous()
                            .to(device);

    input_tensor = (input_tensor - torch::tensor(mean).view({1, 3, 1, 1}).to(device)) /
                   torch::tensor(std).view({1, 3, 1, 1}).to(device);

    // Inferencia
    auto output = model.forward({input_tensor}).toTensor();

    // Aplanar dimensiones
    if (output.dim() == 4) output = output.squeeze(0).squeeze(0);
    else if (output.dim() == 3) output = output.squeeze(0);
    else throw std::runtime_error("Unexpected output shape from depth model.");

    // Convertir a cv::Mat
    output = output.detach().to(torch::kCPU);
    cv::Mat depth_map(output.size(0), output.size(1), CV_32F, output.data_ptr<float>());
    depth_map = depth_map.clone();

    cv::resize(depth_map, depth_map, frame.size(), 0, 0, cv::INTER_CUBIC);
    return depth_map;
}
cv::Mat estimate_depth_map(cv::dnn::Net& net, const cv::Mat& frame,
                           const cv::Size& input_size = cv::Size(256, 256),
                           const std::vector<float>& mean = {},
                           const std::vector<float>& std = {},
                           bool swapRB = true, bool crop = false) {
#ifdef PROF
    InstrumentationTimer timer("estimate_depth_map_dnn");
#endif
    if (frame.empty()) {
        throw std::runtime_error("Empty input frame.");
    }

    // Convertir a formato adecuado
    cv::Mat input, resized, blob;

    if (frame.channels() == 1) {
        cv::cvtColor(frame, input, cv::COLOR_GRAY2BGR);
    } else if (frame.channels() == 3) {
        input = frame;
    } else {
        throw std::runtime_error("Unsupported number of channels in input image.");
    }

    cv::resize(input, resized, input_size);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // Crear blob (normalizado a [0,1], con o sin swapRB)
    blob = cv::dnn::blobFromImage(resized, 1.0, input_size, cv::Scalar(), swapRB, crop, CV_32F);

    // Normalización adicional si se proporciona mean/std
    if (!mean.empty() && !std.empty() && mean.size() == std.size() && mean.size() == blob.size[1]) {
        int channels = blob.size[1];
        int height = blob.size[2];
        int width = blob.size[3];
        for (int c = 0; c < channels; ++c) {
            float* ptr = blob.ptr<float>(0, c);
            for (int i = 0; i < height * width; ++i) {
                ptr[i] = (ptr[i] - mean[c]) / std[c];
            }
        }
    }

    // Inferencia
    net.setInput(blob);
    cv::Mat depth = net.forward();

    // Aplanar dimensiones si es [1,1,H,W] o [1,H,W]
    if (depth.dims == 4 && depth.size[0] == 1 && depth.size[1] == 1) {
        depth = depth.reshape(1, { depth.size[2], depth.size[3] });
    } else if (depth.dims == 3 && depth.size[0] == 1) {
        depth = depth.reshape(1, { depth.size[1], depth.size[2] });
    } else {
        throw std::runtime_error("Unexpected output shape from depth model.");
    }

    // Redimensionar al tamaño original de la imagen
    cv::Mat depth_map;
    cv::resize(depth, depth_map, frame.size(), 0, 0, cv::INTER_CUBIC);
    return depth_map;
}

// cv::Mat estimate_depth_map(cv::dnn::Net& midas, const cv::Mat& frame) {
//     cv::Mat resized, input_blob, depth_map;

//     cv::resize(frame, resized, cv::Size(256, 256));
//     resized.convertTo(resized, CV_32F, 1.0 / 255.0);

//     input_blob = cv::dnn::blobFromImage(resized);
//     midas.setInput(input_blob);
//     depth_map = midas.forward();
//     depth_map = depth_map.reshape(1, 256);
//     cv::resize(depth_map, depth_map, frame.size());

//     return depth_map;
// }




// void annotate_with_depth(cv::Mat& frame, const cv::Mat& depth_map, std::vector<ObjectBBox>& detections) {
//     for (auto& bbox : detections) {
        
//         int x1 = std::clamp(static_cast<int>(bbox.x1), 0, frame.cols - 1);
//         int y1 = std::clamp(static_cast<int>(bbox.y1), 0, frame.rows - 1);
//         int x2 = std::clamp(static_cast<int>(bbox.x2), 0, frame.cols - 1);
//         int y2 = std::clamp(static_cast<int>(bbox.y2), 0, frame.rows - 1);

//         // Asegurarse de que x2 > x1 y y2 > y1
//         if (x2 <= x1 || y2 <= y1) continue;

//         cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
//         cv::Mat depth_roi = depth_map(roi);
//         cv::resize(depth_map, depth_map, frame.size());
//         CV_Assert(depth_map.size() == frame.size());
//         float distance = static_cast<float>(cv::mean(depth_roi)[0]);

//         std::ostringstream label;
//         label << bbox.label << " " << std::fixed << std::setprecision(2)
//               << bbox.conf << " D:" << distance;

//         bbox.draw(frame);
//         cv::putText(frame, label.str(), cv::Point(x1, y1 - 5),
//                     cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 2);
//     }
// }

void annotate_with_depth(cv::Mat& frame,
                         cv::Mat& depth_map,
                         std::vector<ObjectBBox>& detections,
                         const std::unordered_map<std::string, std::pair<float, float>>& object_sizes,
                         bool& depth_scaled_flag) {
#ifdef PROF
    InstrumentationTimer timer("annotate_with_depth");
#endif
    const float focal_px = 500.0f;//1472.768f;  // Estimación de focal (ajustable si calibras)

    cv::resize(depth_map, depth_map, frame.size());
    CV_Assert(depth_map.size() == frame.size());

    for (auto& bbox : detections) {
        int x1 = std::clamp(static_cast<int>(bbox.x1), 0, frame.cols - 1);
        int y1 = std::clamp(static_cast<int>(bbox.y1), 0, frame.rows - 1);
        int x2 = std::clamp(static_cast<int>(bbox.x2), 0, frame.cols - 1);
        int y2 = std::clamp(static_cast<int>(bbox.y2), 0, frame.rows - 1);

        if (x2 <= x1 || y2 <= y1) continue;

        cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
        cv::Mat depth_roi = depth_map(roi);

        float d_midas = static_cast<float>(cv::mean(depth_roi)[0]);

        // Si aún no hemos escalado el mapa, intentamos hacerlo con esta detección
        if (!depth_scaled_flag) {
            float z_real = estimate_distance_from_yolo(bbox, object_sizes, focal_px);
            if (z_real > 0.0f && d_midas > 1e-6f) {
                scale_depth_map(depth_map, z_real, d_midas);
                depth_scaled_flag = true;
                // Recalcular d_midas después del escalado
                d_midas = static_cast<float>(cv::mean(depth_map(roi))[0]);

                std::cout << "[INFO] Escalado de profundidad aplicado: "
                          << z_real << " / " << d_midas << " = " << (z_real / d_midas)
                          << " (clase: " << bbox.label << ")\n";
            }
        }

        std::ostringstream label;
        label << bbox.label << " " << std::fixed << std::setprecision(2)
              << bbox.conf << " D:" << d_midas << "m";

        bbox.draw(frame);
        cv::putText(frame, label.str(), cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255   , 0, 0), 3);
    }
}



cv::Mat estimate_depth_variant(DepthModel& model, const cv::Mat& frame) {
    return std::visit(overloaded {
        [&frame](cv::dnn::Net& net) -> cv::Mat {
            return estimate_depth_map(net, frame,
                cv::Size(256,256),
                {0.5f, 0.5f, 0.5f},
                {0.5f, 0.5f, 0.5f},
                true, false);
        },
        [&frame](torch::jit::script::Module& mod) -> cv::Mat {
            return estimate_depth_map(mod, frame,
                cv::Size(518, 518),
                {0.5f, 0.5f, 0.5f},
                {0.5f, 0.5f, 0.5f},
                true, false);
        }
    }, model);
}

