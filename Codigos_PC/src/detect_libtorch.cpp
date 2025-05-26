#include "detect_libtorch.hpp"


// Convierte una imagen OpenCV a tensor Torch para entrada
at::Tensor preprocess_image(const cv::Mat& img, int height, int width) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(width, height));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    auto input_tensor = torch::from_blob(resized.data, {1, height, width, 3}).permute({0, 3, 1, 2}).contiguous();
    return input_tensor.clone();
}

float get_depth_in_bbox(const cv::Mat& depth_map, const cv::Rect& box) {
    cv::Rect roi = box & cv::Rect(0, 0, depth_map.cols, depth_map.rows);
    cv::Mat region = depth_map(roi);
    cv::Scalar avg = cv::mean(region);
    return static_cast<float>(avg[0]);
}

void detect_with_depth(const cv::Mat& image,
                       torch::jit::script::Module& yolo_model,
                       torch::jit::script::Module& midas_model,
                       std::vector<Detection>& output,
                       const std::vector<std::string>& class_list) {
    //Profundidad (MiDaS)
    auto midas_input = preprocess_image(image, 256, 256);
    midas_input = midas_input.to(torch::kFloat);
    auto midas_out = midas_model.forward({midas_input}).toTensor();
    
    c10::ArrayRef<int64_t> out_size = {image.rows, image.cols};
    c10::optional<c10::ArrayRef<double>> scale_factors = c10::nullopt;

    midas_out = at::upsample_bilinear2d(midas_out, out_size, /*align_corners=*/false, scale_factors);
    cv::Mat depth(image.rows, image.cols, CV_32F, midas_out.squeeze().contiguous().data_ptr<float>());

    //Detección (YOLO)
    auto yolo_input = preprocess_image(image, 640, 640).to(torch::kFloat);
    auto yolo_out_raw = yolo_model.forward({yolo_input}).toTuple()->elements()[0].toTensor();
    auto yolo_out = yolo_out_raw.squeeze(0);

    float x_factor = image.cols / 640.0;
    float y_factor = image.rows / 640.0;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < yolo_out.size(0); ++i) {
        auto row = yolo_out[i];
        float confidence = row[4].item<float>();
        if (confidence < 0.4f) continue;

        auto scores = row.slice(0, 5);
        auto max_result = scores.max(0);
        float score = std::get<0>(max_result).item<float>();
        int class_id = std::get<1>(max_result).item<int>();
        if (score < 0.2f) continue;

        float x = row[0].item<float>();
        float y = row[1].item<float>();
        float w = row[2].item<float>();
        float h = row[3].item<float>();
        int left = static_cast<int>((x - 0.5 * w) * x_factor);
        int top = static_cast<int>((y - 0.5 * h) * y_factor);
        int width = static_cast<int>(w * x_factor);
        int height = static_cast<int>(h * y_factor);

        boxes.push_back(cv::Rect(left, top, width, height));
        confidences.push_back(confidence);
        class_ids.push_back(class_id);
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, 0.2f, 0.4f, nms_result);

    for (int i : nms_result) {
        float depth_val = get_depth_in_bbox(depth, boxes[i]);
        output.push_back({class_ids[i], confidences[i], boxes[i], depth_val});
    }
}

