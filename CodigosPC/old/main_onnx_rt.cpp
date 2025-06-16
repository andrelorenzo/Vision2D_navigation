#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <chrono>
#include <string>
#include <onnxruntime_cxx_api.h>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Uso: " << argv[0] << " <modelo.onnx> <input_width> <input_height> <opencv|onnx> [depth|imagenet]" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    int input_width = std::stoi(argv[2]);
    int input_height = std::stoi(argv[3]);
    std::string backend = argv[4];
    std::string normalization = (argc >= 6) ? argv[5] : "depth";
    const cv::Size vis_size(640, 480);

    bool use_opencv = (backend == "opencv");
    bool use_imagenet_norm = (normalization == "imagenet");

    cv::dnn::Net net;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_backend");
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::string input_name_str, output_name_str;
    const char* input_name = nullptr;
    const char* output_name = nullptr;

    if (use_opencv) {
        net = cv::dnn::readNetFromONNX(model_path);
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout << "Usando OpenCV DNN + CUDA" << std::endl;
        } else {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            std::cout << "Usando OpenCV DNN + CPU" << std::endl;
        }
    } else {
        session_options.SetIntraOpNumThreads(6);
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        Ort::AllocatedStringPtr input_name_ptr = session->GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr output_name_ptr = session->GetOutputNameAllocated(0, allocator);
        input_name_str = input_name_ptr.get();   // Copiamos el contenido antes de que se destruya
        output_name_str = output_name_ptr.get();
        input_name = input_name_str.c_str();
        output_name = output_name_str.c_str();
        std::cout << "Usando ONNX Runtime (CPU)" << std::endl;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: no se pudo abrir la cámara." << std::endl;
        return -1;
    }

    int frame_count = 0;
    auto t_start = std::chrono::high_resolution_clock::now();

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat input;
        cv::resize(frame, input, cv::Size(input_width, input_height));
        input.convertTo(input, CV_32F, 1.0 / 255.0);
        cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
        if (use_imagenet_norm) {
            cv::Mat channels[3];
            cv::split(input, channels);
            channels[0] = (channels[0] - 0.485f) / 0.229f;
            channels[1] = (channels[1] - 0.456f) / 0.224f;
            channels[2] = (channels[2] - 0.406f) / 0.225f;
            cv::merge(channels, 3, input);
        } else {
            input = (input - 0.5f) / 0.5f;
        }

        cv::Mat output;
        if (use_opencv) {
            cv::Mat blob = cv::dnn::blobFromImage(input);
            net.setInput(blob);
            output = net.forward();
            if (output.dims == 4) {
                output = output.reshape(1, output.size[2]);
            }
        } else {
            std::vector<float> input_tensor(1 * 3 * input_height * input_width);
            int idx = 0;
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < input_height; ++y)
                    for (int x = 0; x < input_width; ++x)
                        input_tensor[idx++] = input.at<cv::Vec3f>(y, x)[c];

            std::array<int64_t, 4> input_shape = {1, 3, input_height, input_width};
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor.data(), input_tensor.size(),
                input_shape.data(), input_shape.size());

            auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                               &input_name, &input_tensor_ort, 1,
                                               &output_name, 1);
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

            if (output_shape.size() == 3) {
                output = cv::Mat((int)output_shape[1], (int)output_shape[2], CV_32F, output_data).clone();
            } else if (output_shape.size() == 4 && output_shape[0] == 1) {
                output = cv::Mat((int)output_shape[2], (int)output_shape[3], CV_32F, output_data).clone();
            } else {
                std::cerr << "Forma de salida no esperada." << std::endl;
                break;
            }
        }

        cv::normalize(output, output, 0, 255, cv::NORM_MINMAX);
        output.convertTo(output, CV_8U);
        cv::applyColorMap(output, output, cv::COLORMAP_MAGMA);
        cv::resize(output, output, vis_size);

        cv::imshow("Output", output);

        frame_count++;
        auto t_now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_now - t_start).count();
        if (elapsed >= 1.0) {
            double fps = frame_count / elapsed;
            std::cout << "FPS: " << fps << " (" << 1000.0 / fps << " ms/frame)" << std::endl;
            std::cout << "\033[2J\033[1;1H";
            frame_count = 0;
            t_start = t_now;
        }

        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
