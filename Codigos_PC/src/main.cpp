#include "helper.hpp"

int model = 0;
bool do_bench = true;

void on_mouse_depth(int event, int x, int y, int, void*) {
    if (!depth_scaled_for_debug.empty()) {
        if (x >= 0 && x < depth_scaled_for_debug.cols &&
            y >= 0 && y < depth_scaled_for_debug.rows) {
            float value = depth_scaled_for_debug.at<float>(y, x);
            std::cout << "Depth at (" << x << ", " << y << ") = "
                      << std::fixed << std::setprecision(2) << value << " meters" << std::endl;
        }
    }
}

void frame_capture_thread(bool use_tcp, int sock, cv::VideoCapture& cap) {
    while (!stop_flag) {
        cv::Mat frame = use_tcp ? get_frame_from_tcp(sock) : get_frame_from_camera(cap);
        if (frame.empty()) continue;
        {
            std::lock_guard<std::mutex> guard(frame_mutex);
            shared_frame = frame.clone();
        }
        global_frame_index++;
    }
}


void depth_thread(DepthModel& depth_model, DepthEstimationFn estimate_fn) {
    static uint64_t last_frame_index = 0;

    // Benchmarking variables
    std::deque<double> times_ms;
    auto last_time = std::chrono::high_resolution_clock::now();

    while (!stop_flag) {
        uint64_t current_index = global_frame_index.load();
        if (current_index <= last_frame_index) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            frame = shared_frame.clone();
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        cv::Mat depth = estimate_fn(depth_model, frame);
        if (depth.type() != CV_32F) {
            depth.convertTo(depth, CV_32F);
        }
        auto t_end = std::chrono::high_resolution_clock::now();

        {
            std::lock_guard<std::mutex> lock(depth_mutex);
            current_depth = depth.clone();
        }

        depth_done_frame_index = current_index;
        last_frame_index = current_index;

        // Benchmarking
        if (do_bench) {
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            times_ms.push_back(elapsed_ms);
            if (times_ms.size() > 10) times_ms.pop_front();
            double avg_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
            double fps = 1000.0 / avg_ms;

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "[Benchmark - Depth] Avg frame time (10): " << avg_ms << " ms | FPS: " << fps << std::endl;
            std::cout << "\033[2J\033[1;1H";
        }
    }
}


void yolo_thread(YOLOv11& yolo_model) {
    static uint64_t last_frame_index = 0;

    while (!stop_flag) {
        uint64_t current_index = global_frame_index.load();
        if (current_index <= last_frame_index) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            frame = shared_frame.clone();
        }

        std::vector<ObjectBBox> detections = yolo_model.detect(frame);
        {
            std::lock_guard<std::mutex> lock(yolo_mutex);
            current_detections = detections;
        }
        yolo_done_frame_index = current_index;
        last_frame_index = current_index;
    }
}


void annotate_depth_points(cv::Mat& vis_image, const cv::Mat& depth_map) {
    CV_Assert(!vis_image.empty() && vis_image.type() == CV_8UC3);
    CV_Assert(!depth_map.empty() && depth_map.type() == CV_32F);
    CV_Assert(vis_image.size() == depth_map.size());

    int w = depth_map.cols;
    int h = depth_map.rows;

    std::vector<cv::Point> points = {
        {w / 2, h / 2},           // Centro
        {w / 4, h / 4},           // Cuadrante superior izquierdo
        {3 * w / 4, h / 4},       // Cuadrante superior derecho
        {w / 4, 3 * h / 4},       // Cuadrante inferior izquierdo
        {3 * w / 4, 3 * h / 4}    // Cuadrante inferior derecho
    };

    for (const auto& pt : points) {
        float d = depth_map.at<float>(pt);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << d << " m";

        cv::circle(vis_image, pt, 3, cv::Scalar(255, 255, 255), -1);  // punto blanco
        cv::putText(vis_image, oss.str(), pt + cv::Point(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}


uint64_t frames = 0;
int main(int argc, char** argv) {
#ifdef PROF
    Instrumentor::Get().BeginSession("both_cpu","../profiling/mgpu_ycpu.json");
#endif
    bool use_tcp = true;
    bool use_yolo = false;

    if (torch::cuda::is_available()) {
        std::cout << "CUDA está disponible para LibTorch.\n";
    } else {
        std::cout << "CUDA no está disponible para LibTorch.\n";
    }
    std::cout << "Build with CUDA: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--camera" || arg == "-c") use_tcp = false;
        if (arg == "--yolo" || arg == "-y") use_yolo = true;
        if (arg == "--midasv21" || arg == "-m21") model = 0;
        if (arg == "--depthany" || arg == "-da") model = 1;
        if (arg == "--nobench" || arg == "-nb") do_bench = false;
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ./obs_avoid_full_cpp_mt [PARAMS]\n";
            std::cout << "\t -c,  --camera\t If selected PC camera will be selected\n";
            std::cout << "\t -y,  --yolo\t If selected YOLOv11n will be run in parallel\n";
            std::cout << "\t -m21, --midasv21\t Model: Midas v21 small\n";
            std::cout << "\t -da, --depthany\t Model: Depth anything v2 outdoor dynamic\n";
            std::cout << "\t -nb, --nobench\t Dont do benchmarking, acelerating therefore multitreadhing capabilities\n";
            return 0;
        }
    }

    DepthModel depth_model;
    DepthEstimationFn depth_fn;

    switch (model) {
        case 0:
            std::cout << "Model: Midas v21 small" << std::endl;
            depth_model = cv::dnn::readNetFromONNX("../models/midas/model-small.onnx");
            std::get<cv::dnn::Net>(depth_model).setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            std::get<cv::dnn::Net>(depth_model).setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            // std::get<cv::dnn::Net>(depth_model).setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            // std::get<cv::dnn::Net>(depth_model).setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            break;
        case 1:
            std::cout << "Model: Depth anything v2 outdoor dynamic" << std::endl;
            depth_model = torch::jit::load("../models/depth_anything/depth_anything_v2_vits_traced.pt");
            {
                auto& model_ref = std::get<torch::jit::script::Module>(depth_model);
                torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
                model_ref.to(device);
                model_ref.eval();
            }
            break;
        default:
            std::cerr << "Unknown model, only Midas v21 and depth anything can be used" << std::endl;
            return -1;
    }

    depth_fn = estimate_depth_variant;

    YOLOv11 yolo_model("../models/yolo/yolo11n.onnx", 0.45f, 0.45f, [](int id, const std::string&) {
        return id == 41;
    });
    auto obj_sizes = load_object_sizes("../models/yolo/object_sizes.txt");

    int sock = -1;
    cv::VideoCapture cap;
    if (use_tcp) {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in server{};
        server.sin_family = AF_INET;
        server.sin_port = htons(SERVER_PORT);
        inet_pton(AF_INET, SERVER_IP, &server.sin_addr);
        while (connect(sock, (sockaddr*)&server, sizeof(server)) < 0) {
            std::cout << "\033[2J\033[1;1H";
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            std::cout << "Connecting to camera..." << std::endl;
        }
    } else {
        cap.open(0);
    }

    std::thread t_capture(frame_capture_thread, use_tcp, sock, std::ref(cap));
    std::thread t_depth(depth_thread, std::ref(depth_model), depth_fn);
    std::thread t_yolo;
    if (use_yolo)
        t_yolo = std::thread(yolo_thread, std::ref(yolo_model));
    uint64_t frames = 0;
    std::deque<double> times_ms;

    while (true) {
        // Clonar el frame actual
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            frame = shared_frame.clone();
        }

        // Lectura protegida de resultados
        cv::Mat local_depth;
        {
            std::lock_guard<std::mutex> lock(depth_mutex);
            local_depth = current_depth.clone();
        }

        std::vector<ObjectBBox> local_detections;
        if (use_yolo) {
            std::lock_guard<std::mutex> lock(yolo_mutex);
            local_detections = current_detections;
        }

        static cv::Mat future;
        cv::Mat depth_vis;

        // Procesamiento de profundidad
        if (!local_depth.empty() && local_depth.cols > 0 && local_depth.rows > 0) {
            if (model == 0) {
                future = exponential_smoothing(local_depth, future, 0.5);
            }else {
                future = local_depth.clone();
            }

            cv::Mat smoothed_normalized = normalize_depth_with_percentile(future);
            cv::Mat depth_8u;
            smoothed_normalized.convertTo(depth_8u, CV_8U, 255.0);
            cv::applyColorMap(depth_8u, depth_vis, cv::COLORMAP_MAGMA);
            draw_mean_slope_arrow_sobel(depth_vis, smoothed_normalized);
            annotate_depth_points(depth_vis, smoothed_normalized);

        }

        // Anotación con YOLO + profundidad
        bool depth_scaled = false;
        if (use_yolo && !frame.empty() && !local_detections.empty() && !local_depth.empty()) {
            annotate_with_depth(frame, local_depth, local_detections, obj_sizes, depth_scaled);
        }

        // Mostrar imágenes
        if (use_yolo && !frame.empty() && !local_depth.empty()) {
            cv::imshow("YOLO + depth", frame);
        }
        if (!depth_vis.empty() && !local_depth.empty()) {
            cv::imshow("Depth", depth_vis);
        }

        // Callback de mouse y depuración
        cv::setMouseCallback("Depth", on_mouse_depth);
        depth_scaled_for_debug = local_depth.clone();
        if (cv::waitKey(1) == 27) break;
#ifdef PROF
        if (frames >= 600) break;
        frames++;
#endif
    }


    stop_flag = true;
    t_capture.join();
    t_depth.join();
    if (use_yolo) t_yolo.join();

    if (use_tcp && sock >= 0) close(sock);
    if (!use_tcp) cap.release();
    cv::destroyAllWindows();
#ifdef PROF
    Instrumentor::Get().EndSession(); 
#endif
    return 0;
}
