// Fragmento de código con buffer de sincronización controlado
#include "aux.hpp"

using DepthEstimationFn = std::function<cv::Mat(torch::jit::script::Module& model, const cv::Mat& frame)>;

void frame_capture_thread(bool use_tcp, int sock, cv::VideoCapture& cap) {
    while (!stop_flag) {
        std::unique_lock<std::mutex> lock(frame_mutex);
        frame_cv.wait(lock, [] { return !buffer_full || stop_flag; });
        if (stop_flag) break;

        lock.unlock();

        cv::Mat frame = use_tcp ? get_frame_from_tcp(sock) : get_frame_from_camera(cap);
        if (frame.empty()) continue;

        int idx = frame_index.load();
        buffer_depth[idx] = frame.clone();
        buffer_yolo[idx] = frame.clone();
        frame_index.store(idx ^ 1);  // alterna entre 0 y 1

        {
            std::lock_guard<std::mutex> guard(frame_mutex);
            new_frame_ready = true;
            buffer_full = true;
            threads_done = 0;
        }
        frame_cv.notify_all();
    }
}

void depth_thread(torch::jit::script::Module& depth_net, DepthEstimationFn estimate_fn) {
    while (!stop_flag) {
        std::unique_lock<std::mutex> lock(frame_mutex);
        frame_cv.wait(lock, [] { return new_frame_ready || stop_flag; });
        if (stop_flag) break;
        lock.unlock();

        int idx = frame_index.load() ^ 1;
        cv::Mat frame = buffer_depth[idx].clone(); 

        current_depth = estimate_fn(depth_net, frame);

        {
            std::lock_guard<std::mutex> guard(frame_mutex);
            buffer_full = false;
        }
        frame_cv.notify_all();

        signal_thread_done();
    }
}

void yolo_thread(YOLOv11& yolo_model) {
    while (!stop_flag) {
        std::unique_lock<std::mutex> lock(frame_mutex);
        frame_cv.wait(lock, [] { return new_frame_ready || stop_flag; });
        if (stop_flag) break;
        lock.unlock();

        int idx = frame_index.load() ^ 1;
        cv::Mat frame = buffer_yolo[idx].clone();   // en yolo_thread
        

        current_detections = yolo_model.detect(frame);

        signal_thread_done();
    }
}

int main(int argc, char** argv) {
    bool use_tcp = true;
    bool use_yolo = false;
    int model = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--camera" || arg == "-c") use_tcp = false;
        if (arg == "--yolo" || arg == "-y") use_yolo = true;
        if (arg == "--midasv21" || arg == "-m21") model = 0;
        if (arg == "--depthany" || arg == "-da") model = 1;
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ./obs_avoid_full_cpp_mt [PARAMS]\n";
            std::cout << "\t -c,  --camera\t If selected PC camera will be selected\n";
            std::cout << "\t -y,  --yolo\t If selected YOLOv11n will be run in parallel\n";
            std::cout << "\t -m21, --midasv21\t Model: Midas v21 small\n";
            std::cout << "\t -da, --depthany\t Model: Depth anything v2 outdoor dynamic\n";
            return 0;
        }
    }

    torch::jit::script::Module depth_model;
    DepthEstimationFn depth_fn;

    switch (model) {
        case 0:
            std::cout << "Model: Midas v21 small" << std::endl;
            depth_model = torch::jit::load("../models/model-small-traced.pt");
            depth_fn = estimate_midas_depth_v21;
            break;
        case 1:
            std::cout << "Model: Depth anything v2 outdoor dynamic" << std::endl;
            try {
                depth_model = torch::jit::load("../models/depth_anything_v2_vits_traced.pt");
            } catch (const c10::Error& e) {
                std::cerr << "Error al cargar el modelo\n";
                return -1;
            }
            depth_fn = estimate_depth_anything_v2_outdoor;
            break;
        default:
            std::cerr << "Modelo desconocido" << std::endl;
            return -1;
    }

    YOLOv11 yolo_model("../models/yolo11n.onnx", 0.45f, 0.45f, [](int id, const std::string&) {
        return id >= 0 && id <= 16;
    });

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
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
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

    while (true) {
        static std::deque<double> times_ms;
        auto t_start = std::chrono::high_resolution_clock::now();
        wait_for_threads(use_yolo ? 2 : 1);

        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            int idx = frame_index.load() ^ 1;
            frame = buffer_depth[idx].clone();
            new_frame_ready = false;
        }

        if (use_yolo && !frame.empty() && !current_detections.empty() && !current_depth.empty())
            annotate_with_depth(frame, current_depth, current_detections);


        if (use_yolo && !frame.empty())
            cv::imshow("YOLO + depth", frame);

        if (!current_depth.empty() && current_depth.cols > 0 && current_depth.rows > 0) {
            cv::Mat depth_vis;
            cv::normalize(current_depth, depth_vis, 0, 255, cv::NORM_MINMAX);
            depth_vis.convertTo(depth_vis, CV_8U);
            cv::applyColorMap(depth_vis, depth_vis, cv::COLORMAP_MAGMA);
            draw_mean_slope_arrow_sobel(depth_vis, current_depth);
            cv::imshow("Depth", depth_vis);
        }

        if (cv::waitKey(1) == 27) break;

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        times_ms.push_back(elapsed_ms);
        if (times_ms.size() > 10) times_ms.pop_front();
        double avg_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
        double fps = 1000.0 / avg_ms;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[Benchmark] Avg frame time (10): " << avg_ms << " ms | FPS: " << fps << std::endl;
        std::cout << "\033[2J\033[1;1H";
    }

    stop_flag = true;
    frame_cv.notify_all();
    t_capture.join();
    t_depth.join();
    if (use_yolo) t_yolo.join();

    if (use_tcp && sock >= 0) close(sock);
    if (!use_tcp) cap.release();
    cv::destroyAllWindows();
    return 0;
}
