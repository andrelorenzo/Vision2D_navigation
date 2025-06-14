#include "helper.hpp"

void on_mouse_depth(int event, int x, int y, int, void*) {
    if (!depth_scaled_for_debug.empty()) {
        if (x >= 0 && x < depth_scaled_for_debug.cols &&
            y >= 0 && y < depth_scaled_for_debug.rows) {
            float value = depth_scaled_for_debug.at<float>(y, x);
            system("cls");
            std::cout << "Depth at (" << x << ", " << y << ") = "
                      << std::fixed << std::setprecision(2) << value << " meters" << std::endl;
        }
    }
}

void frame_capture_thread(bool use_tcp, SOCKET sock, cv::VideoCapture& cap) {
    while (!stop_flag.load()) {
        cv::Mat frame = use_tcp ? get_frame_from_tcp(sock) : get_frame_from_camera(cap);
        if (frame.empty()) continue;
        {
            std::lock_guard<std::mutex> guard(frame_mutex);
            shared_frame = frame.clone();
        }
        midas_ready.store(true);
        yolo_ready.store(true);
    }
}


void depth_thread(DepthModel& depth_model, DepthEstimationFn estimate_fn) {
    static uint64_t last_frame_index = 0;

    // Benchmarking variables
    std::deque<double> times_ms;
    while (!stop_flag.load()) {

        if(!midas_ready.load()){
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        midas_ready.store(false);

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

        // Benchmarking
        if (do_bench) {
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            times_ms.push_back(elapsed_ms);
            if (times_ms.size() > 10) times_ms.pop_front();
            double avg_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
            double fps = 1000.0 / avg_ms;

            std::cout << std::fixed << std::setprecision(2);
            system("cls");
            std::cout << "[Benchmark - Depth] Avg frame time (10): " << avg_ms << " ms | FPS: " << fps << std::endl;
        }
    }
}


void yolo_thread(YOLOv11& yolo_model) {

    while (!stop_flag.load()) {
        if(!yolo_ready.load()){
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        yolo_ready.store(false);
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


int parser(int argc, char ** argv, bool &use_yolo, bool &use_tcp, int &model, DepthModel &depth_model, DepthEstimationFn &depth_fn){
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
            return -1;
        }
    }


    switch (model) {
        case 0:
            std::cout << "Model: Midas v21 small" << std::endl;
            depth_model = cv::dnn::readNetFromONNX("C:/Users/ajlorenzo/Documents/vsc/Vision2D_navigation/Codigos_PC/models/midas/model-small.onnx");
            std::get<cv::dnn::Net>(depth_model).setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            std::get<cv::dnn::Net>(depth_model).setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            // std::get<cv::dnn::Net>(depth_model).setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            // std::get<cv::dnn::Net>(depth_model).setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            break;
        case 1:
            std::cout << "Model: Depth anything v2 outdoor dynamic" << std::endl;
            depth_model = torch::jit::load("C:/Users/ajlorenzo/Documents/vsc/Vision2D_navigation/Codigos_PC/models/depth_anything/depth_anything_v2_vits_traced.pt");
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
    return 0;
}

int main(int argc, char** argv) {
#ifdef PROF
    Instrumentor::Get().BeginSession("both_cpu","../profiling/mgpu_ycpu.json");
#endif
    bool use_tcp = true;
    bool use_yolo = false;
    int model = 0;
    DepthModel depth_model;
    DepthEstimationFn depth_fn;

    if(parser(argc, argv, use_yolo,use_tcp, model, depth_model, depth_fn) != 0)return -1;
    
    YOLOv11 yolo_model("C:/Users/ajlorenzo/Documents/vsc/Vision2D_navigation/Codigos_PC/models/yolo/yolo11n.onnx", 0.45f, 0.45f, [](int id, const std::string&) {
        return id == 41;
    });
    auto obj_sizes = load_object_sizes("C:/Users/ajlorenzo/Documents/vsc/Vision2D_navigation/Codigos_PC/models/yolo/object_sizes.txt");

    SOCKET sock = INVALID_SOCKET;  // Correcto para Windows
    cv::VideoCapture cap;

    if (use_tcp) {
        // 1. Inicialización Winsock (debe estar al principio del uso de sockets)
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "WSAStartup failed.\n";
            return -1;
        }

        // 2. Crear socket (retorna SOCKET, no int)
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == INVALID_SOCKET) {
            std::cerr << "Socket creation failed.\n";
            WSACleanup();  // Limpieza por error temprano
            return -1;
        }

        sockaddr_in server{};
        server.sin_family = AF_INET;
        server.sin_port = htons(SERVER_PORT);

        // 3. Convertir IP (inet_pton solo está disponible desde Vista)
        if (inet_pton(AF_INET, SERVER_IP, &server.sin_addr) <= 0) {
            std::cerr << "Invalid address or address not supported.\n";
            WSACleanup();
            return -1;
        }

        // 4. Conexión en bucle (puede usarse sin cambios)
        while (connect(sock, (sockaddr*)&server, sizeof(server)) == SOCKET_ERROR) {
            system("cls");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            std::cout << "Connecting to camera..." << std::endl;
        }
    } else {
        cap.open(0);
        // cap.open("rtsp://admin:admin123@192.168.0.10:554/live0.265");
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
        cv::Vec2f dir(0.0f,0.0f);
        cv::Mat depth_8u, depth_colored, depth_real;

        // Procesamiento de profundidad
        if (!local_depth.empty() && local_depth.cols > 0 && local_depth.rows > 0) {
            cv::Mat depth_normalized = normalize_depth_with_percentile(local_depth);

            if (model == 0) {
                future = exponential_smoothing(depth_normalized, future, 0.10);
            } else {
                future = local_depth.clone();  // sin normalizar para otros modelos
            }
            // future = local_depth.clone();

            if (use_yolo && !frame.empty() && !local_detections.empty()) {
                annotate_with_depth(frame, local_detections, obj_sizes);
                calibrate_and_scale_midas(future, local_detections, obj_sizes, depth_real);
            } else {
                depth_real = future.clone();  // por si no se calibra, pasar algo válido
            }

            // Visualización con color
            depth_normalized.convertTo(depth_8u, CV_8U, 255.0);  // ojo: usa depth_normalized, no future
            cv::applyColorMap(depth_8u, depth_colored, cv::COLORMAP_MAGMA);

            // Anotación visual
            if (!depth_colored.empty() && !depth_real.empty()) {
                annotate_depth_points(depth_colored, depth_real);  // ambos del mismo tamaño
            }
            // dir = draw_mean_slope_arrow_sobel(depth_colored, future);  // visual, real

        }

        
        // Mostrar imágenes
        if (use_yolo && !frame.empty() && !local_depth.empty()) {
            cv::imshow("YOLO + depth", frame);
        }
        if (!depth_colored.empty()) {
            cv::imshow("Depth", depth_colored);
        }

        // Callback de mouse y depuración
        // cv::setMouseCallback("Depth", on_mouse_depth);
        depth_scaled_for_debug = depth_real.clone();
        if (cv::waitKey(1) == 27) break;
#ifdef PROF
        if (frames >= 600) break;
        frames++;
#endif
    }

    stop_flag.store(true);
    t_capture.join();
    t_depth.join();
    if (use_yolo) t_yolo.join();
    if (use_tcp && sock != INVALID_SOCKET) {
        closesocket(sock);
        WSACleanup();
    }
    if (!use_tcp) cap.release();
    cv::destroyAllWindows();
#ifdef PROF
    Instrumentor::Get().EndSession(); 
#endif
    return 0;
}
