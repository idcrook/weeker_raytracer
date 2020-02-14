#include "imgui.h"
#include "bindings/imgui_impl_glfw.h"
#include "bindings/imgui_impl_opengl3.h"

#include <iostream>
#include <stdexcept>
#include <chrono>

#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
#include <GL/gl3w.h>  // Initialize with gl3wInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
#include <GL/glew.h>  // Initialize with glewInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
#include <glad/glad.h>  // Initialize with gladLoadGL()
#else
#include IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#endif

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>


// Image I/O
// define STB_IMAGE*_IMPLEMENTATION-s only once (e.g. in .cpp file)
#define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION // not yet used
#include "../external/rtw_stb_image.h"

// // optix
// #include <optix.h>
// #include <optixu/optixpp.h>

// Executive Director
#include "Director.h"
// Parse command line arguments and options
#include "InputParser.h"

#define Nx_MIN  (320)
#define Ny_MIN  (200)
// set maximum resolution to standard 4K dimensions
#define Nx_MAX  (3840)
#define Ny_MAX  (2240)  // was 2160, but increased so that square resolutions could hit 2240
#define Nscene_MAX  (4)   // Range [0 .. Nscene_MAX]
#define Ns_MAX  (1024*10)


static void glfw_error_callback(int error, const char *description) {
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}


void render_conan_logo()
{
	ImDrawList *draw_list = ImGui::GetWindowDrawList();
	float sz = 300.0f;
	static ImVec4 col1 = ImVec4(68.0 / 255.0, 83.0 / 255.0, 89.0 / 255.0, 1.0f);
	static ImVec4 col2 = ImVec4(40.0 / 255.0, 60.0 / 255.0, 80.0 / 255.0, 1.0f);
	static ImVec4 col3 = ImVec4(50.0 / 255.0, 65.0 / 255.0, 82.0 / 255.0, 1.0f);
	static ImVec4 col4 = ImVec4(20.0 / 255.0, 40.0 / 255.0, 60.0 / 255.0, 1.0f);
	const ImVec2 p = ImGui::GetCursorScreenPos();
	float x = p.x + 4.0f, y = p.y + 4.0f;
	draw_list->AddQuadFilled(ImVec2(x, y + 0.25 * sz), ImVec2(x + 0.5 * sz, y + 0.5 * sz), ImVec2(x + sz, y + 0.25 * sz), ImVec2(x + 0.5 * sz, y), ImColor(col1));
	draw_list->AddQuadFilled(ImVec2(x, y + 0.25 * sz), ImVec2(x + 0.5 * sz, y + 0.5 * sz), ImVec2(x + 0.5 * sz, y + 1.0 * sz), ImVec2(x, y + 0.75 * sz), ImColor(col2));
	draw_list->AddQuadFilled(ImVec2(x + 0.5 * sz, y + 0.5 * sz), ImVec2(x + sz, y + 0.25 * sz), ImVec2(x + sz, y + 0.75 * sz), ImVec2(x + 0.5 * sz, y + 1.0 * sz), ImColor(col3));
	draw_list->AddLine(ImVec2(x + 0.75 * sz, y + 0.375 * sz), ImVec2(x + 0.75 * sz, y + 0.875 * sz), ImColor(col4));
    draw_list->AddBezierCurve(ImVec2(x + 0.72 * sz, y + 0.24 * sz), ImVec2(x + 0.68 * sz, y + 0.15 * sz), ImVec2(x + 0.48 * sz, y + 0.13 * sz), ImVec2(x + 0.39 * sz, y + 0.17 * sz), ImColor(col4), 10, 18);
    draw_list->AddBezierCurve(ImVec2(x + 0.39 * sz, y + 0.17 * sz), ImVec2(x + 0.2 * sz, y + 0.25 * sz), ImVec2(x + 0.3 * sz, y + 0.35 * sz), ImVec2(x + 0.49 * sz, y + 0.38 * sz), ImColor(col4), 10, 18);
}


int main(int argc, char* argv[])
{
    int exit_code = EXIT_SUCCESS;


    // default values
    int Nx = 1200;
    int Ny = 600;
    int Nscene = 0;
    int Ns = 1024;
    bool Qverbose = false;
    bool Qdebug = false;

    InputParser cl_input(argc, argv);
    if(cl_input.cmdOptionExists("-h") or cl_input.cmdOptionExists("--help")) {
        std::cerr << std::endl <<  " HELP - " << argv[0] << std::endl;
        std::cerr << R"(
    -s N           Scene Selection number N (N: 0, 1, 2, etc.)
    -ns N          Sample each pixel N times (N: 1, 2, etc.)
    -dx Nx         Output image width (x dimension)
    -dy Ny         Output image height (y dimension)

    -h             This help message.
    -v             Verbose output.
    -g             Debug output.

)";
        std::exit( exit_code );
    }

    if(cl_input.cmdOptionExists("-v"))
        Qverbose = true;

    if(cl_input.cmdOptionExists("-g"))
        Qdebug = true;

    const std::string &sceneNumber = cl_input.getCmdOption("-s");
    try {
        if (!sceneNumber.empty()){
            std::size_t pos;
            int x = std::stoi(sceneNumber, &pos);
            if (x >= Nscene and x <= Nscene_MAX) {
                Nscene = x;
            } else {
                std::cerr << "WARNING: Scene number " << x << " out of range. Maximum scene number: " << Nscene_MAX << std::endl;
                std::cerr << "WARNING: Using a scene value of " << Nscene << std::endl;
            }
        }
    } catch (std::invalid_argument const &ex) {
        std::cerr << "Invalid scene number: " << sceneNumber << std::endl;
    }

    const std::string &numberOfSamples = cl_input.getCmdOption("-ns");
    try {
        if (!numberOfSamples.empty()){
            std::size_t pos;
            int x = std::stoi(numberOfSamples, &pos);
            if ( (x > 0) && (x <= Ns_MAX))  {
                Ns = x;
            } else {
                std::cerr << "WARNING: Number of samples " << x << " is out of range. ";
                if (x > Ns_MAX) {
                    Ns = Ns_MAX;
                }
                std::cerr << "Using a value of " << Ns << std::endl;
            }
        }
    } catch (std::invalid_argument const &ex) {
        std::cerr << "Invalid number of samples: " << numberOfSamples << std::endl;
    }

    const std::string &dimWidth = cl_input.getCmdOption("-dx");
    try {
        if (!dimWidth.empty()){
            std::size_t pos;
            int x = std::stoi(dimWidth, &pos);
            // std::cerr << pos << std::endl;
            if (x >= Nx_MIN and x <= Nx_MAX) {
                Nx = x;
            } else {
                std::cerr << "WARNING: Width (-dx) " << x << " out of range. ";
                if (x > Nx_MAX) {
                    Nx = Nx_MAX;
                }
                if (x < Nx_MIN) {
                    Nx = Nx_MIN;
                }
                std::cerr << "Using a value of " << Nx <<std::endl;
            }
        }
    } catch (std::invalid_argument const &ex) {
        std::cerr << "Invalid image width (-dx): " << dimWidth << std::endl;
    }

    const std::string &dimHeight = cl_input.getCmdOption("-dy");
    try {
        if (!dimHeight.empty())
        {
            std::size_t pos;
            int x = std::stoi(dimHeight, &pos);
            // std::cerr << pos << std::endl;
            if (x >= Ny_MIN and x <= Ny_MAX)
            {
                Ny = x;
            }
            else
            {
                std::cerr << "WARNING: Width (-dy) " << x << " out of range. ";
                if (x > Ny_MAX) {
                    Ny = Ny_MAX;
                }
                if (x < Ny_MIN) {
                    Ny = Ny_MIN;
                }
                std::cerr << "Using a value of " << Ny <<std::endl;
            }
        }
    } catch (std::invalid_argument const &ex) {
        std::cerr << "Invalid image height (-dy): " << dimHeight << std::endl;
    }


    // Setup window
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		std::exit ( EXIT_FAILURE) ;

    // Decide GL+GLSL versions
#if __APPLE__
	// GL 3.2 + GLSL 150
	const char *glsl_version = "#version 150";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);		   // Required on Mac
#else
	// GL 3.0 + GLSL 130
	const char *glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

	// Create window with graphics context
	GLFWwindow *window = glfwCreateWindow(1280, 720, "Optix Rayocaster", NULL, NULL);
	if (window == NULL)
        std:: exit ( EXIT_FAILURE);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // Enable vsync

	bool err = glewInit() != GLEW_OK;

	if (err)
	{
		fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        std:: exit ( EXIT_FAILURE);
	}

	int screen_width, screen_height;
	glfwGetFramebufferSize(window, &screen_width, &screen_height);
	glViewport(0, 0, screen_width, screen_height);


    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);


	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
		glClear(GL_COLOR_BUFFER_BIT);

		// feed inputs to dear imgui, start new frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();


        ImGui::Begin("Conan logo");
        render_conan_logo();
        ImGui::End();
		// Render dear imgui into screen
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glfwSwapBuffers(window);
	}

	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();

        // Director optixSingleton = Director(Qverbose, Qdebug);

        // auto start = std::chrono::system_clock::now();
        // optixSingleton.init(Nx, Ny, Ns);

        // if (Qverbose) {
        //     std::cerr << "INFO: Output image dimensions: " << Nx << 'x' << Ny << std::endl;
        //     std::cerr << "INFO: Number of rays sent per pixel: " << Ns << std::endl;
        //     std::cerr << "INFO: Scene number selected: " << Nscene << std::endl;
        // }
        // optixSingleton.createScene(Nscene);

        // optixSingleton.renderFrame();
        // auto stop = std::chrono::system_clock::now();
        // auto time_seconds = std::chrono::duration<float>(stop - start).count();
        // std::cerr << "INFO: Took " << time_seconds << " seconds." << std::endl;

        // optixSingleton.printPPM();

        // optixSingleton.destroy();

    std::exit ( EXIT_SUCCESS);
}
