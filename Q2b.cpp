#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>  // Include for measuring execution time

#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << "Error: " << msg << " (Error Code: " << err << ")\n"; \
        exit(EXIT_FAILURE); \
    }

std::string loadKernelSource(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

int main() {
    std::string imagePath = "flickr_cat_000016.png";

    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not load image at path: " << imagePath << std::endl;
        return -1;
    }
    
    int width = inputImage.cols;
    int height = inputImage.rows;
    std::vector<float> inputData(inputImage.begin<uchar>(), inputImage.end<uchar>());

    // Filter (3x3 Gaussian Blur Example)
    const int filterSize = 3;
    float filter[filterSize * filterSize] = {
        1.0f / 16, 2.0f / 16, 1.0f / 16,
        2.0f / 16, 4.0f / 16, 2.0f / 16,
        1.0f / 16, 2.0f / 16, 1.0f / 16
    };
    std::vector<float> outputData(width * height, 0);

    // OpenCL Initialization
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err, "clGetPlatformIDs");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err, "clGetDeviceIDs");
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err, "clCreateContext");
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    CHECK_ERROR(err, "clCreateCommandQueue");

    // Load Kernel
    std::string kernelSource = loadKernelSource("convolution.cl");
    const char *sourceStr = kernelSource.c_str();
    size_t sourceSize = kernelSource.length();
    cl_program program = clCreateProgramWithSource(context, 1, &sourceStr, &sourceSize, &err);
    CHECK_ERROR(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), NULL);
        std::cerr << "Kernel Compilation Failed:\n" << buildLog.data() << std::endl;
        return -1;
    }
    
    cl_kernel kernel = clCreateKernel(program, "convolution", &err);
    CHECK_ERROR(err, "clCreateKernel");

    // Memory Buffers
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                        sizeof(float) * width * height, inputData.data(), &err);
    CHECK_ERROR(err, "clCreateBuffer input");
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                         sizeof(float) * width * height, NULL, &err);
    CHECK_ERROR(err, "clCreateBuffer output");
    cl_mem filterBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                         sizeof(float) * filterSize * filterSize, filter, &err);
    CHECK_ERROR(err, "clCreateBuffer filter");

    // Set Kernel Arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(int), &width);
    clSetKernelArg(kernel, 3, sizeof(int), &height);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &filterBuffer);
    clSetKernelArg(kernel, 5, sizeof(int), &filterSize);

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Execute Kernel
    size_t globalSize[2] = { static_cast<size_t>(width), static_cast<size_t>(height) };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    CHECK_ERROR(err, "clEnqueueNDRangeKernel");
    clFinish(queue);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> executionTime = end - start;
    std::cout << "Kernel Execution Time: " << executionTime.count() << " seconds" << std::endl;

    // Read Output Data
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, 
                              sizeof(float) * width * height, outputData.data(), 0, NULL, NULL);
    CHECK_ERROR(err, "clEnqueueReadBuffer");

    // Save Processed Image
    cv::Mat outputImage(height, width, CV_32F, outputData.data());
    cv::imwrite("output.jpg", outputImage);
    std::cout << "Output image saved successfully!" << std::endl;

    // Cleanup
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(filterBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

