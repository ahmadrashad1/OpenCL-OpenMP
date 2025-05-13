#define CL_TARGET_OPENCL_VERSION 120  // For OpenCL 1.2, change as needed
#include <CL/cl.h>
#include <iostream>

int main() {
    // Initialize OpenCL platform and device
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    std::cout << "OpenCL platform and device initialized successfully!" << std::endl;
    return 0;
}

