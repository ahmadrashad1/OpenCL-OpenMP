#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>  // For execution time measurement

#define K 3  // Kernel size

using namespace std;
using namespace cv;
using namespace chrono;

// Vertical edge detection kernel
float kernel[K][K] = {
    {1, 0, -1},
    {1, 0, -1},
    {1, 0, -1}
};

Mat applyConvolution(const Mat &image) {
    int M = image.rows, N = image.cols;
    Mat output = Mat::zeros(M, N, CV_32F);  

    int pad = K / 2;  

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;

            for (int ki = -pad; ki <= pad; ki++) {
                for (int kj = -pad; kj <= pad; kj++) {
                    int ni = i + ki, nj = j + kj;

                    if (ni >= 0 && ni < M && nj >= 0 && nj < N) {
                        sum += image.at<uchar>(ni, nj) * kernel[ki + pad][kj + pad];
                    }
                }
            }

            output.at<float>(i, j) = sum;
        }
    }

    return output;
}

int main() {
    
    string imagePath = "flickr_cat_000016.png";
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);

    if (image.empty()) {
        cout << "Error: Could not load image from " << imagePath << "\n";
        return -1;
    }

    auto start = high_resolution_clock::now();
    
    Mat output = applyConvolution(image);  

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    normalize(output, output, 0, 255, NORM_MINMAX);
    output.convertTo(output, CV_8U);

    string outputPath = "C:\\Users\\Dell\\Desktop\\22i-1175_F_A3\\edge_detected.png";
    imwrite(outputPath, output);

    imshow("Original Image", image);
    imshow("Edge Detection Result", output);

    cout << "Edge detection completed. Output saved at: " << outputPath << endl;
    cout << "Execution time: " << duration.count() << " ms" << endl;

    waitKey(0); 
    return 0;
}

