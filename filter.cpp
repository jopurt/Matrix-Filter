#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <cstring>
#include <omp.h>
//#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define M_PI 3.14159265358979323846

using namespace std;

// Функция для генерации Гауссового ядра
void generate_gaussian_kernel(float* kernel, int n, float sigma) {
    float sum = 0.0f;
    int half = n / 2;

    for (int y = -half; y <= half; ++y) {
        for (int x = -half; x <= half; ++x) {
            float value = (1.0f / (2.0f * M_PI * sigma * sigma)) * pow(exp(1), (-(x * x + y * y) / (2.0f * sigma * sigma)));
            //cout << value << endl;
            kernel[(y + half) * n + (x + half)] = value;
            sum += value;
        }
    }
    cout << "Sum:" << sum << endl;

    // Нормализация ядра
    //cout << "Normalization" << endl;
    for (int i = 0; i < n * n; ++i) {
        kernel[i] /= sum;
        //cout << kernel[i] << endl;
    }

}

// Функция для применения фильтра (последовательно)
void apply_convolution_seq(unsigned char* input, unsigned char* output, int width, int height, int channels, float* kernel, int kernel_size) {
    int half = kernel_size / 2;

//#pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum[3] = { 0.0f, 0.0f, 0.0f };

            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    int ny = min(max(y + ky, 0), height - 1);
                    int nx = min(max(x + kx, 0), width - 1);

                    for (int c = 0; c < channels; ++c) {
                        sum[c] += input[(ny * width + nx) * channels + c] *
                            kernel[(ky + half) * kernel_size + (kx + half)];
                    }
                }
            }

            for (int c = 0; c < channels; ++c) {
                output[(y * width + x) * channels + c] = (unsigned char)min(255.0f, max(0.0f, sum[c]));
            }
        }
    }
}

// Функция для применения фильтра (параллельно)
void apply_convolution_paral(unsigned char* input, unsigned char* output, int width, int height, int channels, float* kernel, int kernel_size) {
    int half = kernel_size / 2;

#pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum[3] = { 0.0f, 0.0f, 0.0f };

            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    int ny = min(max(y + ky, 0), height - 1);
                    int nx = min(max(x + kx, 0), width - 1);

                    for (int c = 0; c < channels; ++c) {
                        sum[c] += input[(ny * width + nx) * channels + c] *
                            kernel[(ky + half) * kernel_size + (kx + half)];
                    }
                }
            }

            for (int c = 0; c < channels; ++c) {
                output[(y * width + x) * channels + c] = (unsigned char)min(255.0f, max(0.0f, sum[c]));
            }
        }
    }

}

int main(int argc, char* argv[]) {

    const char* input_file = "test5.jpg";
    const char* output_file = "blurred_image5.png";
    int kernel_size = 5;
    float sigma = 0.8;

    int width, height, channels;
    unsigned char* image_data = stbi_load(input_file, &width, &height, &channels, 0);

    if (!image_data) {
        cout << "Failed to load image!" << endl;
        return 1;
    }

    cout << "Loaded image: " << width << "x" << height << " (" << channels << " channels)" << endl;

    vector<unsigned char> output_data(width * height * channels);
    vector<float> kernel(kernel_size * kernel_size);

    generate_gaussian_kernel(kernel.data(), kernel_size, sigma);

    auto start = chrono::high_resolution_clock::now();

    //Seq
    double start_seq = clock();

    apply_convolution_seq(image_data, output_data.data(), width, height, channels, kernel.data(), kernel_size);

    double end_seq = clock();

    //Paral
    double start_paral = omp_get_wtime();

    apply_convolution_paral(image_data, output_data.data(), width, height, channels, kernel.data(), kernel_size);

    double end_paral = omp_get_wtime();

    auto end = chrono::high_resolution_clock::now();

    cout << "Paral:" << end_paral - start_paral << endl;

    cout << "Seq:" << (end_seq - start_seq) / CLOCKS_PER_SEC << endl;


    if (!stbi_write_png(output_file, width, height, channels, output_data.data(), width * channels)) {
        cout << "Failed to save image!" << endl;
    }
    else {
        cout << "Blurred image saved as " << output_file << endl;
    }

    stbi_image_free(image_data);

    return 0;
}
