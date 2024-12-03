#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

using namespace std;

/*
In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. 
In this question, you will implement different matrix-matrix multiplication kernels and compare them. 

a. Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design. 
b. Write a kernel that has each thread produce one output matrix column. Fill in the execution configuration parameters for the design. 
c. Analyze the pros and cons of each of the two kernel designs.
*/

// c. 默认行优先存储，即每行元素连续存储，内存访问效率相交访问列更高，所以 kernel_b的效果更好，每个线程不断访问M的行，而只需要访问一次N的列

// a. Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design. 
__global__ 
void thread_2_row(float* M, float* N, float* P, int width) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < width){
        for (int col = 0; col < width; ++col){
            for (int k = 0; k < width; ++k){
                P[row * width + col] += M[row * width + k] * N[k * width + col];
            }
            printf("Thread: %d, row: %d, col: %d, element: %f\n", row, row, col, P[row * width + col]);
        }
    }
}

// b. Write a kernel that has each thread produce one output matrix column. Fill in the execution configuration parameters for the design. 
__global__ 
void thread_2_col(float* M, float* N, float* P, int width) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col < width){
        for (int row = 0; row < width; ++row){
            for (int k = 0; k < width; ++k){
                P[row * width + col] += M[row * width + k] * N[k * width + col];
            }
            printf("Thread: %d, row: %d, col: %d, element: %f\n", col, row, col, P[row * width + col]);
        }
    }
}

int main() {
    int width = 4; 
    size_t size = width * width;  // 简化为方阵


    float *M_h = new float[size], *N_h = new float[size], *P_h = new float[size];
    
    float *M_d = new float[size], *N_d = new float[size], *P_d = new float[size];
    
    for (int i = 0; i < size; ++i) {
        M_h[i] = static_cast<float>(i), N_h[i] = static_cast<float>(i);
    }

    cudaMalloc(&M_d, size * sizeof(float)), cudaMalloc(&N_d, size * sizeof(float)), cudaMalloc(&P_d, size * sizeof(float));

    cudaMemcpy(M_d, M_h, size * sizeof(float), cudaMemcpyHostToDevice), cudaMemcpy(N_d, N_h, size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 4;
    int numBlocks = static_cast<int>(ceil(width / static_cast<float>(threadsPerBlock)));

    thread_2_row<<<numBlocks, threadsPerBlock>>>(M_d, N_d, P_d, width); // dimGrid(1,1,numBlocks)  dimBlock(1,1,threadsPerBlock>) 
    //thread_2_col<<<numBlocks, threadsPerBlock>>>(M_d, N_d, P_d, width);


    cudaMemcpy(P_h, P_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(M_d), cudaFree(N_d), cudaFree(P_d);
    delete[] M_h, N_h, P_h;

    return 0;
}



