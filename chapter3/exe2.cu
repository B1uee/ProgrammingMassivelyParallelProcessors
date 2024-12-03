#include <iostream>
#include <cuda_runtime.h>
using namespace std;

/*
A matrix-vector multiplication takes an input matrix B and a vector C and produces one output vector A. 
Each element of the output vector A is the dot product of one row of the input matrix B and C, that is, A[i] = Σ_j (B[i][j]*C[j]). 
For simplicity we will handle only square matrices whose elements are singleprecision floating-point numbers. 
Write a matrix-vector multiplication kernel and the host stub function that can be called with four parameters: 
pointer to the output matrix, pointer to the input matrix, pointer to the input vector, and the number of elements in each dimension. 
Use one thread to calculate an output vector element.
*/

__global__
void func(float* B, float* C, float* A, int width){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < width){
        for(int k = 0; k < width; ++k){
            A[row] += B[row*width + k] * C[k];
        }
        printf("Thread: %d, row: %d, element: %f\n", row, row, A[row]);
    }

}


int main(){
    // A  = B * C
    int width = 4;
    int size = width * width;
    float *B_h = new float[size], *C_h = new float[width], *A_h = new float[width]; 
    float *B_d, *C_d, *A_d;
    for (int i = 0; i < size; ++i) {
        B_h[i] = static_cast<float>(i);
        if(i < width)   C_h[i] = static_cast<float>(i);
    }

    cudaMalloc(&B_d, size * sizeof(float)), cudaMalloc(&C_d, width * sizeof(float)), cudaMalloc(&A_d, width * sizeof(float));

    cudaMemcpy(B_d, B_h, size * sizeof(float), cudaMemcpyHostToDevice), cudaMemcpy(C_d, C_h, width * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 4;
    int numBlocks = static_cast<int>(ceil(width / static_cast<float>(threadsPerBlock)));
    dim3 dimGrid(numBlocks, 1, 1), dimBlock(threadsPerBlock, 1, 1);

    func<<<dimGrid, dimBlock>>>(B_d, C_d, A_d, width);

    cudaMemcpy(A_h, A_d, width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(B_d), cudaFree(C_d), cudaFree(A_d);
    delete[] B_h, C_h, A_h;

    return 0;
    
    
}

