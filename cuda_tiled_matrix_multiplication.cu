/*
  Matrix Multiplication w/GPU using Tiling (cuda)
  Víctor Rendón Suárez
  A01022462
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <chrono>
#include <stdlib.h>
// #include "common.h"

using namespace std;

#define SIZE 2000
#define TS 32

void initialize_matrix(float *matrix, const int n)
{
  for (int i = 0; i < n * n; i++){
    matrix[i] = rand() % 10 + 1;
  }
}

void multiply_matrix_cpu(float *matrixA, float *matrixB, float *result, const int n)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        result[i*n + j] += matrixA[i*n + k] * matrixB[j + k*n];
      }
    }
  }
}

__global__ void multiply_matrix_gpu(float *matrixA, float *matrixB, float *result, int n)
{
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = blockIdx.y;
  if(ix < n && iy < n) {
    long add = 0;
    for (int i = 0; i < n; i++) {
      add += matrixA[iy * n + i] * matrixB[i * n + ix];
    }
    result[iy * n + ix] = add;
  }
}

__global__ void tiling_multiply_matrix_gpu(float *matrixA, float *matrixB, float *result, const int n)
{
  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;
  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;
  unsigned int offset = 0;
  __shared__ float tile_A[TS * TS];
  __shared__ float tile_B[TS * TS];

  float add = 0;
  const int limit = (int)ceil((float)n / TS);
  while (offset < limit) {
    if (TS * offset + tx < n && by * TS + ty < n) {
      tile_A[TS * ty + tx] = matrixA[n * by * TS + n * ty + tx + TS * offset];
    } else {
      tile_A[TS * ty + tx] = 0;
    }
    if (TS * offset + ty < n && bx * TS + tx < n) {
      tile_B[TS * ty + tx] = matrixB[bx * TS + n * ty + tx + n * TS * offset];
    } else {
      tile_B[TS * ty + tx] = 0;
    }
    __syncthreads();
    for (int i = 0; i < TS; i++) {
      add += tile_A[TS * ty + i] * tile_B[tx + TS * i];
    }
    offset += 1;
    __syncthreads();
  }
  if (TS * by + ty < n && TS * bx + tx < n) {
    result[n * (TS * by + ty) + (TS * bx + tx)] = add;
  }
}

int main(int argc, char const *argv[])
{

  srand(time(NULL));
  // Setup device
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  cudaSetDevice(dev);

  // Specify size
  int n = SIZE;
  int bytes = n * n * sizeof(float);
  printf("Matrix dims: %d x %d\n", n, n);

  // Matrix definition
  float *matrixA = (float *) malloc(bytes);
  float *matrixB = (float *) malloc(bytes);
  float *result = (float *) malloc(bytes);
  float *d_matrixA;
  float *d_matrixB;
  float *d_result_matrix;
  memset(result, 0, bytes);

  // Initialize matrices
  initialize_matrix(matrixA, n);
  initialize_matrix(matrixB, n);

  // Multiply matrix on cpu
  auto start_time = std::chrono::high_resolution_clock::now();
  multiply_matrix_cpu(matrixA, matrixB, result, n);
  auto end_time =  std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_time - start_time;
  printf("Matrix multiplication on CPU, time elapsed: %f ms\n", duration_ms.count());

  // Allocate device memory
  cudaMalloc((void **)&d_matrixA, bytes);
  cudaMalloc((void **)&d_matrixB, bytes);
  cudaMalloc((void **)&d_result_matrix, bytes);

  // Transfer data from host to device
  cudaMemcpy(d_matrixA, matrixA, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrixB, matrixB, bytes, cudaMemcpyHostToDevice);
  cudaMemset(d_result_matrix, 0, bytes);

  // Kernel configuration
  dim3 block(TS, TS);
  dim3 grid((int)ceil((float)n / TS), (int)ceil((float)n / TS));
  printf("Tile size: %d <<<(%d,%d), (%d,%d)>>>\n", TS, grid.x, grid.y, block.x, block.y);

  // Multiply the matrices with GPU, measure elapsed time
  start_time = chrono::high_resolution_clock::now();
  multiply_matrix_gpu<<<grid, block>>>(d_matrixA, d_matrixB, d_result_matrix, n);
  cudaDeviceSynchronize();
  end_time = chrono::high_resolution_clock::now();
  duration_ms = end_time - start_time;
  printf("Matrix multiplication on GPU, time elapsed: %f ms\n", duration_ms.count());

  // Copy result to host
  cudaMemcpy(result, d_result_matrix, bytes, cudaMemcpyDeviceToHost);

  // Multiply the matrices with GPU and tiling, measure elapsed time
  start_time = chrono::high_resolution_clock::now();
  tiling_multiply_matrix_gpu<<<grid, block>>>(d_matrixA, d_matrixB, d_result_matrix, n);
  cudaDeviceSynchronize();
  end_time = chrono::high_resolution_clock::now();
  duration_ms = end_time - start_time;
  printf("Matrix multiplication on GPU using Tiling, time elapsed: %f ms\n", duration_ms.count());

  // Copy result to host
  cudaMemcpy(result, d_result_matrix, bytes, cudaMemcpyDeviceToHost);
  cudaGetLastError();

  // Free allocated memory
  cudaFree(d_matrixA);
  cudaFree(d_matrixB);
  cudaFree(d_result_matrix);
  free(matrixA);
  free(matrixB);
  free(result);

  cudaDeviceReset();

  return 0;
}
