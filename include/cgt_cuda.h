#pragma once
#include "cuda_runtime.h"

// Code mostly ripped off from caffe

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) printf("%s\n", cudaGetErrorString(error)); \
  } while (0)

#define CUDA_CHECK_ERROR(msg) do { \
  cudaError_t e = cudaGetLastError(); \
  if (e != cudaSuccess) {printf("%s\n", cudaGetErrorString(e));} \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    if (status != CUBLAS_STATUS_SUCCESS) printf("%s\n", cublasGetErrorString(status)); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    if (status != CURAND_STATUS_SUCCESS) printf("%s\n", curandGetErrorString(status)); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: library error reporting.
// const char* cublasGetErrorString(cublasStatus_t error);
// const char* curandGetErrorString(curandStatus_t error);

// TODO better determination of threads info
// todo: this is c++ so let's use refs
static void cgt_get_bt(size_t size, int* num_blocks, int* num_threads) {
  if(size <= 32)
    *num_threads = 32;
  else if(size <= 64)
    *num_threads = 64;
  else if(size <= 128)
    *num_threads = 128;
  else if(size <= 256)
    *num_threads = 256;
  else if(size <= 512)
    *num_threads = 512;
  else
    *num_threads = 1024;
  *num_blocks = (int)(((size + *num_threads - 1) / *num_threads));
  if (*num_blocks < 0 || 128 < *num_blocks) {
    *num_blocks = 128;
  }
}
