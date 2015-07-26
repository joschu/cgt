#include "cgt_cuda.h"
#include "cublas.h"

CudaContext g_context;

#ifdef CGT_ENABLE_CUDA

void cuda_initialize() {
    CUDA_CHECK(cudaStreamCreate(&g_context.stream));
    CUBLAS_CHECK(cublasCreate_v2(&g_context.cublas_handle));
    CUBLAS_CHECK(cublasSetStream(g_context.cublas_handle, g_context.stream));
    // CUDNN_CHECK(cudnnCreate(&g_context.cudnn_handle));
    // CUDNN_CHECK(cudnnSetStream(g_context.cudnn_handle, g_context.stream));
}

#else 

void cuda_initialize() {
}

#endif