#include "cgt_cuda.h"
#include "stdlib.h"
#include "assert.h"
#include "memory.h"
#include "stdio.h"
#include "cgt_utils.h"

void* cgt_alloc(char devtype, size_t size) {
    if (devtype == cgt_cpu) {
        return malloc(size);
    }
    else {
        void* out;
        CUDA_CHECK(cudaMalloc(&out, size));
        return out;
    }
}

void cgt_free(char devtype, void* ptr) {
    if (devtype == cgt_cpu) {
        free(ptr);
    }
    else {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void cgt_memcpy(char dest_type, char src_type, void* dest_ptr, void* src_ptr, size_t nbytes) {
    if (src_type == cgt_cpu && dest_type == cgt_cpu) {
        memcpy(dest_ptr, src_ptr, nbytes);
    }
    else {
        #ifdef ENABLE_CUDA
        enum cudaMemcpyKind kind;
        if       (src_type == cgt_cpu && dest_type == cgt_gpu) kind = cudaMemcpyHostToDevice;
        else if  (src_type == cgt_gpu && dest_type == cgt_cpu) kind = cudaMemcpyDeviceToHost;
        else if  (src_type == cgt_gpu && dest_type == cgt_gpu) kind = cudaMemcpyDeviceToDevice;
        else cgt_always_assert(0 && "invalid src/test types");
        #endif
        CUDA_CHECK(cudaMemcpy(dest_ptr, src_ptr, nbytes, kind));
    }
}
