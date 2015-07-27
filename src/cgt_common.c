#include "stdlib.h"
#include "assert.h"
#include "memory.h"
#include "stdio.h"
#include "cgt_common.h"
#ifdef CGT_ENABLE_CUDA
#include "cgt_cuda.h"
#endif


// ================================================================
// Error handling 
// ================================================================

 void cgt_abort() {
    abort();    
 }

cgt_status cgt_global_status = cgt_ok;
char cgt_global_errmsg[1000];

// ================================================================
// Memory management 
// ================================================================
 

void* cgt_alloc(char devtype, size_t size) {
    if (devtype == cgt_cpu) {
        return malloc(size);
    }
    else {
        #ifdef CGT_ENABLE_CUDA
        void* out;
        CUDA_CHECK(cudaMalloc(&out, size));
        return out;
        #else
        cgt_assert(0 && "CUDA disabled");
        #endif
    }
}

void cgt_free(char devtype, void* ptr) {
    if (devtype == cgt_cpu) {
        free(ptr);
    }
    else {
        #ifdef CGT_ENABLE_CUDA
        CUDA_CHECK(cudaFree(ptr));
        #else
        cgt_assert(0 && "CUDA disabled");
        #endif
    }
}

void cgt_memcpy(char dest_type, char src_type, void* dest_ptr, void* src_ptr, size_t nbytes) {
    if (src_type == cgt_cpu && dest_type == cgt_cpu) {
        memcpy(dest_ptr, src_ptr, nbytes);
    }
    else {
        #ifdef CGT_ENABLE_CUDA
        enum cudaMemcpyKind kind;
        if       (src_type == cgt_cpu && dest_type == cgt_gpu) kind = cudaMemcpyHostToDevice;
        else if  (src_type == cgt_gpu && dest_type == cgt_cpu) kind = cudaMemcpyDeviceToHost;
        else if  (src_type == cgt_gpu && dest_type == cgt_gpu) kind = cudaMemcpyDeviceToDevice;
        else cgt_assert(0 && "invalid src/dest types");
        CUDA_CHECK(cudaMemcpy(dest_ptr, src_ptr, nbytes, kind));
        #else
        cgt_assert(0 && "CUDA disabled");
        #endif
    }
}

