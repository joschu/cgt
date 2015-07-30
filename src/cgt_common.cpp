#include "stdlib.h"
#include "assert.h"
#include "memory.h"
#include "stdio.h"
#include "cgt_common.h"
#ifdef CGT_ENABLE_CUDA
#include "cgt_cuda.h"
#endif

// ================================================================
// Object alloc/dealloc 
// ================================================================
 

cgt_array* new_cgt_array(int ndim, size_t* shape, cgt_dtype dtype, 
    cgt_devtype devtype) {
    size_t* myshape = new size_t[ndim];
    for (int i=0; i < ndim; ++i) myshape[i] = shape[i];
    cgt_array* out = new cgt_array();
    out->typetag = cgt_arraytype;
    out->ndim = ndim;
    out->dtype = dtype;
    out->devtype = devtype;
    out->shape = myshape;
    out->data = malloc(cgt_nbytes(out));
    out->ownsdata = true;
    return out;
}

cgt_tuple* new_cgt_tuple(int len) {
    cgt_object** members = new cgt_object*[len];
    cgt_tuple* out = new cgt_tuple;
    out->typetag = cgt_tupletype;
    out->len =  len;
    out->members = members;
    return out;
}

void delete_cgt_tuple(cgt_tuple* t) {
    delete[] t->members;
}
void delete_cgt_array(cgt_array* a) {
    delete[] a->shape;
    if (a->ownsdata && a->data != NULL) free(a->data);
}


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

