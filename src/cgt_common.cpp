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

namespace cgt {

Array::Array(int ndim, size_t *inshape, Dtype dtype, Devtype devtype)
    : Object(ObjectKind::Array), ndim(ndim), dtype(dtype), 
      devtype(devtype), ownsdata() {
  shape = new size_t[ndim];
  for (int i = 0; i < ndim; ++i) shape[i] = inshape[i];
  size_t nbytes = cgt_nbytes(this);
  data = cgt_alloc(devtype, nbytes);
}

Array::Array(int ndim, size_t *inshape, Dtype dtype, Devtype devtype,
    void* fromdata, bool copy)
    : Object(ObjectKind::Array), ndim(ndim), dtype(dtype), devtype(devtype) {
  shape = new size_t[ndim];
  for (int i = 0; i < ndim; ++i) shape[i] = inshape[i];
  cgt_assert(fromdata != NULL);
  if (copy) {
    size_t nbytes = cgt_nbytes(this);
    data = cgt_alloc(devtype, nbytes);
    ownsdata = true;
    cgt_memcpy(devtype, DevCPU, data, fromdata, nbytes);
  }
  else {
    data = fromdata;
    ownsdata = false;
  }
}

Array::~Array() {
  delete[] shape;
  if (ownsdata) free(data);
}

Tuple::Tuple(size_t len)
    : Object(ObjectKind::Tuple), len(len) {
  members = new IRC<Object>[len];
}

Tuple::~Tuple() {
  delete[] members;
}


// ================================================================
// Error handling 
// ================================================================

void cgt_abort() {
  abort();
}

Status gStatus = StatusOK;
char gErrorMsg[1000];

// ================================================================
// Memory management 
// ================================================================

void *cgt_alloc(char devtype, size_t size) {
  if (devtype == DevCPU) {
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

void cgt_free(char devtype, void *ptr) {
  if (devtype == DevCPU) {
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

void cgt_memcpy(char dest_type, char src_type, void *dest_ptr, void *src_ptr, size_t nbytes) {
  if (src_type == DevCPU && dest_type == DevCPU) {
    memcpy(dest_ptr, src_ptr, nbytes);
  }
  else {
#ifdef CGT_ENABLE_CUDA
        enum cudaMemcpyKind kind;
        if       (src_type == DevCPU && dest_type == DevGPU) kind = cudaMemcpyHostToDevice;
        else if  (src_type == DevGPU && dest_type == DevCPU) kind = cudaMemcpyDeviceToHost;
        else if  (src_type == DevGPU && dest_type == DevGPU) kind = cudaMemcpyDeviceToDevice;
        else cgt_assert(0 && "invalid src/dest types");
        CUDA_CHECK(cudaMemcpy(dest_ptr, src_ptr, nbytes, kind));
        #else
    cgt_assert(0 && "CUDA disabled");
#endif
  }
}

}
