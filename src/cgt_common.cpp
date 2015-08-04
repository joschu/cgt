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

cgtArray::cgtArray(int ndim, size_t *inshape, cgtDtype dtype, cgtDevtype devtype)
    : cgtObject(ObjectKind::ArrayKind), ndim(ndim), dtype(dtype),
      devtype(devtype), ownsdata(true) {
  shape = new size_t[ndim];
  for (int i = 0; i < ndim; ++i) shape[i] = inshape[i];
  size_t nbytes = cgt_nbytes(this);
  data = cgt_alloc(devtype, nbytes);
}

cgtArray::cgtArray(int ndim, size_t *inshape, cgtDtype dtype, cgtDevtype devtype,
    void* fromdata, bool copy)
    : cgtObject(ObjectKind::ArrayKind), ndim(ndim), dtype(dtype), devtype(devtype) {
  shape = new size_t[ndim];
  for (int i = 0; i < ndim; ++i) shape[i] = inshape[i];
  cgt_assert(fromdata != NULL);
  if (copy) {
    size_t nbytes = cgt_nbytes(this);
    data = cgt_alloc(devtype, nbytes);
    ownsdata = true;
    cgt_memcpy(devtype, cgtCPU, data, fromdata, nbytes);
  }
  else {
    data = fromdata;
    ownsdata = false;
  }
}

cgtArray::~cgtArray() {
  delete[] shape;
  if (ownsdata) free(data);
}

cgtTuple::cgtTuple(size_t len)
    : cgtObject(ObjectKind::TupleKind), len(len) {
  members = new IRC<cgtObject>[len];
}

cgtTuple::~cgtTuple() {
  delete[] members;
}


// ================================================================
// Error handling 
// ================================================================

void cgt_abort() {
  abort();
}

cgtStatus cgtGlobalStatus = cgtStatusOK;
char cgtGlobalErrorMsg[1000];


// ================================================================
// Memory management 
// ================================================================

void *cgt_alloc(char devtype, size_t size) {
  if (devtype == cgtCPU) {
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
  if (devtype == cgtCPU) {
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
  if (src_type == cgtCPU && dest_type == cgtCPU) {
    memcpy(dest_ptr, src_ptr, nbytes);
  }
  else {
#ifdef CGT_ENABLE_CUDA
        enum cudaMemcpyKind kind;
        if       (src_type == cgtCPU && dest_type == cgtGPU) kind = cudaMemcpyHostToDevice;
        else if  (src_type == cgtGPU && dest_type == cgtCPU) kind = cudaMemcpyDeviceToHost;
        else if  (src_type == cgtGPU && dest_type == cgtGPU) kind = cudaMemcpyDeviceToDevice;
        else cgt_assert(0 && "invalid src/dest types");
        CUDA_CHECK(cudaMemcpy(dest_ptr, src_ptr, nbytes, kind));
        #else
    cgt_assert(0 && "CUDA disabled");
#endif
  }
}

