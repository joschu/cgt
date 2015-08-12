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

cgtArray::cgtArray(size_t ndim, const size_t* shape, cgtDtype dtype, cgtDevtype devtype)
    : cgtObject(ObjectKind::ArrayKind),
      ndim_(ndim),
      dtype_(dtype),
      devtype_(devtype),
      ownsdata_(true) {
  shape_ = new size_t[ndim];
  memcpy(const_cast<size_t*>(shape_), shape, ndim * sizeof(size_t));
  data_ = cgt_alloc(devtype_, nbytes());
}

cgtArray::cgtArray(size_t ndim, const size_t* shape, cgtDtype dtype, cgtDevtype devtype, void* fromdata, bool copy)
    : cgtObject(ObjectKind::ArrayKind),
      ndim_(ndim),
      shape_(shape),
      dtype_(dtype),
      devtype_(devtype),
      ownsdata_(copy) {
  cgt_assert(fromdata != NULL);
  shape_ = new size_t[ndim];
  memcpy(const_cast<size_t*>(shape_), shape, ndim * sizeof(size_t));
  if (copy) {
    data_ = cgt_alloc(devtype, nbytes());
    cgt_memcpy(devtype, cgtCPU, data_, fromdata, nbytes());
  } else {
    data_ = fromdata;
  }
}

void cgtArray::print() {
  printf("Array{shape=(");
  if (ndim_ > 0) printf("%zu",shape_[0]);
  for (int i=1; i < ndim_; ++i) {
    printf(", %zu", shape_[i]);
  }
  printf("), dtype=%i}", dtype_);
}

cgtArray::~cgtArray() {
  delete[] shape_;
  if (ownsdata_) cgt_free(devtype_, data_);
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

void *cgt_alloc(cgtDevtype devtype, size_t size) {
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

void cgt_free(cgtDevtype devtype, void *ptr) {
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

void cgt_memcpy(cgtDevtype dest_type, cgtDevtype src_type, void *dest_ptr, void *src_ptr, size_t nbytes) {
  if (src_type == cgtCPU && dest_type == cgtCPU) {
    memcpy(dest_ptr, src_ptr, nbytes);
  } else {
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

