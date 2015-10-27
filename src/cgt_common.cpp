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

cgtArray::cgtArray(int ndim, const long* shape, cgtDtype dtype, cgtDevtype devtype)
    : cgtObject(ObjectKind::ArrayKind),
      ndim_(ndim),
      dtype_(dtype),
      devtype_(devtype),
      ownsdata_(true) {
  shape_ = new long[ndim];
  memcpy(const_cast<long*>(shape_), shape, ndim * sizeof(long));
  data_ = cgt_alloc(devtype_, nbytes());
}

cgtArray::cgtArray(int ndim, const long* shape, cgtDtype dtype, cgtDevtype devtype, void* fromdata, bool copy)
    : cgtObject(ObjectKind::ArrayKind),
      ndim_(ndim),
      shape_(shape),
      dtype_(dtype),
      devtype_(devtype),
      ownsdata_(copy) {
  cgt_assert(fromdata != NULL);
  shape_ = new long[ndim];
  memcpy(const_cast<long*>(shape_), shape, ndim * sizeof(long));
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

cgtTuple::cgtTuple(int len)
    : cgtObject(ObjectKind::TupleKind), len(len) {
  members = new IRC<cgtObject>[len];
}

cgtTuple::~cgtTuple() {
  delete[] members;
}


// ================================================================
// Copying
// ================================================================

void cgt_copy_object(cgtObject* to, cgtObject* from) {
  cgt_assert(to->kind() == from->kind());
  if (to->kind() == cgtObject::ArrayKind) {    
    cgt_copy_array(static_cast<cgtArray*>(to), static_cast<cgtArray*>(from));
  }
  else if (to->kind() == cgtObject::TupleKind) {
    cgt_copy_tuple(static_cast<cgtTuple*>(to), static_cast<cgtTuple*>(from));
  }
  else cgt_assert(0 && "unreachable");
}

void cgt_copy_array(cgtArray* to, cgtArray* from) {
  cgt_assert(from->size() == to->size() && from->dtype() == to->dtype()) ;
  cgt_memcpy(to->devtype(), from->devtype(), to->data(), from->data(), from->nbytes());
}

void cgt_copy_tuple(cgtTuple* to, cgtTuple* from) {
  for (int i=0; i < to->size(); ++i) cgt_copy_object(to->getitem(i), from->getitem(i));
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

void *cgt_alloc(cgtDevtype devtype, long size) {
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

void cgt_memcpy(cgtDevtype dest_type, cgtDevtype src_type, void *dest_ptr, void *src_ptr, long nbytes) {
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

