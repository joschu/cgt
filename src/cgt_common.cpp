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

template<typename T>
static T prod(const std::vector<T>& l) {
  T out = T(1);
  for (const T &x : l) {
    out *= x;
  }
  return out;
}

static inline SizeList get_strides(const SizeList& shape) {
  int ndim = shape.size();
  SizeList strides(ndim);
  if (ndim > 0) {
    strides[ndim - 1] = 1;
  }
  for (int i = ndim - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

cgtArray::cgtArray(const SizeList& shape_, cgtDtype dtype_, cgtDevtype devtype_)
    : cgtObject(ObjectKind::ArrayKind),
      shape(shape_),
      size(prod(shape_)),
      nbytes(prod(shape_) * cgt_itemsize(dtype_)),
      ndim(shape_.size()),
      strides(get_strides(shape_)),
      dtype(dtype_),
      devtype(devtype_),
      ownsdata(true) {
  data = cgt_alloc(devtype, nbytes);
}

cgtArray::cgtArray(const SizeList& shape_, cgtDtype dtype_, cgtDevtype devtype_,
    void* fromdata, bool copy)
    : cgtObject(ObjectKind::ArrayKind),
      shape(shape_),
      size(prod(shape_)),
      nbytes(prod(shape_) * cgt_itemsize(dtype_)),
      ndim(shape_.size()),
      strides(get_strides(shape_)),
      dtype(dtype_),
      devtype(devtype_),
      ownsdata(copy) {
  cgt_assert(fromdata != NULL);
  if (copy) {
    data = cgt_alloc(devtype, nbytes);
    cgt_memcpy(devtype, cgtCPU, data, fromdata, nbytes);
  } else {
    data = fromdata;
  }
}

cgtArray::~cgtArray() {
  if (ownsdata) cgt_free(devtype, data);
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

