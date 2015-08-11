#pragma once

#include "stddef.h"
#include "stdbool.h"
#include "IRC.h"
#include <cassert>

// ================================================================
// Basic structs and enums
// ================================================================

typedef enum cgtDtype {
  cgt_i1 = 1,
  cgt_i2 = 3,
  cgt_i4 = 5,
  cgt_i8 = 7,
  cgt_f2 = 23,
  cgt_f4 = 11,
  cgt_f8 = 12,
  cgt_f16 = 13,
  cgt_c8 = 14,
  cgt_c16 = 15,
  cgt_c32 = 16,
  cgt_O = 17
} cgtDtype;
// print np.dtype('i1').num # etc

static inline int cgt_itemsize(cgtDtype dtype) {
  switch (dtype) {
    case cgt_i1:
      return 1;
    case cgt_i2:
      return 2;
    case cgt_i4:
      return 4;
    case cgt_i8:
      return 8;
    case cgt_f2:
      return 2;
    case cgt_f4:
      return 4;
    case cgt_f8:
      return 8;
    case cgt_f16:
      return 16;
    case cgt_c8:
      return 8;
    case cgt_c16:
      return 16;
    case cgt_c32:
      return 32;
    case cgt_O:
      return 8;
  }
  assert(0 && "invalid dtype");
}

typedef enum cgtDevtype {
  cgtCPU,
  cgtGPU
} cgtDevtype;

class cgtObject {
public:
  enum ObjectKind {
    UndefKind=0,
    ArrayKind,
    TupleKind
  };
  cgtObject() : ref_cnt(0), kind(UndefKind) { }
  cgtObject(ObjectKind kind) : ref_cnt(0), kind(kind) { }
  void Retain() const { ++ref_cnt; }
  inline void Release() const;
  ObjectKind kind;
private:
  mutable unsigned ref_cnt;
};

class cgtArray : public cgtObject {
public:
  cgtArray(size_t ndim, const size_t* shape, cgtDtype dtype, cgtDevtype devtype);
  cgtArray(size_t ndim, const size_t* shape, cgtDtype dtype, cgtDevtype devtype, void* fromdata, bool copy);
  ~cgtArray();

  size_t ndim() const { return ndim_; }
  const size_t* shape() const { return shape_; }
  size_t size() const {
    size_t s = 1;
    for (size_t i = 0; i < ndim_; ++i) {
      s *= shape_[i];
    }
    return s;
  }
  size_t nbytes() const { return size() * cgt_itemsize(dtype_); }
  size_t stride(size_t i) const {
    if (ndim_ == 0) {
      return 0;
    }
    assert(0 <= i && i < ndim_ && ndim_ >= 1);
    size_t s = 1;
    for (size_t j = i; j < ndim_ - 1; ++j) { // note that (ndim_-1) >= 0, which is important because ndim_ is unsigned
      s *= shape_[j + 1];
    }
    return s;
  }
  cgtDtype dtype() const { return dtype_; }
  cgtDevtype devtype() const { return devtype_; }
  bool ownsdata() const { return ownsdata_; }
  void* data() { return data_; }
  
  template <typename T>
  T& at(size_t i) {return static_cast<T*>(data_)[i];}
  template <typename T>
  T& at(size_t i, size_t j) {return static_cast<T*>(data_)[i*shape_[1]+j];}
  template <typename T>
  T& at(size_t i, size_t j, size_t k) {return static_cast<T*>(data_)[(i*shape_[1]+j)*shape_[2]+k];}
  template <typename T>
  T& at(size_t i, size_t j, size_t k, size_t l) {return static_cast<T*>(data_)[((i*shape_[1]+j)*shape_[2]+k)*shape_[3]+l];}
  void print();

private:
  const size_t ndim_;
  const size_t* shape_;
  const cgtDtype dtype_;
  const cgtDevtype devtype_;
  const bool ownsdata_;
  void* data_;
};

class cgtTuple : public cgtObject {
public:
  cgtTuple(size_t len);
  void setitem(int i, cgtObject *o) {
    members[i] = o;
  }
  cgtObject *getitem(int i) {
    return members[i].get();
  }
  size_t size() {return len;}
  ~cgtTuple();
  size_t len;
  IRC<cgtObject> *members;
};

void cgtObject::Release() const {
  assert (ref_cnt > 0 && "Reference count is already zero.");
  if (--ref_cnt == 0) {
    if (kind==ArrayKind) delete (const cgtArray *)this;  // XXX is this legit?
    else if (kind==TupleKind) delete (const cgtTuple *)this;
    else assert(0 && "invalid kind");
  }
}

typedef void (*cgtByRefFun)(void * /* closure data */, cgtObject ** /* read */, cgtObject * /* write */);
typedef cgtObject *(*cgtByValFun)(void * /* closure data */, cgtObject ** /* read */);

// ================================================================
// Error handling 
// ================================================================

#define cgt_assert(x)  \
    do {\
        if (!(x)) {\
            fprintf (stderr, "Assertion failed: %s (%s:%d)\n", #x, \
                __FILE__, __LINE__);\
            fflush (stderr);\
            abort();\
        }\
    } while (0)

#define CGT_NORETURN __attribute__ ((noreturn))


typedef enum {
  cgtStatusOK = 0,
  cgtStatusErr
} cgtStatus;

extern cgtStatus cgtGlobalStatus;
extern char cgtGlobalErrorMsg[1000];

static inline void clear_error() {
  cgtGlobalStatus = cgtStatusOK;
}

// TODO can do it more safely now that we're in c++
#define cgt_check(x, msg, ...) \
    do {\
        if ((!(x))) {\
            sprintf(cgtGlobalErrorMsg, msg, ##__VA_ARGS__);\
            cgtGlobalStatus = cgtStatusErr;\
        }\
    } while(0)


// ================================================================
// Memory management
// ================================================================

static inline bool cgt_is_array(cgtObject *o) { return o->kind == cgtObject::ArrayKind; }
static inline bool cgt_is_tuple(cgtObject *o) { return o->kind == cgtObject::TupleKind; }

void *cgt_alloc(char devtype, size_t size);
void cgt_free(char devtype, void *ptr);
void cgt_memcpy(char dest_type, char src_type, void *dest_ptr, void *src_ptr, size_t nbytes);

