#pragma once

#include "stddef.h"
#include "stdbool.h"
#include "IRC.h"
#include <cassert>

// ================================================================
// Basic structs and enums
// ================================================================

namespace cgt {

typedef enum Dtype {
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
} Dtype;
// print np.dtype('i1').num # etc

typedef enum Devtype {
  DevCPU,
  DevGPU
} Devtype;

class Object {
public:
  enum ObjectKind {
    Undef=0,
    Array,
    Tuple
  };
  Object() : ref_cnt(0), kind(Undef) { }
  Object(ObjectKind kind) : ref_cnt(0), kind(kind) { }
  void Retain() const { ++ref_cnt; }
  void Release() const {
    assert (ref_cnt > 0 && "Reference count is already zero.");
    if (--ref_cnt == 0) delete static_cast<const Object *>(this);
  }
  ObjectKind kind;
private:
  mutable unsigned ref_cnt;
};

class Array : public Object {
public:
  Array(int ndim, size_t *shape, Dtype, Devtype);
  Array(int ndim, size_t *shape, Dtype, Devtype, void* fromdata, bool copy);
  ~Array();
  int ndim;
  Dtype dtype;
  Devtype devtype;
  size_t *shape;
  void *data;
  bool ownsdata;
};

static inline bool cgt_is_array(Object *o) { return o->kind == Object::Array; }
static inline bool cgt_is_tuple(Object *o) { return o->kind == Object::Tuple; }

class Tuple : public Object {
public:
  Tuple(size_t len);
  void setitem(int i, Object *o) {
    members[i] = o;
  }
  Object *getitem(int i) {
    return members[i].get();
  }
  size_t size() {return len;}
  ~Tuple();
  size_t len;
  IRC<Object> *members;
};

static inline size_t cgt_size(const Array *a) {
  size_t out = 1;
  for (size_t i = 0; i < a->ndim; ++i) out *= a->shape[i];
  return out;
}

static inline void cgt_get_strides(const Array *a, size_t *strides) {
  if (a->ndim > 0) strides[a->ndim - 1] = 1;
  for (int i = a->ndim - 2; i >= 0; --i) strides[i] = strides[i + 1] * a->shape[i + 1];
}

static inline int cgt_itemsize(Dtype dtype) {
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
}

static inline size_t cgt_nbytes(const Array *a) {
  return cgt_size(a) * cgt_itemsize(a->dtype);
}

typedef void (*Inplacefun)(void * /* closure data */, Object ** /* read */, Object* /* write */);
typedef Object *(*Valretfun)(void * /* closure data */, Object ** /* read */);

// ================================================================
// Error handling 
// ================================================================

#define cgt_assert(x)  \
    do {\
        if (!(x)) {\
            fprintf (stderr, "Assertion failed: %s (%s:%d)\n", #x, \
                __FILE__, __LINE__);\
            fflush (stderr);\
            cgt::cgt_abort();\
        }\
    } while (0)

#define CGT_NORETURN __attribute__ ((noreturn))

CGT_NORETURN void cgt_abort();

typedef enum {
  StatusOK = 0,
  StatusErr
} Status;

extern Status gStatus;
extern char gErrorMsg[1000];

static inline void cgt_clear_error() {
  gStatus = StatusOK;
}

// XXX how do we check that message doesn't overflow?
#define cgt_check(x, msg, ...) \
    do {\
        if ((!(x))) {\
            sprintf(cgt::gErrorMsg, msg, ##__VA_ARGS__);\
            cgt::gStatus = cgt::StatusErr;\
        }\
    } while(0)


// ================================================================
// Memory management
// ================================================================

void *cgt_alloc(char devtype, size_t size);
void cgt_free(char devtype, void *ptr);
void cgt_memcpy(char dest_type, char src_type, void *dest_ptr, void *src_ptr, size_t nbytes);

}


