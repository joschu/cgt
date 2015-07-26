#pragma once

#include "stddef.h"
#include "stdbool.h"

typedef enum cgt_dtype {
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
} cgt_dtype;
// print np.dtype('i1').num # etc

typedef enum cgt_devtype {
    cgt_cpu,
    cgt_gpu
} cgt_devtype;

typedef struct cgt_array {
    int ndim;
    cgt_dtype dtype;
    cgt_devtype devtype;
    size_t* shape;
    void* data;
    size_t stride;
    bool ownsdata;
} cgt_array;

static inline size_t cgt_size(const cgt_array* a) {
    size_t out = 1;
    for (size_t i=0; i < a->ndim; ++i) out *= a->shape[i];
    return out;
}

static inline void cgt_get_strides(const cgt_array* a, size_t* strides) {
    if (a->ndim > 0) strides[a->ndim-1] = 1;
    for (int i=a->ndim-2; i >= 0; --i) strides[i] = strides[i+1]*a->shape[i+1];
}

static inline int cgt_itemsize(cgt_dtype dtype) {
    switch (dtype) {
        case cgt_i1: return 1;
        case cgt_i2: return 2;
        case cgt_i4: return 4;
        case cgt_i8: return 8;
        case cgt_f2: return 2;
        case cgt_f4: return 4;
        case cgt_f8: return 8;
        case cgt_f16: return 16;
        case cgt_c8: return 8;
        case cgt_c16: return 16;
        case cgt_c32: return 32;
        case cgt_O: return 8;
    }
}

static inline size_t cgt_nbytes(const cgt_array* a) {
    return cgt_size(a) * cgt_itemsize(a->dtype);
}
