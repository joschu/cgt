#pragma once

#include "stddef.h"
#include "stdbool.h"

#ifdef __cplusplus
extern "C" {
#endif

// ================================================================
// Basic structs and enums
// ================================================================
 
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

typedef enum cgt_typetag {
    cgt_arraytype,
    cgt_tupletype
} cgt_typetag;

typedef struct cgt_object {
    cgt_typetag typetag;
} cgt_object;

typedef struct cgt_array {
    cgt_typetag typetag;    
    int ndim;
    cgt_dtype dtype;
    cgt_devtype devtype;
    size_t* shape;
    void* data;
    bool ownsdata;
} cgt_array;

cgt_array* new_cgt_array(int ndim, size_t* shape, cgt_dtype, cgt_devtype);
void delete_cgt_array(cgt_array*);

typedef struct cgt_tuple {
    cgt_typetag typetag;
    int len;
    cgt_object** members;
} cgt_tuple;

cgt_tuple* new_cgt_tuple(int ndim);
void delete_cgt_tuple(cgt_tuple*);

static inline cgt_typetag cgt_type(cgt_object* o) {
    return ((cgt_typetag*)o)[0];
}

static inline bool cgt_is_array(cgt_object* o) {
    return cgt_type(o) == cgt_arraytype;
}

static inline bool cgt_is_tuple(cgt_object* o) {
    return cgt_type(o) == cgt_tupletype;
}

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

typedef void (*cgt_fun)(void*, cgt_object**);

// ================================================================
// Error handling 
// ================================================================

#define cgt_assert(x)  \
    do {\
        if (!(x)) {\
            fprintf (stderr, "Assertion failed: %s (%s:%d)\n", #x, \
                __FILE__, __LINE__);\
            fflush (stderr);\
            cgt_abort();\
        }\
    } while (0)

#define CGT_NORETURN __attribute__ ((noreturn))
CGT_NORETURN void nn_err_abort (void);

CGT_NORETURN void cgt_abort();

typedef enum {
    cgt_ok = 0,
    cgt_err
} cgt_status;

extern cgt_status cgt_global_status;
extern char cgt_global_errmsg[1000];

static inline void cgt_clear_error() {
    cgt_global_status = cgt_ok;
}


// XXX how do we check that message doesn't overflow?
#define cgt_check(x, msg, ...) \
    do {\
        if ((!(x))) {\
            sprintf(cgt_global_errmsg, msg, ##__VA_ARGS__);\
            cgt_global_status = cgt_err;\
        }\
    } while(0)

// ================================================================
// Memory management 
// ================================================================
 
void* cgt_alloc(char devtype, size_t size);
void cgt_free(char devtype, void* ptr);
void cgt_memcpy(char dest_type, char src_type, void* dest_ptr, void* src_ptr, size_t nbytes);





#ifdef __cplusplus
}
#endif
