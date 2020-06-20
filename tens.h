#pragma once
#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#define TENS_ORDER 4
#define tens_initial_strides(shape) {\
        shape[3] * shape[2] * shape[1], \
        shape[3] * shape[2], \
        shape[3], \
        1 \
    }
#define tens_index(...) ((size_t []) { __VA_ARGS__ })


struct tens {
    float *const arr;
    const size_t shape[TENS_ORDER];
    const size_t strides[TENS_ORDER];
};

#define tens(ARR, ...) \
    ((struct tens) { \
     .arr = ARR, \
     .shape = __VA_ARGS__, \
     .strides = tens_initial_strides(((size_t[]) __VA_ARGS__)) \
    });

static inline
float *tens_getp(struct tens A, size_t *index)
{
    size_t offset = 0;
    for (uint8_t i = 0; i < TENS_ORDER; i++) {
        assert (index[i] < A.shape[i]);
        offset += A.strides[i] * index[i];
    }
    return &A.arr[offset];
}

static inline
float tens_get(struct tens A, size_t *index)
{
    return *tens_getp(A, index);
}
