#pragma once
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#define TENS_MAX_ORDER 4

struct tens {
    float *const arr;
    size_t shape[TENS_MAX_ORDER];
    size_t strides[TENS_MAX_ORDER];
    const uint8_t order;
};

struct tens tens(float *arr, uint8_t order, size_t *shape);

static inline
float *tens_getp(struct tens A, size_t *index)
{
    size_t offset = 0;
    for (uint8_t i = 0; i < A.order; i++) {
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
