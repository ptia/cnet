#pragma once
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TENS_MAX_ORDER 4

struct tens {
    float *const arr;
    size_t shape[TENS_MAX_ORDER];
    size_t strides[TENS_MAX_ORDER];
    const uint8_t order;
};


static inline
float *tens_getp(struct tens T, size_t *index)
{
    size_t offset = 0;
    for (uint8_t i = 0; i < T.order; i++) {
        assert (index[i] < T.shape[i]);
        offset += T.strides[i] * index[i];
    }
    return &T.arr[offset];
}

static inline
float tens_get(struct tens T, size_t *index)
{
    return *tens_getp(T, index);
}

/* Create new tensor */

static inline
struct tens tens(float *arr, uint8_t order, size_t *shape)
{
    assert (order <= TENS_MAX_ORDER);

    struct tens out = (struct tens) {.arr = arr, .order = order};
    
    size_t stride = 1;
    for (int8_t i = order - 1; i >= 0; i--) {
        assert (shape[i] != 0);
        out.strides[i] = stride;
        out.shape[i] = shape[i];
        stride *= shape[i];
    }

    return out;
}

static inline
size_t tens_size(uint8_t order, size_t *shape)
{
    size_t size = 1;
    for (uint8_t i = 0; i < order ; i++) {
        size *= shape[i];
    }
    return size;
}

static inline
struct tens tens_zeros(uint8_t order, size_t *shape)
{
    return tens(
        calloc(1, tens_size(order, shape) * sizeof(float)),
        order, shape
    );
}

/* New views to same tensor */

/* Pre: T not strided TODO assert */
static inline
struct tens tens_reshape(struct tens T, uint8_t order, size_t *shape)
{
    assert (tens_size(T.order, T.shape) == tens_size(order, shape));
    return tens(T.arr, order, shape);
}

/* Slice of T (same order), starting from start, 
 * extending for shape along all axes */
static inline
struct tens tens_slice_shape(struct tens T, size_t *start, size_t *shape)
{
    struct tens S = (struct tens) {
        .arr = tens_getp(T, start),
        .order = T.order,
    };

    for (uint8_t i = 0; i < T.order; i++) {
        assert (shape[i] != 0);
        assert (start[i] + shape[i] <= T.shape[i]);
        S.shape[i] = shape[i];
        S.strides[i] = T.strides[i];
    }

    return S;
}

/* Slice of T (same order), going from start (inclusive) 
 * to end (exclusive) along all axes */
static inline
struct tens tens_slice(struct tens T, size_t *start, size_t *end)
{
    size_t shape[T.order + 1];
    for (uint8_t i = 0; i < T.order; i++) {
        assert (start[i] < end[i]);
        assert (end[i] <= T.shape[i]);
        shape[i] = end[i] - start[i];
    }
    return tens_slice_shape(T, start, shape);
}
