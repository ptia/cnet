#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TENS_MAX_ORDER 4

/* STRUCT TENS
 * 
 * Each instance is a view on an underlying data array (arr),
 * Multiple views can share the same data, changes to it are shared.
 * Indexes are C-like by defalt: first index changes slowly, last one quickly,
 * but this can be changed with striding.
 *
 * Struct tens (the view, not the data) is small enough to be passed by value
 * and all the functions here do so. This means that no function here will 
 * change the view that's passed to it (some change the data).
 * It is possible to manually change shape and strides, but this might corrupt
 * access to the data.
 */
struct tens {
    float *const arr;
    size_t shape[TENS_MAX_ORDER];
    size_t strides[TENS_MAX_ORDER];
    const int8_t order;
};


/* There are 4 kinds of functions on tensors, all declared in this file:
 *
 * GETTERS:
 * Get a property of a tensor, making no changes, allocating no memory.
 * e.g.: tens_get() - Get the (scalar) element of a tensor at given coordinates
 *
 * CREATORS:
 * Generate and return a brand new tensor with default layout, 
 * either on a pre-existing array or malloc'ing (remember to free!).
 * e.g.: tens_zeros() - Calloc a new zero tensor
 *
 * VIEWS:
 * Return a new (struct tens) instance with a different view, on the same data.
 * These functions never change the underlying data.
 * Any subsequent changes to the data will be shared.
 * No memory is allocated, all (struct tens) instances exist on the stack.
 * e.g.: tens_slice() - Get a slice (subtensor) of a tensor
 *
 * MODIFIERS:
 * Apply an operation changing the data of a tensor, leave the view unchanged
 * These functions return null and no memory is allocated, 
 * if a function returns a result, a result tensor has to be provided 
 * as the last argument, matching the shape and order of the expected result.
 * If the function takes multiple tensors, only the last one is modified
 * (unless otherwise specified).
 * e.g.: tens_add() - Add two matching tensors element-wise
 *
 * All these functions take struct tens by value, not pointer, so they won't
 * change the original view.
 */


/* GETTERS */

/* Get pointer to (scalar) element in tensor */
static inline
float *tens_getp(struct tens T, const size_t *index)
{
    size_t offset = 0;
    for (int8_t i = 0; i < T.order; i++) {
        assert (index[i] < T.shape[i]);
        offset += T.strides[i] * index[i];
    }
    return &T.arr[offset];
}

/* Get value of (scalar) element in tensor */
static inline
float tens_get(struct tens T, const size_t *index)
{
    return *tens_getp(T, index);
}

/* Get total linear size of the data (product of shape) */
static inline
size_t tens_shape_size(int8_t order, const size_t *shape)
{
    size_t size = 1;
    for (int8_t i = 0; i < order ; i++) {
        size *= shape[i];
    }
    return size;
}

/* Get total linear size of the data (product of shape) */
static inline
size_t tens_size(struct tens T)
{
    return tens_shape_size(T.order, T.shape);
}


/* CREATORS */

/* Tensor from raw data array (in-place, does not malloc),
 * assuming default layout */
static inline
struct tens tens(float *arr, int8_t order, const size_t *shape)
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

/* New zero-initialised tensor (calloc'ing, remember to free A.arr) */
static inline
struct tens tens_zeros(int8_t order, const size_t *shape)
{
    return tens(
        calloc(1, tens_shape_size(order, shape) * sizeof(float)),
        order, shape
    );
}


/* VIEWS */

/* Pre (TODO assert): T not strided */
static inline
struct tens tens_reshape(struct tens T, int8_t order, const size_t *shape)
{
    assert (tens_size(T) == tens_shape_size(order, shape));
    return tens(T.arr, order, shape);
}

/* Slice of T (same order), starting from start, 
 * extending for shape along all axes */
static inline
struct tens tens_slice_shape(
        struct tens T, const size_t *start, const size_t *shape)
{
    struct tens S = (struct tens) {
        .arr = tens_getp(T, start),
        .order = T.order,
    };

    for (int8_t i = 0; i < T.order; i++) {
        assert (shape[i] != 0);
        assert (start[i] + shape[i] <= T.shape[i]);
        S.shape[i] = shape[i];
        S.strides[i] = T.strides[i];
    }

    return S;
}

/* Slice of T (same order), going from start (inclusive) to end (exclusive),
 * along all axes */
static inline
struct tens tens_slice(struct tens T, const size_t *start, const size_t *end)
{
    size_t shape[T.order + 1];
    for (int8_t i = 0; i < T.order; i++) {
        assert (start[i] < end[i]);
        assert (end[i] <= T.shape[i]);
        shape[i] = end[i] - start[i];
    }
    return tens_slice_shape(T, start, shape);
}

/* Swap axes (like matrix transpose). Will change shape */
static inline
struct tens tens_swap_axes(struct tens T, int8_t axis1, int8_t axis2)
{
    struct tens S = T;
    S.shape[axis1] = T.shape[axis2];
    S.shape[axis2] = T.shape[axis1];
    S.strides[axis1] = T.strides[axis2];
    S.strides[axis2] = T.strides[axis1];
    return S;
}


/* MODIFIERS */

/* Add two matching tensors in place
 * T = S + T  */
void tens_add(struct tens S, struct tens T);

/* Multiply all elements of a tensor by a scalar */
void tens_scalar_mul(struct tens S, float l);
