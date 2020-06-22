#pragma once

#include "util.h"
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define TENS_MAX_ORDER 4

/* STRUCT TENSOR
 * 
 * Each instance is a view on an underlying data array (arr),
 * Multiple views can share the same data, changes to it are shared.
 * Indexes are C-like by defalt: first index changes slowly, last one quickly,
 * but this can be changed with striding.
 *
 * struct tensor (the view, not the data) is small enough to be passed by value
 * and all the functions here do so. This means that no function here will 
 * change the view that's passed to it (some change the data).
 * It is possible to manually change shape and strides, but this might corrupt
 * access to the data.
 */
struct tensor {
    float *arr;
    size_t shape[TENS_MAX_ORDER];
    size_t strides[TENS_MAX_ORDER];
    int8_t order;
};

struct tens_pair {
    struct tensor S, T;
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
 * Return a new struct tensor instance with a different view, on the same data.
 * These functions never change the underlying data.
 * Any subsequent changes to the data will be shared.
 * No memory is allocated, all (struct tensor) instances exist on the stack.
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
 * All these functions take struct tensor by value, not pointer, so they won't
 * change the original view.
 */


/* GETTERS */

/* Get pointer to (scalar) element in tensor */
static inline
float *tens_getp(struct tensor T, const size_t *index)
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
float tens_get(struct tensor T, const size_t *index)
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
size_t tens_size(struct tensor T)
{
    return tens_shape_size(T.order, T.shape);
}

/* Test whether two tensors have the exact same shape and order */
static inline
bool tens_match(struct tensor S, struct tensor T)
{
    return S.order == T.order && !memcmp(S.shape, T.shape, S.order);
}

/* CREATORS */

/* Tensor from raw data array (in-place, does not malloc),
 * assuming default layout */
static inline
struct tensor tensor(float *arr, int8_t order, const size_t *shape)
{
    assert (order <= TENS_MAX_ORDER);

    struct tensor out = (struct tensor) { .arr = arr, .order = order };
    
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
struct tensor tens_zeros(int8_t order, const size_t *shape)
{
    return tensor(
        calloc(1, tens_shape_size(order, shape) * sizeof(float)),
        order, shape
    );
}


/* VIEWS */

/* Pre (TODO assert): T not strided */
static inline
struct tensor tens_reshape(struct tensor T, int8_t order, const size_t *shape)
{
    assert (tens_size(T) == tens_shape_size(order, shape));
    return tensor(T.arr, order, shape);
}

/* Slice of T (same order), starting from start, 
 * extending for shape along all axes */
static inline
struct tensor tens_slice_shape(
        struct tensor T, const size_t *start, const size_t *shape)
{
    struct tensor S = (struct tensor) {
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
struct tensor tens_slice(
        struct tensor T, const size_t *start, const size_t *end)
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
struct tensor tens_swap_axes(struct tensor T, int8_t axis1, int8_t axis2)
{
    assert (axis1 < T.order && axis2 < T.order);

    struct tensor S = T;
    S.shape[axis1] = T.shape[axis2];
    S.shape[axis2] = T.shape[axis1];
    S.strides[axis1] = T.strides[axis2];
    S.strides[axis2] = T.strides[axis1];
    return S;
}

/* Add an multiple axes to the tensor at dimension n. 
 * They will have length 1 */
static inline
struct tensor tens_add_axes(struct tensor T, int8_t axis, int8_t count)
{
    assert (axis <= T.order);
    assert (T.order + count <= TENS_MAX_ORDER);

    if (count == 0)
        return T;

    struct tensor S = { .arr = T.arr, .order = T.order + count };
    for (int8_t i = 0; i < T.order + count; i++) {
        if (i < axis) {
            S.shape[i] = T.shape[i];
            S.strides[i] = T.strides[i];
        } else if (axis <= i && i < axis + count) {
            S.shape[i] = 1;
            S.strides[i] = 0;
        } else {
            S.shape[i] = T.shape[i - count];
            S.strides[i] = T.strides[i - count];
        }
    }
    return S;
}

/* Broadcast two tensors together using numpy rules,
 * skipping lower n axes */
static inline
struct tens_pair tens_broadcast(
        struct tensor S, struct tensor T, int8_t skip_axes)
{
    /* Match orders by adding axes at the highest dimension */
    if (S.order < T.order)
        S = tens_add_axes(S, 0, T.order - S.order);
    if (T.order < S.order)
        T = tens_add_axes(T, 0, S.order - T.order);

    assert (skip_axes < S.order);

    for (int8_t i = 0; i < S.order - skip_axes; i++) {
        if (S.shape[i] == T.shape[i])
            continue;

        if (S.shape[i] == 1) {
            S.shape[i] = T.shape[i];
            S.strides[i] = 0;
            continue;
        }

        if (T.shape[i] == 1) {
            T.shape[i] = S.shape[i];
            T.strides[i] = 0;
            continue;
        }
        
        assert (false);
    }

    return (struct tens_pair) {S, T};
}

/* MODIFIERS */

/* Multiply all elements of a tensor by a scalar 
 * T = lT  */
void tens_scalar_mul(struct tensor T, float l);

/* Add two matching tensors in place
 * D = S + T  */
void tens_add(struct tensor S, struct tensor T, struct tensor D);

/* Matrix multiplication over the last two dimensions
 * U = S@T  */
void tens_mat_mul(struct tensor S, struct tensor T, struct tensor D);
