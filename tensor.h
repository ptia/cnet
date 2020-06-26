#pragma once

#include "util.h"
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdarg.h>

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

struct tens_shape {
    int8_t order;
    size_t shape[TENS_MAX_ORDER];
};

struct tensor {
    float *arr;
    size_t strides[TENS_MAX_ORDER];
    struct tens_shape shape;
};

struct tens_pair {
    struct tensor S, T;
};

/* There are 4 kinds of functions on tensors, all declared in this file:
 *
 * GETTERS:
 * Get a property of a tensor, making no changes, allocating no memory.
 * e.g.: tens_get() - Get the (scalar) element of a tensor at given coords
 *
 * CREATORS:
 * Generate and return a brand new tensor with default layout, 
 * either on a pre-existing array or malloc'ing (remember to free!).
 * e.g.: tens_zeros() - Calloc a new zero tensor
 *
 * VIEWS:
 * Return a new struct tensor with a different view on the same data.
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
    for (int8_t i = 0; i < T.shape.order; i++) {
        assert (index[i] < T.shape.shape[i]);
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
size_t tens_size(struct tens_shape shape)
{
    size_t size = 1;
    for (int8_t i = 0; i < shape.order ; i++) {
        size *= shape.shape[i];
    }
    return size;
}

/* Test whether two tensor shapes match exactly */
static inline
bool tens_match(struct tens_shape s1, struct tens_shape s2)
{
    return s1.order == s2.order && !memcmp(s1.shape, s2.shape, s1.order);
}


/* CREATORS */

/* Tensor from raw data array (in-place, does not malloc),
 * assuming default layout */
static inline
struct tensor tensor(float *arr, struct tens_shape shape)
{
    assert (shape.order <= TENS_MAX_ORDER);

    struct tensor out = (struct tensor) { .arr = arr, .shape = shape};
    
    size_t stride = 1;
    for (int8_t i = shape.order - 1; i >= 0; i--) {
        assert (shape.shape[i] != 0);
        out.strides[i] = stride;
        stride *= shape.shape[i];
    }

    return out;
}

/* New zero-initialised tensor (calloc'ing, remember to free A.arr) */
static inline
struct tensor tens_zeros(struct tens_shape shape)
{
    return tensor(calloc(1, tens_size(shape) * sizeof(float)), shape);
}


/* VIEWS */

/* Pre (TODO assert): T not strided */
static inline
struct tensor tens_reshape(struct tensor *T, struct tens_shape shape)
{
    assert (tens_size(T->shape) == tens_size(shape));
    return tensor(T->arr, shape);
}

/* Slice of T (same order), starting from start, 
 * extending for shape along all axes */
static inline
struct tensor tens_sliceshape(
        struct tensor *T, const size_t *start, const size_t *shape)
{
    struct tensor S = (struct tensor) {
        .arr = tens_getp(*T, start),
        .shape.order = T->shape.order,
    };

    for (int8_t i = 0; i < T->shape.order; i++) {
        assert (shape[i] != 0);
        assert (start[i] + shape[i] <= T->shape.shape[i]);
        S.shape.shape[i] = shape[i];
        S.strides[i] = T->strides[i];
    }

    return S;
}

/* Slice of T (same order), going from start (inclusive) to end (exclusive),
 * along all axes */
static inline
struct tensor tens_slice(
        struct tensor *T, const size_t *start, const size_t *end)
{
    size_t shape[T->shape.order + 1];
    for (int8_t i = 0; i < T->shape.order; i++) {
        assert (start[i] < end[i]);
        assert (end[i] <= T->shape.shape[i]);
        shape[i] = end[i] - start[i];
    }
    return tens_sliceshape(T, start, shape);
}

/* Swap axes (like matrix transpose). Will change shape */
static inline
struct tensor tens_swapaxes(struct tensor *T, int8_t axis1, int8_t axis2)
{
    assert (axis1 < T->shape.order && axis2 < T->shape.order);

    struct tensor S = *T;
    S.shape.shape[axis1] = T->shape.shape[axis2];
    S.shape.shape[axis2] = T->shape.shape[axis1];
    S.strides[axis1] = T->strides[axis2];
    S.strides[axis2] = T->strides[axis1];
    return S;
}

/* Add an multiple axes to the tensor at dimension n. 
 * They will have length 1 */
static inline
struct tensor tens_addaxes(struct tensor *T, int8_t axis, int8_t count)
{
    assert (axis <= T->shape.order);
    assert (T->shape.order + count <= TENS_MAX_ORDER);

    if (count == 0)
        return *T;

    struct tensor S = { .arr = T->arr, .shape.order = T->shape.order + count };
    for (int8_t i = 0; i < T->shape.order + count; i++) {
        if (i < axis) {
            S.shape.shape[i] = T->shape.shape[i];
            S.strides[i] = T->strides[i];
        } else if (axis <= i && i < axis + count) {
            S.shape.shape[i] = 1;
            S.strides[i] = 0;
        } else {
            S.shape.shape[i] = T->shape.shape[i - count];
            S.strides[i] = T->strides[i - count];
        }
    }
    return S;
}

/* Broadcast two tensors together using numpy rules,
 * skipping lower n axes */
static inline
struct tens_pair tens_broadcastskipaxes(
        struct tensor *S, struct tensor *T, int8_t skip_axes)
{
    struct tens_pair out;
    /* Match orders by adding axes at the highest dimension */
    if (S->shape.order < T->shape.order) {
        out.S = tens_addaxes(S, 0, T->shape.order - S->shape.order);
        out.T = *T;
    } else if (T->shape.order < S->shape.order) {
        out.S = *S;
        out.T = tens_addaxes(T, 0, S->shape.order - T->shape.order);
    } else {
        out.S = *S;
        out.T = *T;
    }

    assert (skip_axes <= out.S.shape.order);

    for (int8_t i = 0; i < out.S.shape.order - skip_axes; i++) {
        if (out.S.shape.shape[i] == out.T.shape.shape[i])
            continue;

        if (out.S.shape.shape[i] == 1) {
            out.S.shape.shape[i] = out.T.shape.shape[i];
            out.S.strides[i] = 0;
            continue;
        }

        if (out.T.shape.shape[i] == 1) {
            out.T.shape.shape[i] = out.S.shape.shape[i];
            out.T.strides[i] = 0;
            continue;
        }
        
        assert (false);
    }

    return out;
}

/* Broadcast two tensors using numpy rules, matching all axes */
static inline
struct tens_pair tens_broadcast(struct tensor *S, struct tensor *T)
{
    return tens_broadcastskipaxes(S, T, 0);
}

/* MODIFIERS */

/* Multiply all elements of a tensor by a scalar 
 * D = lS  */
struct tensor tens_scalarmul(struct tensor *T, float l, struct tensor *D);

/* Add two matching tensors in place
 * D = S + T  */
struct tensor tens_add(struct tensor *S, struct tensor *T, struct tensor *D);

/* Matrix multiplication over the last two dimensions
 * U = S@T  */
struct tensor tens_matmul(
        struct tensor *S, struct tensor *T, struct tensor *D);
