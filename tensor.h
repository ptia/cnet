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
 * No functions here change the view passed to them (some change the data).
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
 * All these functions take struct tensor by pointer to save on copying 
 * arguments, but they won't change the original view.
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

/* Tensor from raw data array (in-place, does not malloc), default layout */
struct tensor tensor(float *arr, struct tens_shape shape);

/* New zero-initialised tensor (calloc'ing, remember to free .arr) */
struct tensor tens_zeros(struct tens_shape shape);


/* VIEWS */

/* Pre (TODO assert): T not strided */
struct tensor tens_reshape(struct tensor *T, struct tens_shape shape);

/* Slice of T (same order), from start, 
 * extending for shape along all axes */
struct tensor tens_sliceshape(
        struct tensor *T, const size_t *start, const size_t *shape);

/* Slice of T (same order), going from start (inclusive) to end (exclusive),
 * along all axes */
struct tensor tens_slice(
        struct tensor *T, const size_t *start, const size_t *end);

/* Swap axes (like matrix transpose). Will change shape */
struct tensor tens_swapaxes(struct tensor *T, int8_t axis1, int8_t axis2);

/* Add an multiple axes to the tensor at dimension n. 
 * They will have length 1 */
struct tensor tens_addaxes(struct tensor *T, int8_t axis, int8_t count);

/* Broadcast two tensors together using numpy rules,
 * skipping lower n axes */
struct tens_pair tens_broadcastskipaxes(
        struct tensor *S, struct tensor *T, int8_t skip_axes);

/* Broadcast two tensors using numpy rules, matching all axes */
struct tens_pair tens_broadcast(struct tensor *S, struct tensor *T);

/* MODIFIERS */

/* Apply function to all scalar elements */
struct tensor tens_apply(
        struct tensor *T, float (*f) (float), struct tensor *D);
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
