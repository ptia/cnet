#pragma once

#include "tensor.h"
#include <stdbool.h>

/* STRUCT TENS_ITERATOR
 *
 * Iterate over scalar entries in tensor, following axes right to left.
 * Optionally, iterate skipping lower n axes
 */
struct tens_iterator {
    struct tensor tensor;
    size_t index[TENS_MAX_ORDER];   
    bool has_next;
    int8_t skip_axes;
};

/* New iterator on given tensor starting from {0, 0, ...}
 * and along the last axis (so no elements are skipped) */
static inline
struct tens_iterator tens_iterator(struct tensor tensor)
{
    return (struct tens_iterator) {
        .tensor = tensor, 
        .index = {0}, 
        .has_next = true, 
        .skip_axes = 0
    };
}

/* New iterator on given tensor starting from {0, 0, ...}
 * skipping lower n axes in iteration */
static inline
struct tens_iterator tens_iterator_skip_axes(
        struct tensor tensor, int8_t skip_axes)
{
    assert (skip_axes <= tensor.order);
    return (struct tens_iterator) {
        .tensor = tensor, 
        .index = {0}, 
        .has_next = true, 
        .skip_axes = skip_axes
    };
}

/* Return pointer to next entry and advance the iterator.
 * Return NULL if the end of the tensor has been reached */
float *tens_iter_next(struct tens_iterator *iter);
