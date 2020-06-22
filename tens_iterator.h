#pragma once

#include "tensor.h"
#include <stdbool.h>

/* STRUCT TENS_ITERATOR
 *
 * Iterate over scalar entries in tensor, following axes right to left.
 * Optionally, iterate starting from a custom axis, skipping lowe axes
 * and skipping elements.
 */
struct tens_iterator {
    struct tensor tensor;
    size_t index[TENS_MAX_ORDER];   
    bool has_next;
    int8_t axis;
};

/* New iterator on given tensor starting from the start
 * and along the last axis (so no elements are skipped) */
static inline
struct tens_iterator tens_iterator(struct tensor tensor)
{
    return (struct tens_iterator) {
        .tensor = tensor, 
        .index = {0}, 
        .has_next = true, 
        .axis = tensor.order - 1
    };
}

/* New iterator on given tensor starting from the start
 * along a custom axis, skipping lower axes in iteration */
static inline
struct tens_iterator tens_iterator_axis(struct tensor tensor, int8_t axis)
{
    return (struct tens_iterator) {
        .tensor = tensor, 
        .index = {0}, 
        .has_next = true, 
        .axis = axis
    };
}

/* Return pointer to next entry and advance the iterator.
 * Return NULL if the end of the tensor has been reached */
float *tens_iter_next(struct tens_iterator *iter);
