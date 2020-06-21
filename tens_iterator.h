#pragma once

#include "tens.h"
#include <stdbool.h>

/* Iterate over scalar entries in tensor, 
 * following the natural axis ordering */
struct tens_iterator {
    struct tens tensor;
    size_t index[TENS_MAX_ORDER];   
    bool has_next;
};

/* New iterator on given tensor starting from the start */
static inline
struct tens_iterator tens_iterator(struct tens tensor)
{
    return (struct tens_iterator) {
        .tensor = tensor, .index = {0}, .has_next = true
    };
}

/* Return pointer to next entry and advance the iterator.
 * Return NULL if the end of the tensor has been reached */
float *tens_iter_next(struct tens_iterator *iter);
