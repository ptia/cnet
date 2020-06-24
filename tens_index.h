#pragma once

#include "tensor.h"
#include <stdbool.h>

struct tens_index {
    size_t index[TENS_MAX_ORDER];   
    size_t shape[TENS_MAX_ORDER];
    const int8_t order;
};

static inline
struct tens_index tens_index(const size_t *shape, int8_t order)
{
    struct tens_index out = (struct tens_index) { 
        .index = {0},
        .order = order
    };
    for (int8_t i = 0; i < order; i++)
        out.shape[i] = shape[i];
    return out;
}

bool tens_index_next(struct tens_index *index);
bool tens_index_nextaxis(struct tens_index *index, int8_t axis);
