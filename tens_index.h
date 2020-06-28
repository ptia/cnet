#pragma once

#include "tensor.h"
#include <stdbool.h>

struct tens_index {
    size_t index[TENS_MAX_ORDER];   
    struct tens_shape shape;
};

static inline
struct tens_index tens_index(struct tens_shape shape)
{
    struct tens_index out = (struct tens_index) { 
        .index = {0},
        .shape = shape
    };
    return out;
}

bool tens_index_next(struct tens_index *index);
bool tens_index_nextaxis(struct tens_index *index, int8_t axis);
