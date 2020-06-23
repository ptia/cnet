#pragma once

#include "tensor.h"
#include <stdbool.h>

struct tensindex {
    size_t index[TENS_MAX_ORDER];   
    size_t shape[TENS_MAX_ORDER];
    const int8_t order;
};

static inline
struct tensindex tensindex(const size_t *shape, int8_t order)
{
    struct tensindex out = (struct tensindex) { 
        .index = {0},
        .order = order
    };
    for (int8_t i = 0; i < order; i++)
        out.shape[i] = shape[i];
    return out;
}

bool tensindex_next(struct tensindex *index);
bool tensindex_nextaxis(struct tensindex *index, int8_t axis);
