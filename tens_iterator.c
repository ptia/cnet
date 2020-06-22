#include "tens_iterator.h"

float *tens_iter_next(struct tens_iterator *iter)
{
    if (!iter->has_next)
        return NULL;

    float *out = tens_getp(iter->tensor, iter->index);

    int8_t i = iter->axis + 1;
    do {
        if (i == 0)
            iter->has_next = false;
        i--;
        iter->index[i] = (iter->index[i] + 1) % iter->tensor.shape[i];
    } while (iter->index[i] == 0);

    return out;
}
