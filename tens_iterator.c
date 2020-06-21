#include "tens_iterator.h"

float *tens_iter_next(struct tens_iterator *iter)
{
    int8_t i = iter->tensor.order - i;
    do {
        if (i < 0)
            return NULL;
        iter->index[i] = (iter->index[i] + 1) % iter->tensor.shape[i];
    } while (iter->index[i] == 0);
    return tens_getp(iter->tensor, iter->index);
}
