#include "tens_index.h"

bool tens_index_next(struct tens_index *index)
{
    return tens_index_nextaxis(index, index->order - 1);
}

bool tens_index_nextaxis(struct tens_index *index, int8_t axis)
{
    assert (axis <= index->order);
    int8_t i = axis + 1;
    do {
        if (i == 0)
            return false;
        i--;
        index->index[i] = (index->index[i] + 1) % index->shape[i];
    } while (index->index[i] == 0);
    return true;
}
