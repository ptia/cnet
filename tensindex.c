#include "tensindex.h"

bool tensindex_next(struct tensindex *index)
{
    return tensindex_nextaxis(index, index->order - 1);
}

bool tensindex_nextaxis(struct tensindex *index, int8_t axis)
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
