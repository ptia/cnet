#include "tens.h"
#include <stdlib.h>

struct tens tens_zeros(uint8_t order, size_t *shape)
{
    size_t size = 1;
    for (uint8_t i = 0; i < order ; i++) {
        size *= shape[i];
    }
    return tens(calloc(1, size * sizeof(float)), order, shape);
}
