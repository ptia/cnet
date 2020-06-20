#include "tens.h"

struct tens tens(float *arr, uint8_t order, size_t *shape)
{
    assert (order <= TENS_MAX_ORDER);

    struct tens out = (struct tens) {.arr = arr, .order = order};
    
    size_t stride = 1;
    for (int8_t i = order - 1; i >= 0; i--) {
        out.strides[i] = stride;
        out.shape[i] = shape[i];
        stride *= shape[i];
    }

    return out;
}
