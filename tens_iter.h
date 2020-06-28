#include "tensor.h"
#include "tens_index.h"

struct tens_iter {
    bool continuous;
    union {
        float *next_ptr;
        struct tens_index next_index;
    };
};
