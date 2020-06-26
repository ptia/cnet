#include "neuralnet.h"

struct dense_layer {
    union {
        size_t units;
        struct tensor weights;
    };
    bool initialised;
    struct nn_layer nn_layer;
};

struct nn_layer *dense_layer(size_t units);
