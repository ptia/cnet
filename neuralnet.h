#include "tensor.h"

struct neuralnet {
    struct nn_layer *first, *last;
};

struct nn_layer {
    struct tensor (*feedforward) (
            struct nn_layer *, struct tensor *, struct tensor *);
    struct tensor (*backprop) (
            struct nn_layer *, struct tensor *, struct tensor *);
};
