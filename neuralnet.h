#include "tensor.h"

struct neuralnet {
    struct nn_layer *first, *last;
};

struct nn_layer {
    struct tensor (*feedforward) (
            struct nn_layer *, 
            struct tensor *data_in, struct tensor *data_out);

    struct tensor (*backprop) (
            struct nn_layer *,
            struct tensor *err_in, struct tensor *err_out);

    void (*descend) (
            struct nn_layer *,
            struct tensor *data_in, struct tensor *err_in, float eta);
};
