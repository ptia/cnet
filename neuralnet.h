#include "tensor.h"

struct neuralnet {
    struct nn_layer *first, *last;
};

struct nn_layer {
    void (*feedforward) (
            struct nn_layer *, 
            struct tensor *data_in, struct tensor *data_out);

    void (*backprop) (
            struct nn_layer *,
            struct tensor *err_in, struct tensor *err_out);

    void (*descend) (
            struct nn_layer *,
            struct tensor *data_in, struct tensor *err_in, float eta);
};
