#pragma once

#include "tensor.h"

struct neuralnet {
    struct nn_layer *first, *last;
};

struct nn_layer {
    struct tensor data_out, nabla_out;
    struct nn_layer *prev, *next;

    void (*feedforward) (struct nn_layer *, struct tensor *data_in);

    void (*backprop) (struct nn_layer *, struct tensor *nabla_in);

    void (*descend) (
            struct nn_layer *,
            struct tensor *data_in, struct tensor *nabla_in, float eta);
};
