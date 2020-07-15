#pragma once

#include "tensor.h"

struct neuralnet {
    struct nn_layer *first, *last;
};


struct nn_layer {
    struct tensor Z, nabla_X;
    struct nn_layer *prev, *next;

    void (*feedforward) (struct nn_layer *, struct tensor *X);

    void (*backprop) (struct nn_layer *, struct tensor *nabla_Z);

    void (*descend) (
            struct nn_layer *,
            struct tensor *X, struct tensor *nabla_Z, float eta);

    void (*free_batch) (struct nn_layer *);
    
    void (*free_layer) (struct nn_layer *);
};

void nn_addlayer(struct neuralnet *, struct nn_layer *);
struct tensor nn_eval(struct neuralnet *, struct tensor *);
