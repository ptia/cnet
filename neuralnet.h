#include "tensor.h"

struct neuralnet {
    struct nn_layer *first, *last;
};

struct nn_layer {
    void (*init_shapes) (struct nn_layer *);
    struct tensor (*feedforward) (struct nn_layer *, struct tensor);
    struct tensor (*backprop) (struct nn_layer *, struct tensor);
    void (*layer_free) (struct nn_layer *);
    struct tens_shape shape_in, shape_out;
    struct nn_layer *next;
};

void nn_addlayer(struct neuralnet *nn, struct nn_layer *layer);
