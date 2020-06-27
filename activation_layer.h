#include "neuralnet.h"
#include <stdlib.h>

struct activation {
    float (*f) (float);
    float (*df) (float);
};

struct activation_layer {
    struct activation activation;
    struct nn_layer nn_layer;
};

struct nn_layer *activation_layer(struct activation activation);
