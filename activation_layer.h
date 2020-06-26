#include "neuralnet.h"
#include <stdlib.h>

struct activation_layer {
    float (*activation) (float);
    struct nn_layer nn_layer;
};

static
struct tensor feedforward(
        struct nn_layer *nn_layer, struct tensor *in, struct tensor *out)
{
    struct activation_layer *layer = getparent(
            nn_layer, struct activation_layer, nn_layer);

    return tens_apply(in, layer->activation, out);
}

struct nn_layer *activation_layer(float (*activation) (float))
{
    struct activation_layer *layer = malloc(sizeof(*layer));
    layer->activation = activation;
    layer->nn_layer.feedforward = feedforward;
    return &layer->nn_layer;
}
