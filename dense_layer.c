#include "dense_layer.h"
#include <stdlib.h>

static
struct tensor feedforward(
        struct nn_layer *nn_layer, struct tensor *in, struct tensor *out)
{
    struct dense_layer *layer = getparent(
            nn_layer, struct dense_layer, nn_layer);

    if (!layer->initialised) {
        layer->weights = tens_zeros((struct tens_shape) { 2,
                    {layer->units, in->shape.shape[in->shape.order - 1]}
                });
        layer->initialised = true;
    }

    return tens_matmul(in, &layer->weights, out);
}

struct nn_layer *dense_layer(size_t units)
{
    struct dense_layer *layer = malloc(sizeof(struct dense_layer));
    layer->initialised = false;
    layer->units = units;
    layer->nn_layer.feedforward = feedforward;
    return &layer->nn_layer;
}
