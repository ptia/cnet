#include "dense_layer.h"
#include <stdlib.h>

static
struct tensor feedforward(
        struct nn_layer *nn_layer, 
        struct tensor *data_in, struct tensor *data_out)
{
    struct dense_layer *layer = getparent(
            nn_layer, struct dense_layer, nn_layer);

    assert (data_in->shape.order <= 2);

    if (!layer->initialised) {
        layer->weights = tens_zeros((struct tens_shape) { 
                    2, {
                        data_in->shape.shape[data_in->shape.order - 1],
                        layer->units, 
                    }});
        layer->initialised = true;
    }

    return tens_matmul(data_in, &layer->weights, data_out);
}

static
struct tensor backprop(
        struct nn_layer *nn_layer, 
        struct tensor *err_in, struct tensor *err_out)
{
    struct dense_layer *layer = getparent(
            nn_layer, struct dense_layer, nn_layer);

    assert(layer->initialised);
    assert (err_in->shape.order <= 2);

    struct tensor weightsT = tens_transpose(&layer->weights);
    return tens_matmul(err_in, &weightsT, err_out);
}

static
void descend(
        struct nn_layer *nn_layer,
        struct tensor *data_in, struct tensor *err_in, float eta)
{
    struct dense_layer *layer = getparent(
            nn_layer, struct dense_layer, nn_layer);

    assert(layer->initialised);

    //TODO
}

struct nn_layer *dense_layer(size_t units)
{
    struct dense_layer *layer = malloc(sizeof(*layer));
    layer->initialised = false;
    layer->units = units;
    layer->nn_layer.feedforward = feedforward;
    layer->nn_layer.backprop = backprop;
    layer->nn_layer.descend = descend;
    return &layer->nn_layer;
}
