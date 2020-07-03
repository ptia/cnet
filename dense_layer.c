#include "dense_layer.h"
#include <stdlib.h>

struct dense_layer {
    struct nn_layer nn_layer;
    struct tensor weights;
    size_t units;
    
    /* Batch cache tensors */
    struct tensor Dweights, Dweights_wide;
};

static
void feedforward(
        struct nn_layer *nn_layer, 
        struct tensor *data_in, struct tensor *data_out)
{
    struct dense_layer *layer = getparent(
            nn_layer, struct dense_layer, nn_layer);

    assert (data_in->shape.order <= 2);

    if (tens_null(&layer->weights)) {
        struct tens_shape weights_shape = (struct tens_shape) { 
                2, {
                    data_in->shape.shape[data_in->shape.order - 1],
                    layer->units, 
                }};
        layer->weights = tens_zeros(weights_shape);
    }

    tens_matmul(data_in, &layer->weights, data_out);
}

static
void backprop(
        struct nn_layer *nn_layer, 
        struct tensor *err_in, struct tensor *err_out)
{
    struct dense_layer *layer = getparent(
            nn_layer, struct dense_layer, nn_layer);

    assert (err_in->shape.order <= 2);

    struct tensor weightsT = tens_transpose(&layer->weights);
    tens_matmul(err_in, &weightsT, err_out);
}

static
void descend(
        struct nn_layer *nn_layer,
        struct tensor *data_in, struct tensor *err_in, float eta)
{
    struct dense_layer *layer = getparent(
            nn_layer, struct dense_layer, nn_layer);

    assert(data_in->shape.order == 2);
    assert(err_in->shape.order == 2);

    // (err_in[:, np.newaxis, :] * data_in[:, :, np.newaxis]).sum(axis=0)
    struct tensor data_in_ = tens_addaxes(data_in, 0, 1);
    struct tensor err_in_ = tens_addaxes(err_in, 1, 1);
    tens_entrymul(&data_in_, &err_in_, &layer->Dweights_wide);
    tens_sumaxis(&layer->Dweights_wide, 0, &layer->Dweights);
    tens_scalarmul(&layer->Dweights, eta, &layer->Dweights);
    tens_add(&layer->weights, &layer->Dweights, &layer->weights);
}

struct nn_layer *dense_layer(size_t units)
{
    struct dense_layer *layer = malloc(sizeof(*layer));
    layer->units = units;
    layer->weights = TENS_NULL;
    layer->Dweights = TENS_NULL;
    layer->Dweights_wide = TENS_NULL;

    layer->nn_layer.feedforward = feedforward;
    layer->nn_layer.backprop = backprop;
    layer->nn_layer.descend = descend;
    return &layer->nn_layer;
}
