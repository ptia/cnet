#include "dense_layer.h"
#include <stdlib.h>

struct dense_layer {
    struct nn_layer nn_layer;
    struct tensor W, B;
    size_t units;
    
    /* Batch cache tensors */
    struct tensor nabla_W, nabla_W_wide, nabla_B;
};

static
void feedforward(
        struct nn_layer *nn_layer, 
        struct tensor *data_in, struct tensor *data_out)
{
    struct dense_layer *layer = getparent(
            nn_layer, struct dense_layer, nn_layer);

    assert (data_in->shape.order <= 2);

    if (tens_null(&layer->W)) {
        struct tens_shape W_shape = (struct tens_shape) { 2, {
                data_in->shape.shape[data_in->shape.order - 1],
                layer->units}};
        struct tens_shape B_shape = (struct tens_shape) { 1, {layer->units} };
        layer->W = tens_zeros(W_shape);
        layer->B = tens_zeros(B_shape);
    }

    tens_matmul(data_in, &layer->W, data_out);
}

static
void backprop(
        struct nn_layer *nn_layer, 
        struct tensor *nabla_in, struct tensor *nabla_out)
{
    struct dense_layer *layer = getparent(
            nn_layer, struct dense_layer, nn_layer);

    assert (nabla_in->shape.order <= 2);

    struct tensor W_T = tens_transpose(&layer->W);
    tens_matmul(nabla_in, &W_T, nabla_out);
}

static
void descend(
        struct nn_layer *nn_layer,
        struct tensor *data_in, struct tensor *nabla_in, float eta)
{
    struct dense_layer *layer = getparent(
            nn_layer, struct dense_layer, nn_layer);

    assert(data_in->shape.order == 2);
    assert(nabla_in->shape.order == 2);

    // (nabla_in[:, np.newaxis, :] * data_in[:, :, np.newaxis]).sum(axis=0)
    struct tensor data_in_ = tens_addaxes(data_in, 0, 1);
    struct tensor nabla_in_ = tens_addaxes(nabla_in, 1, 1);
    tens_entrymul(&data_in_, &nabla_in_, &layer->nabla_W_wide);
    tens_sumaxis(&layer->nabla_W_wide, 0, &layer->nabla_W);
    tens_scalarmul(&layer->nabla_W, -eta, &layer->nabla_W);
    tens_add(&layer->W, &layer->nabla_W, &layer->W);

    tens_scalarmul(nabla_in, -eta, &layer->nabla_B);
    tens_add(&layer->B, &layer->nabla_B, &layer->B);
}

struct nn_layer *dense_layer(size_t units)
{
    struct dense_layer *layer = malloc(sizeof(*layer));
    layer->units = units;
    layer->W = TENS_NULL;
    layer->nabla_W = TENS_NULL;
    layer->nabla_W_wide = TENS_NULL;

    layer->nn_layer.feedforward = feedforward;
    layer->nn_layer.backprop = backprop;
    layer->nn_layer.descend = descend;
    return &layer->nn_layer;
}
