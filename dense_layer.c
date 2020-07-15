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
void feedforward(struct nn_layer *nn_layer, struct tensor *X)
{
    struct dense_layer *layer = getparent(
            nn_layer, struct dense_layer, nn_layer);

    assert (X->shape.order == 2);

    if (tens_null(&layer->W)) {
        struct tens_shape W_shape = (struct tens_shape) { 2, {
                X->shape.shape[1],
                layer->units}};
        struct tens_shape B_shape = (struct tens_shape) { 1, {layer->units} };
        layer->W = tens_randn(W_shape);
        layer->B = tens_randn(B_shape);
    }

    tens_matmul(X, &layer->W, &nn_layer->Z);
}

static
void backprop( struct nn_layer *nn_layer, struct tensor *nabla_Z)
{
    struct dense_layer *layer = getparent(
            nn_layer, struct dense_layer, nn_layer);

    assert (nabla_Z->shape.order == 2);

    struct tensor W_T = tens_transpose(&layer->W);
    tens_matmul(nabla_Z, &W_T, &nn_layer->nabla_X);
}

static
void descend(
        struct nn_layer *nn_layer,
        struct tensor *X, struct tensor *nabla_Z, float eta)
{
    struct dense_layer *layer = getparent(
            nn_layer, struct dense_layer, nn_layer);

    assert(X->shape.order == 2);
    assert(nabla_Z->shape.order == 2);

    // (nabla_Z[:, np.newaxis, :] * X[:, :, np.newaxis]).sum(axis=0)
    struct tensor X_ = tens_addaxes(X, 0, 1);
    struct tensor nabla_Z_ = tens_addaxes(nabla_Z, 1, 1);
    tens_entrymul(&X_, &nabla_Z_, &layer->nabla_W_wide);
    tens_sumaxis(&layer->nabla_W_wide, 0, &layer->nabla_W);
    tens_scalarmul(&layer->nabla_W, -eta, &layer->nabla_W);
    tens_add(&layer->W, &layer->nabla_W, &layer->W);

    tens_scalarmul(nabla_Z, -eta, &layer->nabla_B);
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
