#include "activation_layer.h"

struct activation_layer {
    struct activation activation;
    struct nn_layer nn_layer;
};

static
void feedforward (struct nn_layer *nn_layer, struct tensor *X)
{
    struct activation_layer *layer = getparent(
            nn_layer, struct activation_layer, nn_layer);

    tens_apply(X, layer->activation.f, &nn_layer->Z);
}

static
void backprop (struct nn_layer *nn_layer, struct tensor *nabla_Z)
{
    struct activation_layer *layer = getparent(
            nn_layer, struct activation_layer, nn_layer);

    tens_apply(nabla_Z, layer->activation.df, &nn_layer->nabla_X);
}

struct nn_layer *activation_layer(struct activation activation)
{
    struct activation_layer *layer = malloc(sizeof(*layer));
    layer->activation = activation;
    layer->nn_layer.feedforward = feedforward;
    layer->nn_layer.backprop = backprop;
    layer->nn_layer.descend = NULL;
    return &layer->nn_layer;
}
