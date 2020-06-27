#include "activation_layer.h"

static
struct tensor feedforward(
        struct nn_layer *nn_layer, struct tensor *in, struct tensor *out)
{
    struct activation_layer *layer = getparent(
            nn_layer, struct activation_layer, nn_layer);

    return tens_apply(in, layer->activation.f, out);
}

static
struct tensor backprop(
        struct nn_layer *nn_layer, struct tensor *in, struct tensor *out)
{
    struct activation_layer *layer = getparent(
            nn_layer, struct activation_layer, nn_layer);

    return tens_apply(in, layer->activation.df, out);
}

struct nn_layer *activation_layer(struct activation activation)
{
    struct activation_layer *layer = malloc(sizeof(*layer));
    layer->activation = activation;
    layer->nn_layer.feedforward = feedforward;
    layer->nn_layer.backprop = backprop;
    return &layer->nn_layer;
}

