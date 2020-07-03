#include "activation_layer.h"

struct activation_layer {
    struct activation activation;
    struct nn_layer nn_layer;
};

static
void feedforward (
        struct nn_layer *nn_layer,
        struct tensor *data_in, struct tensor *data_out)
{
    struct activation_layer *layer = getparent(
            nn_layer, struct activation_layer, nn_layer);

    tens_apply(data_in, layer->activation.f, data_out);
}

static
void backprop (
        struct nn_layer *nn_layer,
        struct tensor *err_in, struct tensor *err_out)
{
    struct activation_layer *layer = getparent(
            nn_layer, struct activation_layer, nn_layer);

    tens_apply(err_in, layer->activation.df, err_out);
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
