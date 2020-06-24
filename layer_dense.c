#include "layer_dense.h"
#include "util.h"
#include "tensor.h"

struct layer_dense {
    size_t units;
    struct tensor W;
    struct nn_layer nn_layer;
};

static
void init_shapes(struct nn_layer *_layer)
{
    struct layer_dense *layer = 
        getparent(_layer, struct layer_dense, nn_layer);
    
}

static
struct tensor feedforward(struct nn_layer *_layer, struct tensor in)
{
    struct layer_dense *layer = 
        getparent(_layer, struct layer_dense, nn_layer);
    return (struct tensor) {0};
}

static
struct tensor backprop(struct nn_layer *_layer, struct tensor in)
{
    struct layer_dense *layer = 
        getparent(_layer, struct layer_dense, nn_layer);
    return (struct tensor) {0};
}

static
void layer_free(struct nn_layer *_layer)
{
    struct layer_dense *layer = 
        getparent(_layer, struct layer_dense, nn_layer);
}

struct nn_layer *layer_dense(size_t units)
{
    struct layer_dense *layer = malloc(sizeof(layer_dense));
    layer->nn_layer.init_shapes = init_shapes;
    layer->nn_layer.feedforward = feedforward;
    layer->nn_layer.backprop = backprop;
    layer->nn_layer.layer_free = layer_free;
    layer->units = units;
    return &layer->nn_layer;
}

