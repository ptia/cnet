#include "neuralnet.h"

void nn_addlayer(struct neuralnet *net, struct nn_layer *layer, struct activation af)
{
    layer->af = af;
    layer->A = TENS_NULL;
    layer->prev = net->last;
    layer->next = NULL;
    if (net->last)
        net->last->next = layer;
    else
        net->first = layer;
    net->last = layer;
}

struct tensor nn_eval(struct neuralnet *net, struct tensor *X)
{
    for (struct nn_layer *l = net->first; l; l = l->next) {
        if (l == net->first)
            l->feedforward(l, X);
        else
            l->feedforward(l, &l->prev->A);
        tens_map(&l->Z, l->af.f, &l->A);
        
    }
    return net->last->Z;
}
