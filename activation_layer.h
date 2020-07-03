#pragma once

#include "neuralnet.h"
#include <stdlib.h>

struct activation {
    float (*f) (float);
    float (*df) (float);
};

struct nn_layer *activation_layer(struct activation activation);
