#include "minunit.h"
#include "../dense_layer.h"

MU_TEST(test_dense_feedforward_single)
{
    struct nn_layer *layer = dense_layer(4);
    struct tensor data_in = tens_range(0, 15, 1);
    layer->feedforward(layer, &data_in);
}

MU_TEST_SUITE(test_dense_layer)
{
    MU_RUN_TEST(test_dense_feedforward_single);
}
