#include "minunit.h"
#include "../dense_layer.h"

MU_TEST(test_dense_feedforward_single)
{
    struct nn_layer *layer = dense_layer(4);
    struct tensor data_in = tens_range(0, 15, 1);
    data_in = tens_addaxes(&data_in, 0, 1);
    layer->feedforward(layer, &data_in);
    mu_check(tens_match((struct tens_shape) {2, {1, 4}}, 
                layer->Z.shape));
}

MU_TEST(test_dense_feedforward_batch)
{
    struct nn_layer *layer = dense_layer(4);
    struct tensor data_in = tens_range(0, 45, 1);
    data_in = tens_reshape(&data_in, (struct tens_shape) {2, {3, 15}});
    layer->feedforward(layer, &data_in);
    mu_check(tens_match((struct tens_shape) {2, {3, 4}}, 
                layer->Z.shape));
}

MU_TEST_SUITE(test_dense_layer)
{
    MU_RUN_TEST(test_dense_feedforward_single);
    MU_RUN_TEST(test_dense_feedforward_batch);
}
