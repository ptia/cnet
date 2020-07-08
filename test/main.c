#include "minunit.h"
#include "tensor.c"
#include "tens_index.c"
#include "dense_layer.c"

int main()
{
    MU_RUN_SUITE(test_tensor);
    MU_RUN_SUITE(test_tens_index);
    MU_RUN_SUITE(test_dense_layer);
}
