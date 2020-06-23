#include "minunit.h"
#include "mat.c"
#include "tensor.c"
#include "tensindex.c"

int main()
{
    MU_RUN_SUITE(test_mat);
    MU_RUN_SUITE(test_tensor);
    MU_RUN_SUITE(test_tensindex);
}
