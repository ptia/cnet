#include "minunit.h"
#include "tensor.c"
#include "tensindex.c"

int main()
{
    MU_RUN_SUITE(test_tensor);
    MU_RUN_SUITE(test_tensindex);
}
