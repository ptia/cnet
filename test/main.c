#include "minunit.h"
#include "mat.c"
#include "tens.c"

int main()
{
    MU_RUN_SUITE(test_mat);
    MU_RUN_SUITE(test_tens);
}
