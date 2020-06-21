#include "minunit.h"
#include "mat.c"
#include "tens.c"
#include "tens_iterator.c"

int main()
{
    MU_RUN_SUITE(test_mat);
    MU_RUN_SUITE(test_tens);
    MU_RUN_SUITE(test_tens_iterator);
}
