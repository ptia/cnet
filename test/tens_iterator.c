#include "minunit.h"
#include "../tens_iterator.h"
#include <stdlib.h>

MU_TEST(test_tens_iter_next)
{
    float arr[] = {
        1, 2,
        3, 4,
        5, 6,

        7, 8,
        9, 10,
        11, 12
    };

    struct tens T = tens(arr, 3, (size_t []) {2, 3, 2});
    struct tens_iterator iter = tens_iterator(T);

    for (int i = 0; i < 12; i++)
        mu_assert_float_eq(arr[i], *tens_iter_next(&iter));
    mu_assert_ptr_eq(NULL, tens_iter_next(&iter));
}

MU_TEST_SUITE(test_tens_iterator)
{
    MU_RUN_TEST(test_tens_iter_next);
}
