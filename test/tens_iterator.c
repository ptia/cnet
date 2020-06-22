#include "minunit.h"
#include "../tens_iterator.h"
#include <stdlib.h>

MU_TEST(test_tens_iter_all_axes)
{
    float arr[] = {
        1, 2,
        3, 4,
        5, 6,

        7, 8,
        9, 10,
        11, 12
    };
    struct tensor T = tensor(arr, 3, (size_t []) {2, 3, 2});
    struct tens_iterator iter = tens_iterator(T);

    for (int i = 0; i < 12; i++)
        mu_assert_float_eq(arr[i], *tens_iter_next(&iter));
    mu_assert_ptr_eq(NULL, tens_iter_next(&iter));
}

MU_TEST(test_tens_iterator_skip_axes)
{
    float arr[] = {
        1, 2,
        3, 4,
        5, 6,

        7, 8,
        9, 10,
        11, 12
    };
    struct tensor T = tensor(arr, 3, (size_t []) {2, 3, 2});
    struct tens_iterator iter = tens_iterator_skip_axes(T, 1);

    mu_assert_float_eq(1, *tens_iter_next(&iter));
    mu_assert_float_eq(3, *tens_iter_next(&iter));
    mu_assert_float_eq(5, *tens_iter_next(&iter));
    mu_assert_float_eq(7, *tens_iter_next(&iter));
    mu_assert_float_eq(9, *tens_iter_next(&iter));
    mu_assert_float_eq(11, *tens_iter_next(&iter));
    mu_assert_ptr_eq(NULL, tens_iter_next(&iter));
}


MU_TEST_SUITE(test_tens_iterator)
{
    MU_RUN_TEST(test_tens_iter_all_axes);
    MU_RUN_TEST(test_tens_iterator_skip_axes);
}
