#include "../tens.h"
#include "minunit.h"

MU_TEST(test_tens_get)
{
    float arr[] = {
        1, 2,
        3, 4,
        5, 6
    };
    struct tens A = tens(arr, {1, 1, 3,2});

    mu_assert_int_eq(1, A.shape[0]);
    mu_assert_int_eq(1, A.shape[1]);
    mu_assert_int_eq(3, A.shape[2]);
    mu_assert_int_eq(2, A.shape[3]);

    mu_assert_float_eq(6, tens_get(A, tens_index(0, 0, 2, 1)));
}

MU_TEST_SUITE(test_tens)
{
    MU_RUN_TEST(test_tens_get);
}
