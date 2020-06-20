#include "minunit.h"
#include "../tens.h"
#include <stdlib.h>

MU_TEST(test_tens_get)
{
    float arr[] = {
        1, 2,
        3, 4,
        5, 6,

        7, 8,
        9, 10,
        11, 12
    };
    struct tens A = tens(arr, 3, (size_t []) {2, 3, 2});

    mu_assert_int_eq(2, A.shape[0]);
    mu_assert_int_eq(3, A.shape[1]);
    mu_assert_int_eq(2, A.shape[2]);

    mu_assert_float_eq(12, tens_get(A, (size_t []) {1, 2, 1}));
}

MU_TEST(test_tens_zeros)
{
    struct tens T = tens_zeros(3, (size_t []) {2, 3, 3});
    
    mu_assert_int_eq(2, T.shape[0]);
    mu_assert_int_eq(3, T.shape[1]);
    mu_assert_int_eq(3, T.shape[2]);

    mu_assert_float_eq(0, tens_get(T, (size_t []) {1, 2, 2}));
    free(T.arr);
}


MU_TEST_SUITE(test_tens)
{
    MU_RUN_TEST(test_tens_get);
    MU_RUN_TEST(test_tens_zeros);
}
