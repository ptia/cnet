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
    mu_assert_float_eq(6, tens_get(A, (size_t []) {0, 2, 1}));
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

MU_TEST(test_tens_reshape)
{
    float arr[] = {
        1, 2,
        3, 4,
        5, 6,

        7, 8,
        9, 10,
        11, 12
    };
    struct tens S = tens(arr, 3, (size_t []) {2, 3, 2});
    struct tens T = tens_reshape(S, 1, (size_t []) {2 * 3 * 2});

    mu_assert_float_eq(6, tens_get(S, (size_t []) {0, 2, 1}));
    mu_assert_float_eq(6, tens_get(T, (size_t []) {5}));

    *tens_getp(S, (size_t []) {0, 2, 1}) = -6;

    mu_assert_float_eq(-6, tens_get(S, (size_t []) {0, 2, 1}));
    mu_assert_float_eq(-6, tens_get(T, (size_t []) {5}));
}

MU_TEST(test_tens_slice)
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
    struct tens S = tens_slice(T, (size_t []) {0, 1, 0}, (size_t[]) {2, 2, 2});

    mu_assert_int_eq(3, S.order);
    mu_assert_int_eq(2, S.shape[0]);
    mu_assert_int_eq(1, S.shape[1]);
    mu_assert_int_eq(2, S.shape[2]);
    
    mu_assert_float_eq(9, tens_get(S, (size_t []) {1, 0, 0}));
}

MU_TEST(test_tens_swap_axes)
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
    struct tens S = tens_swap_axes(T, 0, 2);

    mu_assert_int_eq(3, S.order);
    mu_assert_int_eq(2, S.shape[0]);
    mu_assert_int_eq(3, S.shape[1]);
    mu_assert_int_eq(2, S.shape[2]);

    mu_assert_float_eq(9, tens_get(S, (size_t []) {0, 1, 1}));
}

MU_TEST_SUITE(test_tens)
{
    MU_RUN_TEST(test_tens_get);
    MU_RUN_TEST(test_tens_zeros);
    MU_RUN_TEST(test_tens_reshape);
    MU_RUN_TEST(test_tens_slice);
    MU_RUN_TEST(test_tens_swap_axes);
}
