#include "minunit.h"
#include "../mat.h"
#include <stdlib.h>

MU_TEST(test_mat_zeros)
{
    struct mat A = mat_zeros(2, 3);
    mu_assert_int_eq(2, A.m);
    mu_assert_int_eq(3, A.n);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            mu_assert_float_eq(0, mat_get(A, i, j));
    free(A.arr);
}

MU_TEST(test_mat_range)
{
    struct mat A = mat_range(1, 3, 0.5);
    mu_assert_int_eq(4, A.m);
    for (int i = 0; i < 4; i++)
        mu_assert_float_eq(1. + 0.5 * i, mat_get(A, i, 0));
    free(A.arr);
}

MU_TEST(test_mat_T)
{
    float arr[] = {
        1, 2,
        3, 4,
        5, 6
    };

    struct mat A = mat(arr, 3, 2);
    struct mat A_T = mat_T(A);
    struct mat A_T_T = mat_T(A_T);

    mu_assert_int_eq(2, A_T.m);
    mu_assert_int_eq(3, A_T.n);
    mu_assert_float_eq(mat_get(A, 2, 1), mat_get(A_T, 1, 2));
    mu_assert_float_eq(mat_get(A, 2, 1), mat_get(A_T_T, 2, 1));
}

MU_TEST(test_mat_slice)
{
    float arr[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    struct mat A = mat(arr, 3, 3);

    struct mat A_topleft = mat_slice(A, 0, 2, 0, 2);
    mu_assert_int_eq(2, A_topleft.m);
    mu_assert_int_eq(2, A_topleft.n);
    mu_assert_int_eq(3, A_topleft.stride);
    mu_assert_float_eq(mat_get(A, 1, 1), mat_get(A_topleft, 1, 1));

    struct mat A_right = mat_slice(A, 0, 3, 2, 3);
    mu_assert_int_eq(3, A_right.m);
    mu_assert_int_eq(1, A_right.n);
    mu_assert_float_eq(mat_get(A, 2, 2), mat_get(A_right, 2, 0));
}

MU_TEST(test_mat_mul)
{
    float arrA[] = {
        1, 2,
        3, 4,
        5, 6
    };

    float arrB[] = {
        7, 8,
        9, 10
    };

    float arrRES[] = {
        25, 28,
        57, 64,
        89, 100
    };

    struct mat A = mat(arrA, 3, 2);
    struct mat B = mat(arrB, 2, 2);
    struct mat C = mat_zeros(3, 2);
    struct mat RES = mat(arrRES, 3, 2);

    mat_mul(A, B, C);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
            mu_assert_float_eq(mat_get(RES, i, j), mat_get(C, i, j));

    free(C.arr);
}

MU_TEST_SUITE(test_mat)
{
    MU_RUN_TEST(test_mat_zeros);
    MU_RUN_TEST(test_mat_range);
    MU_RUN_TEST(test_mat_T);
    MU_RUN_TEST(test_mat_slice);
    MU_RUN_TEST(test_mat_mul);
}
