#include "minunit.h"
#include "../tensor.h"
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
    struct tensor A = tensor(arr, 3, (size_t []) {2, 3, 2});

    mu_assert_int_eq(2, A.shape[0]);
    mu_assert_int_eq(3, A.shape[1]);
    mu_assert_int_eq(2, A.shape[2]);

    mu_assert_float_eq(12, tens_get(A, (size_t []) {1, 2, 1}));
    mu_assert_float_eq(6, tens_get(A, (size_t []) {0, 2, 1}));
}

MU_TEST(test_tens_zeros)
{
    struct tensor T = tens_zeros(3, (size_t []) {2, 3, 3});
    
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
    struct tensor S = tensor(arr, 3, (size_t []) {2, 3, 2});
    struct tensor T = tens_reshape(S, 1, (size_t []) {2 * 3 * 2});

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

    struct tensor T = tensor(arr, 3, (size_t []) {2, 3, 2});
    struct tensor S = tens_slice(T, (size_t []) {0, 1, 0}, (size_t[]) {2, 2, 2});

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
    struct tensor T = tensor(arr, 3, (size_t []) {2, 3, 2});
    struct tensor S = tens_swap_axes(T, 0, 2);

    mu_assert_int_eq(3, S.order);
    mu_assert_int_eq(2, S.shape[0]);
    mu_assert_int_eq(3, S.shape[1]);
    mu_assert_int_eq(2, S.shape[2]);

    mu_assert_float_eq(9, tens_get(S, (size_t []) {0, 1, 1}));
}

MU_TEST(test_tens_scalar_mul)
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
    tens_scalar_mul(T, 2);

    mu_assert_float_eq(2, tens_get(T, (size_t []) {0, 0, 0}));
    mu_assert_float_eq(20, tens_get(T, (size_t []) {1, 1, 1}));
}

MU_TEST(test_tens_add)
{
    float arr1[] = {
        1, 2,
        3, 4,
        5, 6,
    };
    float arr2[] = {
        7, 8,
        9, 10,
        11, 12
    };
    struct tensor T1 = tensor(arr1, 2, (size_t []) {3, 2});
    struct tensor T2 = tensor(arr2, 2, (size_t []) {3, 2});

    tens_add(T1, T2, T2);

    mu_assert_float_eq(1, tens_get(T1, (size_t []) {0, 0}));
    mu_assert_float_eq(8, tens_get(T2, (size_t []) {0, 0}));

    mu_assert_float_eq(4, tens_get(T1, (size_t []) {1, 1}));
    mu_assert_float_eq(14, tens_get(T2, (size_t []) {1, 1}));

    mu_assert_float_eq(6, tens_get(T1, (size_t []) {2, 1}));
    mu_assert_float_eq(18, tens_get(T2, (size_t []) {2, 1}));
}

MU_TEST(test_tens_add_axes_single)
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
    struct tensor S = tens_add_axes(T, 1, 1);

    mu_assert_int_eq(4, S.order);

    mu_assert_int_eq(2, S.shape[0]);
    mu_assert_int_eq(1, S.shape[1]);
    mu_assert_int_eq(3, S.shape[2]);
    mu_assert_int_eq(2, S.shape[3]);

    mu_assert_int_eq(6, S.strides[0]);
    mu_assert_int_eq(0, S.strides[1]);
    mu_assert_int_eq(2, S.strides[2]);
    mu_assert_int_eq(1, S.strides[3]);

    mu_assert_float_eq(9, tens_get(S, (size_t []) {1, 0, 1, 0}));
}

MU_TEST(test_tens_add_axes_many)
{
    float arr[] = {
        1, 2,
        3, 4,
        5, 6,
    };
    struct tensor T = tensor(arr, 2, (size_t []) {3, 2});
    struct tensor S = tens_add_axes(T, 0, 2);

    mu_assert_int_eq(4, S.order);

    mu_assert_int_eq(1, S.shape[0]);
    mu_assert_int_eq(1, S.shape[1]);
    mu_assert_int_eq(3, S.shape[2]);
    mu_assert_int_eq(2, S.shape[3]);

    mu_assert_int_eq(0, S.strides[0]);
    mu_assert_int_eq(0, S.strides[1]);
    mu_assert_int_eq(2, S.strides[2]);
    mu_assert_int_eq(1, S.strides[3]);

    mu_assert_float_eq(3, tens_get(S, (size_t []) {0, 0, 1, 0}));
}

MU_TEST(test_tens_broadcast)
{
    struct tensor S = tensor(NULL, 4, (size_t []) {3, 2, 1, 7});
    struct tensor T = tensor(NULL, 3, (size_t []) {   2, 5, 7});

    struct tens_pair ST = tens_broadcast(S, T);

    mu_assert_int_eq(3, ST.T.shape[0]);
    mu_assert_int_eq(2, ST.T.shape[1]);
    mu_assert_int_eq(5, ST.S.shape[2]);
    mu_assert_int_eq(7, ST.T.shape[3]);
}

MU_TEST(test_tens_broadcast_skip_axes)
{
    struct tensor S = tensor(NULL, 4, (size_t []) {3, 2, 1, 7});
    struct tensor T = tensor(NULL, 3, (size_t []) {   2, 5, 3});

    struct tens_pair ST = tens_broadcast_skip_axes(S, T, 1);

    mu_assert_int_eq(3, ST.T.shape[0]);
    mu_assert_int_eq(2, ST.T.shape[1]);
    mu_assert_int_eq(5, ST.S.shape[2]);
    mu_assert_int_eq(7, ST.S.shape[3]);
    mu_assert_int_eq(3, ST.T.shape[3]);
}

MU_TEST(test_tens_mat_mul)
{
    float arra[] = {
        0, 1, 2,
        3, 4, 5
    };
    float arrb[] = {
        0, 1,
        2, 3,
        4, 5
    };
    struct tensor A = tensor(arra, 2, (size_t []) {2, 3});
    struct tensor B = tensor(arrb, 2, (size_t []) {3, 2});
    struct tensor D = tens_zeros(2, (size_t []) {2, 2});

    tens_mat_mul(A, B, D);
    mu_assert_float_eq(10, tens_get(D, (size_t []) {0, 0}));
    mu_assert_float_eq(13, tens_get(D, (size_t []) {0, 1}));
    mu_assert_float_eq(28, tens_get(D, (size_t []) {1, 0}));
    mu_assert_float_eq(40, tens_get(D, (size_t []) {1, 1}));
}

MU_TEST(test_tens_mat_mul_broadcast)
{
    float arra[] = {
        0, 1, 2,
        3, 4, 5,

        6, 7, 8,
        9, 10, 11
    };
    float arrb[] = {
        0, 1,
        2, 3,
        4, 5
    };
    struct tensor A = tensor(arra, 3, (size_t []) {2, 2, 3});
    struct tensor B = tensor(arrb, 2, (size_t []) {3, 2});
    struct tensor D = tens_zeros(3, (size_t []) {2, 2, 2});

    tens_mat_mul(A, B, D);
    mu_assert_float_eq(10, tens_get(D, (size_t []) {0, 0, 0}));
    mu_assert_float_eq(13, tens_get(D, (size_t []) {0, 0, 1}));
    mu_assert_float_eq(28, tens_get(D, (size_t []) {0, 1, 0}));
    mu_assert_float_eq(40, tens_get(D, (size_t []) {0, 1, 1}));

    mu_assert_float_eq(46, tens_get(D, (size_t []) {1, 0, 0}));
    mu_assert_float_eq(67, tens_get(D, (size_t []) {1, 0, 1}));
    mu_assert_float_eq(64, tens_get(D, (size_t []) {1, 1, 0}));
    mu_assert_float_eq(94, tens_get(D, (size_t []) {1, 1, 1}));
}

MU_TEST_SUITE(test_tensor)
{
    MU_RUN_TEST(test_tens_get);
    MU_RUN_TEST(test_tens_zeros);
    MU_RUN_TEST(test_tens_reshape);
    MU_RUN_TEST(test_tens_slice);
    MU_RUN_TEST(test_tens_swap_axes);
    MU_RUN_TEST(test_tens_add_axes_single);
    MU_RUN_TEST(test_tens_add_axes_many);
    MU_RUN_TEST(test_tens_broadcast);
    MU_RUN_TEST(test_tens_broadcast_skip_axes);
    MU_RUN_TEST(test_tens_scalar_mul);
    MU_RUN_TEST(test_tens_add);
    MU_RUN_TEST(test_tens_mat_mul);
    MU_RUN_TEST(test_tens_mat_mul_broadcast);
}
