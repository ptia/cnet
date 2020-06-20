#include "mat.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

struct mat mat_zeros(size_t m, size_t n)
{
    return mat(calloc(1, m * n * sizeof(float)), m, n);
}

struct mat mat_range(float start, float end, float step)
{
    if (start >= end)
        return mat(NULL, 0, 0);

    size_t size = (size_t) ((end - start) / step);
    float *arr = malloc(size * sizeof(float));

    for (int i = 0; i < size; i++)
        arr[i] = start + i * step;

    return mat(arr, size, 1);
}

struct mat mat_T(struct mat A)
{
    return (struct mat) {
        .arr = A.arr,
        .m = A.n,
        .n = A.m,
        .stride = A.stride,
        .layout = A.layout == row_major ? column_major : row_major
    };
}

struct mat mat_reshape(struct mat A, size_t m, size_t n)
{
    assert (A.stride == 0); /* can't reshape strided matrices in place */
    assert (A.m * A.n == m * n);
    return (struct mat) {
        .arr = A.arr,
        .m = m,
        .n = n,
        .stride = A.stride,
        .layout = A.layout
    };
}

struct mat mat_slice(
    struct mat A, size_t starti, size_t endi, size_t startj, size_t endj)
{
    assert (starti <= endi && endi <= A.m);
    assert (startj <= endj && endj <= A.n);

    return (struct mat) {
        .arr = mat_getp(A, starti, startj), 
        .m = endi - starti,
        .n = endj - startj,
        .stride = A.stride,
        .layout = A.layout
    };

}

void mat_mul(struct mat A, struct mat B, struct mat C)
{
    assert (A.n == B.m);
    assert (C.m == A.m);
    assert (C.n == B.n);

    for (int i = 0; i < C.m; i++)
        for (int j = 0; j < C.n; j++)
            for (int k = 0; k < A.n; k++)
                *mat_getp(C, i, j) += mat_get(A, i, k) * mat_get(B, k, j);
}

void mat_print(struct mat A)
{
    for (int i = 0; i < A.m; i++) {
        for (int j = 0; j < A.n; j++)
            printf(MAT_PRINT_FORMAT, mat_get(A, i, j));
        printf("\n");
    }
}
