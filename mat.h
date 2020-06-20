#pragma once
#include <stdbool.h>
#include <assert.h>
#include <stddef.h>

#define MAT_PRINT_FORMAT "%.3g " 

/* Each instance is an immutable view on underlying data array
 * Multiple views can share the same data, changes to it are shared
 * Only mutable part of a struct mat is the matrix data
 * Row-major by default */
struct mat {
    float *const arr;
    const size_t m, n;
    const size_t stride;
    const enum {
        row_major, column_major
    } layout;
};

static inline
struct mat mat(float *arr, size_t m, size_t n)
{
    return (struct mat) {arr, m, n, n, .layout=row_major};
}

static inline
float *mat_getp(struct mat A, size_t i, size_t j)
{
    assert (i < A.m);
    assert (j < A.n);

    if (A.layout == row_major)
        return &(A.arr[i * A.stride + j]);

    if (A.layout == column_major)
        return &(A.arr[j * A.stride + i]);

    assert (false);
}

static inline
float mat_get(struct mat A, size_t i, size_t j)
{
    return *mat_getp(A, i, j);
}

/* Create new matrices (malloc'ing) */
struct mat mat_zeros(size_t m, size_t n);
struct mat mat_range(float start, float end, float step);

/* New view, same data */
struct mat mat_T(struct mat A);
struct mat mat_reshape(struct mat A, size_t m, size_t n);
struct mat mat_slice(
    struct mat A, size_t starti, size_t endi, size_t startj, size_t endj);

/* Change data */
void mat_mul(struct mat A, struct mat B, struct mat C); /* C = AB + C */

/* Info about matrix */
void mat_print(struct mat A);
