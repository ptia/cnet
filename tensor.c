#include "tensor.h"
#include "tens_index.h"
#include <assert.h>

/* CREATORS */

struct tensor tensor(float *arr, struct tens_shape shape)
{
    assert (shape.order <= TENS_MAX_ORDER);

    struct tensor out = (struct tensor) { .arr = arr, .shape = shape};
    
    size_t stride = 1;
    for (int8_t i = shape.order - 1; i >= 0; i--) {
        assert (shape.shape[i] != 0);
        out.strides[i] = stride;
        stride *= shape.shape[i];
    }

    return out;
}

struct tensor tens_zeros(struct tens_shape shape)
{
    return tensor(calloc(1, tens_size(shape) * sizeof(float)), shape);
}


/* VIEWS */

struct tensor tens_reshape(struct tensor *T, struct tens_shape shape)
{
    assert (tens_size(T->shape) == tens_size(shape));
    return tensor(T->arr, shape);
}

struct tensor tens_sliceshape(
        struct tensor *T, const size_t *start, const size_t *shape)
{
    struct tensor S = (struct tensor) {
        .arr = tens_getp(*T, start),
        .shape.order = T->shape.order,
    };

    for (int8_t i = 0; i < T->shape.order; i++) {
        assert (shape[i] != 0);
        assert (start[i] + shape[i] <= T->shape.shape[i]);
        S.shape.shape[i] = shape[i];
        S.strides[i] = T->strides[i];
    }

    return S;
}

struct tensor tens_swapaxes(struct tensor *T, int8_t axis1, int8_t axis2)
{
    assert (axis1 < T->shape.order && axis2 < T->shape.order);

    struct tensor S = *T;
    S.shape.shape[axis1] = T->shape.shape[axis2];
    S.shape.shape[axis2] = T->shape.shape[axis1];
    S.strides[axis1] = T->strides[axis2];
    S.strides[axis2] = T->strides[axis1];
    return S;
}

struct tensor tens_transpose(struct tensor *T)
{
    assert (T->shape.order >= 2);
    return tens_swapaxes(T, T->shape.order - 1, T->shape.order - 2);
}

struct tensor tens_slice(
        struct tensor *T, const size_t *start, const size_t *end)
{
    size_t shape[T->shape.order + 1];
    for (int8_t i = 0; i < T->shape.order; i++) {
        assert (start[i] < end[i]);
        assert (end[i] <= T->shape.shape[i]);
        shape[i] = end[i] - start[i];
    }
    return tens_sliceshape(T, start, shape);
}


struct tensor tens_addaxes(struct tensor *T, int8_t axis, int8_t count)
{
    assert (axis <= T->shape.order);
    assert (T->shape.order + count <= TENS_MAX_ORDER);

    if (count == 0)
        return *T;

    struct tensor S = { .arr = T->arr, .shape.order = T->shape.order + count };
    for (int8_t i = 0; i < T->shape.order + count; i++) {
        if (i < axis) {
            S.shape.shape[i] = T->shape.shape[i];
            S.strides[i] = T->strides[i];
        } else if (axis <= i && i < axis + count) {
            S.shape.shape[i] = 1;
            S.strides[i] = 0;
        } else {
            S.shape.shape[i] = T->shape.shape[i - count];
            S.strides[i] = T->strides[i - count];
        }
    }
    return S;
}

struct tens_pair tens_broadcastskipaxes(
        struct tensor *S, struct tensor *T, int8_t skip_axes)
{
    struct tens_pair out;
    /* Match orders by adding axes at the highest dimension */
    if (S->shape.order < T->shape.order) {
        out.S = tens_addaxes(S, 0, T->shape.order - S->shape.order);
        out.T = *T;
    } else if (T->shape.order < S->shape.order) {
        out.S = *S;
        out.T = tens_addaxes(T, 0, S->shape.order - T->shape.order);
    } else {
        out.S = *S;
        out.T = *T;
    }

    assert (skip_axes <= out.S.shape.order);

    for (int8_t i = 0; i < out.S.shape.order - skip_axes; i++) {
        if (out.S.shape.shape[i] == out.T.shape.shape[i])
            continue;

        if (out.S.shape.shape[i] == 1) {
            out.S.shape.shape[i] = out.T.shape.shape[i];
            out.S.strides[i] = 0;
            continue;
        }

        if (out.T.shape.shape[i] == 1) {
            out.T.shape.shape[i] = out.S.shape.shape[i];
            out.T.strides[i] = 0;
            continue;
        }
        
        assert (false);
    }

    return out;
}

struct tens_pair tens_broadcast(struct tensor *S, struct tensor *T)
{
    return tens_broadcastskipaxes(S, T, 0);
}


/* MODIFIERS */

struct tensor tens_apply(
        struct tensor *T, float (*f) (float), struct tensor *_D)
{
    struct tensor D = _D ? *_D : tens_zeros(T->shape);
    assert (tens_match(T->shape, D.shape));

    struct tens_index index = tens_index(T->shape.shape, T->shape.order);
    do {
        *tens_getp(D, index.index) = f(tens_get(*T, index.index));
    } while (tens_index_next(&index));

    return D;
}

struct tensor tens_scalarmul(struct tensor *T, float l, struct tensor *_D)
{
    struct tensor D = _D ? *_D : tens_zeros(T->shape);
    assert (tens_match(T->shape, D.shape));

    struct tens_index index = tens_index(T->shape.shape, T->shape.order);
    do {
        *tens_getp(D, index.index) = tens_get(*T, index.index) * l;
    } while (tens_index_next(&index));

    return D;
}

struct tensor tens_add(
        struct tensor *S, struct tensor *T, struct tensor *_D)
{
    struct tens_pair ST = tens_broadcast(S, T);
    struct tensor D = _D ? *_D : tens_zeros(ST.S.shape);
    assert (tens_match(ST.S.shape, D.shape));

    struct tens_index index = tens_index(ST.S.shape.shape, ST.S.shape.order);
    do {
        *tens_getp(D, index.index) = 
            tens_get(ST.S, index.index) + tens_get(ST.T, index.index);
    } while (tens_index_next(&index));

    return D;
}

struct tensor tens_matmul(
        struct tensor *S, struct tensor *T, struct tensor *_D)
{
    assert (S->shape.order >= 1);
    assert (T->shape.order >= 1);
    assert (S->shape.shape[S->shape.order - 1] 
            == T->shape.shape[T->shape.order - 2]);

    struct tens_pair ST = tens_broadcastskipaxes(S, T, 2);
    int8_t order = ST.S.shape.order;

    struct tens_shape D_shape = (struct tens_shape) {.order = order };
    for (int8_t i = 0; i < order - 2; i++)
        D_shape.shape[i] = ST.S.shape.shape[i];
    D_shape.shape[order - 2] = ST.S.shape.shape[order - 2];
    D_shape.shape[order - 1] = ST.S.shape.shape[order - 1];
    struct tensor D = _D ? *_D : tens_zeros(D_shape);
    assert (tens_match(D.shape, D_shape));

    struct tens_index index = tens_index(ST.S.shape.shape, order);
    do {
        for (size_t i = 0; i < D.shape.shape[order - 2]; i++) {
            for (size_t j = 0; j < D.shape.shape[order - 1]; j++) {
                for (size_t k = 0; k < ST.S.shape.shape[order - 1]; k++) {
                    index.index[order - 2] = i; index.index[order - 1] = k;
                    float s = tens_get(ST.S, index.index);
                    index.index[order - 2] = k; index.index[order - 1] = j;
                    float t = tens_get(ST.T, index.index);
                    index.index[order - 2] = i; index.index[order - 1] = j;
                    *tens_getp(D, index.index) += s * t;
                }
            }
        }
    } while (tens_index_nextaxis(&index, order - 2));

    return D;
}
