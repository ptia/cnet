#include "tensor.h"
#include "tens_index.h"
#include <assert.h>

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
