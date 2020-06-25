#include "tensor.h"
#include "tens_index.h"
#include <assert.h>

void tens_scalarmul(struct tensor T, float l)
{
    struct tens_index index = tens_index(T.shape.shape, T.shape.order);
    do {
        *tens_getp(T, index.index) *= l;
    } while (tens_index_next(&index));
}

void tens_add(struct tensor S, struct tensor T, struct tensor D)
{
    struct tens_pair ST = tens_broadcast(S, T);
    S = ST.S;
    T = ST.T;
    assert (tens_match(S.shape, D.shape));

    struct tens_index index = tens_index(S.shape.shape, S.shape.order);
    do {
        *tens_getp(D, index.index) = 
            tens_get(S, index.index) + tens_get(T, index.index);
    } while (tens_index_next(&index));
}

void tens_matmul(struct tensor S, struct tensor T, struct tensor D)
{
    assert (S.shape.order >= 1);
    assert (T.shape.order >= 1);
    assert (S.shape.shape[S.shape.order - 1] 
            == T.shape.shape[T.shape.order - 2]);

    struct tens_pair ST = tens_broadcastskipaxes(S, T, 2);
    S = ST.S; T = ST.T;
    int8_t order = S.shape.order;

    struct tens_shape D_shape = (struct tens_shape) {.order = order };
    for (int8_t i = 0; i < order - 2; i++)
        D_shape.shape[i] = S.shape.shape[i];
    D_shape.shape[order - 2] = S.shape.shape[order - 2];
    D_shape.shape[order - 1] = S.shape.shape[order - 1];
    assert (tens_match(D.shape, D_shape));

    struct tens_index index = tens_index(S.shape.shape, order);
    do {
        for (size_t i = 0; i < D.shape.shape[order - 2]; i++) {
            for (size_t j = 0; j < D.shape.shape[order - 1]; j++) {
                for (size_t k = 0; k < S.shape.shape[order - 1]; k++) {
                    index.index[order - 2] = i; index.index[order - 1] = k;
                    float s = tens_get(S, index.index);
                    index.index[order - 2] = k; index.index[order - 1] = j;
                    float t = tens_get(T, index.index);
                    index.index[order - 2] = i; index.index[order - 1] = j;
                    *tens_getp(D, index.index) += s * t;
                }
            }
        }
    } while (tens_index_nextaxis(&index, order - 2));
}
