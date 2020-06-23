#include "tensor.h"
#include "tensindex.h"
#include <assert.h>

void tens_scalar_mul(struct tensor T, float l)
{
    struct tensindex index = tensindex(T.shape, T.order);
    do {
        *tens_getp(T, index.index) *= l;
    } while (tensindex_next(&index));
}

void tens_add(struct tensor S, struct tensor T, struct tensor D)
{
    struct tens_pair ST = tens_broadcast(S, T);
    S = ST.S;
    T = ST.T;
    assert (tens_match(S, D));

    struct tensindex index = tensindex(S.shape, S.order);
    do {
        *tens_getp(D, index.index) = 
            tens_get(S, index.index) + tens_get(T, index.index);
    } while (tensindex_next(&index));
}

void tens_mat_mul(struct tensor S, struct tensor T, struct tensor D)
{
    assert (S.order >= 1);
    assert (T.order >= 1);
    assert (S.shape[S.order - 1] == T.shape[T.order - 2]);

    struct tens_pair ST = tens_broadcast_skip_axes(S, T, 2);
    S = ST.S; T = ST.T;
    int8_t order = S.order;

    size_t D_shape[TENS_MAX_ORDER];
    for (int8_t i = 0; i < order - 2; i++)
        D_shape[i] = S.shape[i];
    D_shape[order - 2] = S.shape[order - 2];
    D_shape[order - 1] = S.shape[order - 1];
    tens_match_shape(D, order, D_shape);

    struct tensindex index = tensindex(S.shape, order);
    do {
        for (int i = 0; i < D.shape[order - 2]; i++) {
            for (int j = 0; j < D.shape[order - 1]; j++) {
                for (int k = 0; k < S.shape[order - 1]; k++) {
                    index.index[order - 2] = i; index.index[order - 1] = k;
                    float s = tens_get(S, index.index);
                    index.index[order - 2] = k; index.index[order - 1] = j;
                    float t = tens_get(T, index.index);
                    index.index[order - 2] = i; index.index[order - 1] = j;
                    *tens_getp(D, index.index) += s * t;
                }
            }
        }
    } while (tensindex_nextaxis(&index, order - 2));
}
