#include "tensor.h"
#include "tens_iterator.h"
#include <assert.h>

void tens_scalar_mul(struct tensor T, float l)
{
    struct tens_iterator iter = tens_iterator(T);
    for (float *t; (t = tens_iter_next(&iter)); )
        *t *= l;
}

void tens_add(struct tensor S, struct tensor T, struct tensor D)
{
    struct tens_pair ST = tens_broadcast(S, T);
    S = ST.S;
    T = ST.T;
    assert (tens_match(S, D));

    //TODO optimise this. These are all the same indices!
    struct tens_iterator iterS = tens_iterator(S);
    struct tens_iterator iterT = tens_iterator(T);
    struct tens_iterator iterD = tens_iterator(D);
    for (float *s, *t, *d;
            (s = tens_iter_next(&iterS),
            t = tens_iter_next(&iterT),
            d = tens_iter_next(&iterD)); )
        *d = *s + *t;
}

void tens_mat_mul(struct tensor S, struct tensor T, struct tensor D)
{
    assert (S.shape[S.order - 1] == T.shape[T.order - 2]);

    struct tens_pair ST = tens_broadcast_skip_axes(S, T, 2);
    S = ST.S;
    T = ST.T;

    size_t D_shape[TENS_MAX_ORDER];
    for (int8_t i = 0; i < S.order - 2; i++)
        D_shape[i] = S.shape[i];
    D_shape[S.order - 2] = S.shape[S.order - 2];
    D_shape[S.order - 1] = S.shape[S.order - 1];

    struct tens_iterator iterS = tens_iterator_skip_axes(S, 2);
    struct tens_iterator iterT = tens_iterator_skip_axes(T, 2);
    struct tens_iterator iterD = tens_iterator_skip_axes(D, 2);
    for (float *s, *t, *d;
            (s = tens_iter_next(&iterS),
            t = tens_iter_next(&iterT),
            d = tens_iter_next(&iterD)); )
        ;
}
