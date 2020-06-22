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
    struct tens_pair ST = tens_broadcast(S, T, 0);
    S = ST.S;
    T = ST.T;
    assert (tens_match(S, D));

    //TODO optimise this. These are all the same indices!
    struct tens_iterator iterS = tens_iterator(S);
    struct tens_iterator iterT = tens_iterator(T);
    struct tens_iterator iterD = tens_iterator(T);
    for (float *s, *t, *d;
            (s = tens_iter_next(&iterS),
            t = tens_iter_next(&iterT),
            d = tens_iter_next(&iterD)); )
        *d = *s + *t;
}
