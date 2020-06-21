#include "tens.h"
#include "tens_iterator.h"
#include <assert.h>

void tens_scalar_mul(struct tens T, float l)
{
    struct tens_iterator iter = tens_iterator(T);
    for (float *t; (t = tens_iter_next(&iter)); )
        *t *= l;
}

void tens_add(struct tens S, struct tens T)
{
    assert (tens_match(S, T));
    struct tens_iterator iterS = tens_iterator(S);
    struct tens_iterator iterT = tens_iterator(T);
    for (float *s; (s = tens_iter_next(&iterS)); )
        *tens_iter_next(&iterT) += *s;
}
