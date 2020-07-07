#include "random.h"
#include <stdlib.h>

float randf()
{
    return ((float) rand()) / ((float) RAND_MAX);
}

float randnf()
{
    float sum = 0; 
    for (int i = 0; i < RANDNF_MAX * 2; i++)
        sum += randf();
    return sum - RANDNF_MAX;
}
