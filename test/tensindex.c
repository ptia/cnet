#include "minunit.h"
#include "../tensindex.h"

MU_TEST(test_tensindex_next)
{
    struct tensindex index = (struct tensindex) {
        .index = {0, 0, 0}, 
        .shape = {2, 1, 2}, 
        .order = 3
    };

    mu_check(tensindex_next(&index));
    mu_check(tensindex_next(&index));
    mu_check(tensindex_next(&index));
    mu_check(!tensindex_next(&index));
}

MU_TEST_SUITE(test_tensindex)
{
    MU_RUN_TEST(test_tensindex_next);
}
