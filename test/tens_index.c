#include "minunit.h"
#include "../tens_index.h"

MU_TEST(test_tens_index_next)
{
    struct tens_index index = (struct tens_index) {
        .index = {0, 0, 0}, 
        .shape = {3, {2, 1, 2}}, 
    };

    mu_check(tens_index_next(&index));
    mu_check(tens_index_next(&index));
    mu_check(tens_index_next(&index));
    mu_check(!tens_index_next(&index));
}

MU_TEST_SUITE(test_tens_index)
{
    MU_RUN_TEST(test_tens_index_next);
}
