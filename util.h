#define getparent(member_p, parent_t, field) \
            ((parent_t *) ( \
                ((char *) member_p) - offsetof(parent_t, field) \
            ))
