#ifndef AMR_PATCH_H
#define AMR_PATCH_H
#include <stdio.h>
#include <stdlib.h>

#define ALLOC(N,T) (T *)calloc(N, sizeof(T))
#define REALLOC(P,N,T) (T *)realloc(P, (N) * sizeof(T))
#define FREE(P) if(P != NULL) {free(P); P = NULL; }

typedef struct
{
    unsigned char *data;
    int nx;
    int ny;
    int nz;
} image_t;

// The data type to use for fields.
typedef float patch_data_t;

// Forward declare
struct patch_t;

struct patch_t
{
    int            id;
    int            level;
    int            logical_extents[6];
    float          window[6];

    /* Begin data -- patch_alloc_data, patch_free_data used for these. */
    patch_data_t  *data;
    /* End data */

    unsigned char *blank;
    int            nx;
    int            ny;
    int            nz;

    int            nowners;
    int           *owners;
    int            single_owner;

    patch_t       *subpatches;
    int            nsubpatches;
};

void      patch_ctor(patch_t *patch);
void      patch_dtor(patch_t *patch);
void      patch_shallow_copy(patch_t *dest, patch_t *src);
void      patch_print(FILE *f, patch_t *patch);
void      patch_alloc_blank(patch_t *patch, int nx, int ny, int nz);
patch_t  *patch_add_subpatches(patch_t *patch, int n);
void      patch_add_owner(patch_t *patch, int owner);
int       patch_num_patches(patch_t *patch);
patch_t  *patch_get_patch(patch_t *patch, int id);
int       patch_num_levels(patch_t *patch);
patch_t **patch_flat_array(patch_t *patch, int *np);
void      patch_refine(patch_t *patch, int refinement_ratio, 
                       void (*maskcb)(patch_t *, image_t*, void*),
                       void *maskcbdata);

// Patch data-specific
void      patch_alloc_data(patch_t *patch, int nx, int ny, int nz);
void      patch_free_data(patch_t *patch);

#endif
