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
} image_t;

struct patch_t;

struct patch_t
{
    int            id;
    int            level;
    int            logical_extents[4];
    float          window[4];

    unsigned char *data;
    unsigned char *blank;
    int            nx;
    int            ny;

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
void      patch_alloc_data(patch_t *patch, int nx, int ny);
void      patch_alloc_blank(patch_t *patch, int nx, int ny);
patch_t  *patch_add_subpatches(patch_t *patch, int n);
void      patch_add_owner(patch_t *patch, int owner);
int       patch_num_patches(patch_t *patch);
patch_t  *patch_get_patch(patch_t *patch, int id);
int       patch_num_levels(patch_t *patch);
patch_t **patch_flat_array(patch_t *patch, int *np);
void      patch_free_flat_array(patch_t **pfa);
void      patch_refine(patch_t *patch, int refinement_ratio, 
                       void (*maskcb)(patch_t *, image_t*));
long long patch_num_points(patch_t *patch);
long long patch_num_cells(patch_t *patch);
int       patch_find_patch(patch_t **plist, int np, int id, patch_t *&p);

#endif
