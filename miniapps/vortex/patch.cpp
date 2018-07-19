#include <map>
#include "patch.h"
#include <cstring>
#include <iostream>
// ****************************************************************************
// Functions: Functions related to the patch object
//
// Programmer: Brad Whitlock
// Creation:   Wed Mar 25 14:05:02 PDT 2009
//
// Modifications:
//   
// ****************************************************************************

void
patch_print(FILE *f, patch_t *patch)
{
    fprintf(f, "patch {\n");
    fprintf(f, "    id=%d\n", patch->id);
    fprintf(f, "    level=%d\n", patch->level);
    fprintf(f, "    logical_extents={x=%d %d  y=%d %d  z=%d %d}\n",
        patch->logical_extents[0],
        patch->logical_extents[1],
        patch->logical_extents[2],
        patch->logical_extents[3],
        patch->logical_extents[4],
        patch->logical_extents[5]);
    fprintf(f, "    window={x=%g %g  y=%g %g  z=%g %g}\n",
        patch->window[0],
        patch->window[1],
        patch->window[2],
        patch->window[3],
        patch->window[4],
        patch->window[5]);
    fprintf(f, "    nx=%d\n", patch->nx);
    fprintf(f, "    ny=%d\n", patch->ny);
    fprintf(f, "    nz=%d\n", patch->nz);
    fprintf(f, "    nowners=%d\n", patch->nowners);
    fprintf(f, "    owners={");
    for(int i = 0; i < patch->nowners; ++i)
        fprintf(f, "%d ", patch->owners[i]);
    fprintf(f, "}\n");
    fprintf(f, "    nsubpatches=%d\n", patch->nsubpatches);
    fprintf(f, "    subpatches={");
    for(int i = 0; i < patch->nsubpatches; ++i)
        fprintf(f, "%d ", patch->subpatches[i].id);
    fprintf(f, "}\n");
    fprintf(f, "}\n");
}

void
patch_ctor(patch_t *patch)
{
    memset(patch, 0, sizeof(patch_t));
    patch->single_owner = -1;
}

void
patch_dtor(patch_t *patch)
{
    if(patch != NULL)
    {
        patch_free_data(patch);
        FREE(patch->blank);
        if(patch->nowners > 1)
        {
            FREE(patch->owners);
        }
        for(int i = 0; i < patch->nsubpatches; ++i)
            patch_dtor(&patch->subpatches[i]);
        FREE(patch->subpatches);
    }
}

void
patch_shallow_copy(patch_t *dest, patch_t *src)
{
    memcpy(dest, src, sizeof(patch_t));
    // update the owner pointer.
    if(dest->nowners == 1)
        dest->owners = &dest->single_owner;
}

void
patch_alloc_blank(patch_t *patch, int nx, int ny, int nz)
{
    patch->nx = nx;
    patch->ny = ny;
    patch->nz = nz;
    patch->blank = ALLOC(nx*ny*nz, unsigned char);
    memset(patch->blank, 0, nx*ny*nz*sizeof(unsigned char));
}

patch_t *
patch_add_subpatches(patch_t *patch, int n)
{
    if(n == 0)
        return NULL;
    if(patch->nsubpatches == 0)
    {
        patch->subpatches = ALLOC(n, patch_t);
        patch->nsubpatches = n;
    }
    else
    {
        patch->nsubpatches += n;
        patch->subpatches = REALLOC(patch->subpatches, patch->nsubpatches, patch_t);
    }
    for(int i = 0; i < n; ++i)
        patch_ctor(&patch->subpatches[patch->nsubpatches-n+i]);
    return &patch->subpatches[patch->nsubpatches-n];
}

void
patch_add_owner(patch_t *patch, int owner)
{
    if(patch->nowners == 0)
    {
        // Let's us avoid extra allocations for most leaf patches.
        patch->single_owner = owner;
        patch->owners = &patch->single_owner;
        patch->nowners = 1;
    }
    else
    {
        if(patch->nowners == 1)
        {
            patch->nowners++;
            patch->owners = ALLOC(patch->nowners, int);
            patch->owners[0] = patch->single_owner;
            patch->owners[1] = owner;
            patch->single_owner = -1;
        }
        else
        {
            patch->nowners++;
            patch->owners = REALLOC(patch->owners, patch->nowners, int);
            patch->owners[patch->nowners-1] = owner;
        }
    }
}

static void
patch_num_patches_helper(patch_t *patch, int *sum)
{
    *sum = *sum + 1;
    for(int i = 0; i < patch->nsubpatches; ++i)
        patch_num_patches_helper(&patch->subpatches[i], sum);
}

int
patch_num_patches(patch_t *patch)
{
    int sum = 0;
    patch_num_patches_helper(patch, &sum);
    return sum;
}

patch_t *
patch_get_patch(patch_t *patch, int id)
{
    if(patch->id == id)
        return patch;

    for(int i = 0; i < patch->nsubpatches; ++i)
    {
        patch_t *p = patch_get_patch(&patch->subpatches[i], id);
        if(p != NULL)
            return p;
    }

    return NULL;
}

static void
patch_num_levels_helper(patch_t *patch, int *maxlevel)
{
    if(patch->level > *maxlevel)
        *maxlevel = patch->level;
    for(int i = 0; i < patch->nsubpatches; ++i)
        patch_num_levels_helper(&patch->subpatches[i], maxlevel);
}

int
patch_num_levels(patch_t *patch)
{
    int maxlevel = 0;
    patch_num_levels_helper(patch, &maxlevel);
    return maxlevel+1;
}

static void
patch_flat_array_helper(patch_t *patch, patch_t **arr, int *index)
{
    arr[*index] = patch;
    *index = *index + 1;
    for(int i = 0; i < patch->nsubpatches; ++i)
        patch_flat_array_helper(&patch->subpatches[i], arr, index);
}

patch_t **
patch_flat_array(patch_t *patch, int *np)
{
    *np = patch_num_patches(patch);
    patch_t **patches = ALLOC(*np, patch_t*);
    int index = 0;
    patch_flat_array_helper(patch, patches, &index);
    return patches;
}

/******************************************************************************/

typedef struct
{
    int *data;
    int nx;
    int ny;
    int nz;
} score_t;

static void
score_fill_box(score_t *score, int indices[6], int value)
{
    for(int k = indices[4]; k < indices[5]; ++k)
    {
        int koffset = score->nx*score->ny * k;
        for(int j = indices[2]; j < indices[3]; ++j)
        {
            int offset = koffset + score->nx * j;
            for(int i = indices[0]; i < indices[1]; ++i)
            {
                int thispix = offset + i;
                score->data[thispix] = value;
            }
        }
    }
}

static void
score_get_partition(int divide, int indices[6], int partition0[6], int partition1[6])
{
    int dx = indices[1] - indices[0];
    int dy = indices[3] - indices[2];
    int dz = indices[5] - indices[4];

    if(divide == 0)
    {
        // Divide X
        partition0[0] = indices[0];
        partition0[1] = indices[0] + dx/2;
        partition0[2] = indices[2];
        partition0[3] = indices[3];
        partition0[4] = indices[4];
        partition0[5] = indices[5];

        partition1[0] = indices[0] + dx/2;
        partition1[1] = indices[1];
        partition1[2] = indices[2];
        partition1[3] = indices[3];
        partition1[4] = indices[4];
        partition1[5] = indices[5];
    }
    else if(divide == 1)
    {
        // Divide Y
        partition0[0] = indices[0];
        partition0[1] = indices[1];
        partition0[2] = indices[2];
        partition0[3] = indices[2] + dy/2;
        partition0[4] = indices[4];
        partition0[5] = indices[5];

        partition1[0] = indices[0];
        partition1[1] = indices[1];
        partition1[2] = indices[2] + dy/2;
        partition1[3] = indices[3];
        partition1[4] = indices[4];
        partition1[5] = indices[5];
    }
    else if(divide == 2)
    {
        // Divide Z
        partition0[0] = indices[0];
        partition0[1] = indices[1];
        partition0[2] = indices[2];
        partition0[3] = indices[3];
        partition0[4] = indices[4];
        partition0[5] = indices[4] + dz/2;

        partition1[0] = indices[0];
        partition1[1] = indices[1];
        partition1[2] = indices[2];
        partition1[3] = indices[3];
        partition1[4] = indices[4] + dz/2;
        partition1[5] = indices[5];
    }
}

// ****************************************************************************
// Function: score_image_helper
//
// Purpose: 
//   Helper function that recurses on a region of the mask image and determines
//   whether a new patch can be created for the region.
//
// Arguments:
//
// Returns:    
//
// Note:       
//
// Programmer: Brad Whitlock
// Creation:   Wed Mar 25 14:00:09 PDT 2009
//
// Modifications:
//   Brad Whitlock, Wed Jul 18 15:11:15 PDT 2018
//   Made it 3D.
//
// ****************************************************************************

#define SCORE_PIXELS_DIFFERENT 0
#define SCORE_PIXELS_ALL_SAME  1

static int
score_image_helper(image_t *mask, score_t *score, int level, int indices[4],
    int *boxid)
{
    int retval = 0;
    int dx = indices[1] - indices[0];
    int dy = indices[3] - indices[2];
    int dz = indices[5] - indices[4];

    const int min_patch_size = 8;
    const int patch_divisor = 16;
    int minx = ((mask->nx / patch_divisor) < min_patch_size) ? min_patch_size : (mask->nx / patch_divisor);
    int miny = ((mask->ny / patch_divisor) < min_patch_size) ? min_patch_size : (mask->ny / patch_divisor);
    int minz = ((mask->nz / patch_divisor) < min_patch_size) ? min_patch_size : (mask->nz / patch_divisor);

    bool verticalDivideOkay = (dx > minx);
    bool horizontalDivideOkay = (dy > miny);
    bool depthDivideOkay = (dz > minz);

    if(!verticalDivideOkay && !horizontalDivideOkay && !depthDivideOkay)
    {
        /* Check all of the pixels in the indices and see if they are
           the same.
         */
        int mnxy = mask->ny*mask->nx;
        int pix0 = (mnxy * indices[4]) + (mask->nx * indices[2]) + indices[0];
        for(int k = indices[4]; k < indices[5]; ++k)
        {
            int koffset = mnxy * k;
            for(int j = indices[2]; j < indices[3]; ++j)
            {
                int offset = koffset + mask->nx * j;
                for(int i = indices[0]; i < indices[1]; ++i)
                {
                    int thispix = offset + i;
                    if(mask->data[pix0] != mask->data[thispix])
                    {
                        score_fill_box(score, indices, *boxid);
                        *boxid = *boxid + 1;
                        return SCORE_PIXELS_DIFFERENT;
                    }
                }
            }
        }
        retval = SCORE_PIXELS_ALL_SAME;
    }
    else
    {
        int partition0[6], partition1[6];
#if 1
        int divideDim = -1;
        int divideSize = -1;
        // Prefer the longest dim to divide.
        if(verticalDivideOkay)
        {
            divideDim = 0;
            divideSize = dx;
        }
        if(horizontalDivideOkay && dy > divideSize)
        {
            divideDim = 1;
            divideSize = dy;
        }
        if(depthDivideOkay && dz > divideSize)
        {
            divideDim = 2;
            divideSize = dz;
        }
#else
        bool divideVertical   = verticalDivideOkay && ((level & 1) == 0); // this makes it alternate divisions.
        bool divideHorizontal = horizonalDivideOkay && ((level & 1) == 0);
#endif
        // NOTE: It's not alternating division dimensions now.

        score_get_partition(divideDim, indices, partition0, partition1);

        int s0 = score_image_helper(mask, score, level+1, partition0, boxid);
        int s1 = score_image_helper(mask, score, level+1, partition1, boxid);

        if(s0 == SCORE_PIXELS_DIFFERENT && s1 == SCORE_PIXELS_DIFFERENT)
        {
            // Fill in the score with the same id.
            score_fill_box(score, indices, *boxid);
            *boxid = *boxid + 1;
            retval = SCORE_PIXELS_DIFFERENT;
        }
        else
            retval = SCORE_PIXELS_ALL_SAME;
    }
    return retval;
}

// ****************************************************************************
// Function: score_image
//
// Purpose: 
//   Creates a score image from a mask image. The score image is an image 
//   that contains a bunch of "colored" rectangles that correspond to patches
//   that we'll create.
//
// Arguments:
//
// Returns:    
//
// Note:       
//
// Programmer: Brad Whitlock
// Creation:   Wed Mar 25 14:00:09 PDT 2009
//
// Modifications:
//   
// ****************************************************************************

static void
score_image(image_t *mask, score_t *score)
{
    int indices[6];
    int boxid = 0;

    indices[0] = 0;
    indices[1] = mask->nx;
    indices[2] = 0;
    indices[3] = mask->ny;
    indices[4] = 0;
    indices[5] = mask->nz;
    score_image_helper(mask, score, 0, indices, &boxid);
}

struct score_patch_info
{
    int startx, endx;
    int starty, endy;
    int startz, endz;
};

// ****************************************************************************
// Function: patches_abut
//
// Purpose: 
//   Returns true if the 2 patches abut in a way that would allow them to
//   be combined.
//
// Arguments:
//
// Returns:    
//
// Note:       
//
// Programmer: Brad Whitlock
// Creation:   Fri Mar 27 08:49:10 PDT 2009
//
// Modifications:
//   Brad Whitlock, Wed Jul 18 15:08:21 PDT 2018
//   Made it 3D.
//
// ****************************************************************************

static bool
patches_abut(const score_patch_info &p0, const score_patch_info &p1)
{
    if(p0.startx == p1.startx && p0.endx == p1.endx)
    {
        if(p0.endz+1 == p1.startz || 
           p0.startz-1 == p1.endz)
        {
            // front-back alignment

            if(p0.endy+1 == p1.starty || 
               p0.starty-1 == p1.endy)
            {
                // top-bottom alignment
                return true;
            }
        }
    }
    else if(p0.starty == p1.starty && p0.endy == p1.endy)
    {
        if(p0.endz+1 == p1.startz || 
           p0.startz-1 == p1.endz)
        {
            // front-back alignment

            if(p0.endx+1 == p1.startx ||
               p0.startx-1 == p1.endx)
            {
                // left-right alignment
                return true;
            }
        }
    }
    else if(p0.startz == p1.startz && p0.endz == p1.endz)
    {
        if(p0.endy+1 == p1.starty || 
           p0.starty-1 == p1.endy)
        {
            // top-bottom alignment

            if(p0.endx+1 == p1.startx ||
               p0.startx-1 == p1.endx)
            {
                // left-right alignment
                return true;
            }
        }
    }
    return false;
}

// ****************************************************************************
// Function: patch_add
//
// Purpose: 
//   Adds 2 patches together.
//
// Arguments:
//
// Returns:    
//
// Note:       
//
// Programmer: Brad Whitlock
// Creation:   Fri Mar 27 08:49:50 PDT 2009
//
// Modifications:
//   Brad Whitlock, Wed Jul 18 12:45:58 PDT 2018
//   Made it 3D.
//   
// ****************************************************************************
#define MIN(A,B) (((A)<(B)) ? (A) : (B))
#define MAX(A,B) (((A)>(B)) ? (A) : (B))

static score_patch_info
patch_add(const score_patch_info &p0, const score_patch_info &p1)
{
    score_patch_info s;

    s.startx = MIN(p0.startx, p1.startx);
    s.endx = MAX(p0.endx, p1.endx);
    s.starty = MIN(p0.starty, p1.starty);
    s.endy = MAX(p0.endy, p1.endy);
    s.startz = MIN(p0.startz, p1.startz);
    s.endz = MAX(p0.endz, p1.endz);

    return s;
}

// ****************************************************************************
// Method: combine_patches
//
// Purpose: 
//   Iterates through the patch map and combines compatible patches to 
//   reduce the patch count.
//
// Arguments:
//
// Returns:    
//
// Note:       
//
// Programmer: Brad Whitlock
// Creation:   Fri Mar 27 08:50:24 PDT 2009
//
// Modifications:
//   Brad Whitlock, Wed Jul 18 12:45:58 PDT 2018
//   Made it 3D.
//   
// ****************************************************************************

void
combine_patches(std::map<int, score_patch_info> &patchmap)
{
    std::map<int, score_patch_info>::iterator p0, p1;
    bool found_matches;

    do
    {
        found_matches = false;
        for(p0 = patchmap.begin(); p0 != patchmap.end(); ++p0)
        {
            std::map<int, score_patch_info> matches;
            for(p1 = patchmap.begin(); p1 != patchmap.end(); ++p1)
            {
                if(p0->first == p1->first)
                    continue;

                if(patches_abut(p0->second, p1->second))
                    matches[p1->first] = p1->second;
            }

            // Now that we have a list of matches for p0, let's choose the
            // one that gives us the most volume.
            int max_vol = -1;
            int best_match = -1;
            for(p1 = matches.begin(); p1 != matches.end(); ++p1)
            {
                int nx = p1->second.endx - p1->second.startx + 1;
                int ny = p1->second.endy - p1->second.starty + 1;
                int nz = p1->second.endz - p1->second.startz + 1;
                if(nx*ny*nz > max_vol)
                {
                    max_vol = nx*ny*nz;
                    best_match = p1->first;
                }
            }

            p1 = patchmap.find(best_match);
            if(p1 != patchmap.end())
            {
                // Let's add best_match to p0 and remove p1 from the map.
                p0->second = patch_add(p0->second, p1->second);
                patchmap.erase(p1);
                found_matches = true;
                break;
            }
        }
    } while(found_matches);
}

// ****************************************************************************
// Function: patch_refine
//
// Purpose: 
//   This method examines the patch's data using the provided maskcb function
//   and refines according to the mask that the callback created. New patches
//   get created as children of the input patch and we can further calculate
//   on them.
//
// Arguments:
//
// Returns:    
//
// Note:       
//
// Programmer: Brad Whitlock
// Creation:   Wed Mar 25 14:01:33 PDT 2009
//
// Modifications:
//   Brad Whitlock, Wed Jul 18 12:45:58 PDT 2018
//   Made it 3D.
//
// ****************************************************************************

void
patch_refine(patch_t *patch, int refinement_ratio, 
    void (*maskcb)(patch_t *, image_t *, void *), void *maskcbdata)
{
    int i, j, k;

    /* Call the maskcb callback to flag the cells in the patch that need
     * to be refined.
     */
    image_t mask;
    mask.nx = patch->nx;
    mask.ny = patch->ny;
    mask.nz = patch->nz;
    mask.data = ALLOC(mask.nx*mask.ny*mask.nz, unsigned char);
    (*maskcb)(patch, &mask, maskcbdata);

    /* Use the mask image to create a set of patches that fit well and 
     * cover the values that are on.
     */
    score_t score;
    score.nx = patch->nx;
    score.ny = patch->ny;
    score.nz = patch->nz;
    score.data = ALLOC(score.nx*score.ny*score.nz, int);
    score_image(&mask, &score);

    /* Now that we have a score image, which breaks up the input patch
       into subpatches, let's scan through the image and get the extents
       for the patches so we can create subpatches.
     */
    std::map<int, score_patch_info> patchmap;
    int current_patch = -1;
    for(k = 0; k < patch->nz; ++k)
    {
        int koffset = (k*score.nx*score.ny);
        for(j = 0; j < patch->ny; ++j)
        {
            int offset = koffset + (j*score.nx);
            for(i = 0; i < patch->nx; ++i)
            {
                int si = offset + i;
                if(score.data[si] > 0)
                {
                    // see if we need to add a new patch to the map
                    current_patch = score.data[si];
                    std::map<int, score_patch_info>::iterator it = patchmap.find(current_patch);
                    if(it == patchmap.end())
                    {
                        // add a new patch
                        patchmap[current_patch].startx = i;
                        patchmap[current_patch].starty = j;
                        patchmap[current_patch].startz = k;

                        patchmap[current_patch].endx = i;
                        patchmap[current_patch].endy = j;
                        patchmap[current_patch].endz = k;
                    }
                }
            }
        }
    }

    // Iterate through the patch list and the score image to determine
    // each patch's end indices.
    for(std::map<int, score_patch_info>::iterator it = patchmap.begin();
        it != patchmap.end(); ++it)
    {
        for(k = it->second.startz+1; k < patch->nz; ++k)
        {
            int si = (k*score.nx*score.ny) + (it->second.starty*score.nx) + it->second.startx;
            if(score.data[si] == it->first)
                it->second.endz++;
            else 
                break;
        }
        for(j = it->second.starty+1; j < patch->ny; ++j)
        {
            int si = (it->second.startz*score.nx*score.ny) + (j*score.nx) + it->second.startx;
            if(score.data[si] == it->first)
                it->second.endy++;
            else 
                break;
        }
        for(i = it->second.startx+1; i < patch->nx; ++i)
        {
            int si = (it->second.startz*score.nx*score.ny) + (it->second.starty*score.nx) + i;
            if(score.data[si] == it->first)
                it->second.endx++;
            else 
                break;
        }
    }
    FREE(score.data);
    FREE(mask.data);

    // Combine compatible patches
    combine_patches(patchmap);    

    // Count how many patches we want to make and allocate them at once.
    std::map<int, score_patch_info>::iterator it;
    int npatches = 0;
    for(it = patchmap.begin(); it != patchmap.end(); ++it)
        npatches++;
    patch_t *newpatch = nullptr;

    if(npatches > 0)
    {
        newpatch = patch_add_subpatches(patch, npatches);

        // Allocate blank data for the patch we're refining so we can blank
        // out the cells covered by child patches.
        patch_alloc_blank(patch, patch->nx, patch->ny, patch->nz);
    }

    // This is a VisIt-ism used for indicating refinement.
    const int REFINED_ZONE_IN_AMR_GRID = 3;
    const unsigned char blank = 1 << REFINED_ZONE_IN_AMR_GRID;

    // Now that we have a list of patches, create subpatches based on them.
    for(it = patchmap.begin(); it != patchmap.end(); ++it)
    {
        // now that we know the patch's indices, let's create a subpatch
        int nx = it->second.endx - it->second.startx + 1;
        int ny = it->second.endy - it->second.starty + 1;
        int nz = it->second.endz - it->second.startz + 1;
        float cellWidth = (patch->window[1] - patch->window[0]) / ((float)patch->nx);
        float cellHeight = (patch->window[3] - patch->window[2]) / ((float)patch->ny);
        float cellDepth = (patch->window[5] - patch->window[4]) / ((float)patch->nz);
        newpatch->window[0] = patch->window[0] + it->second.startx * cellWidth;
        newpatch->window[1] = newpatch->window[0] + ((float)nx) * cellWidth;
        newpatch->window[2] = patch->window[2] + it->second.starty * cellHeight;
        newpatch->window[3] = newpatch->window[2] + ((float)ny) * cellHeight;
        newpatch->window[4] = patch->window[4] + it->second.startz * cellDepth;
        newpatch->window[5] = newpatch->window[4] + ((float)nz) * cellDepth;

        int startI = patch->logical_extents[0] + it->second.startx;
        int endI = patch->logical_extents[0] + it->second.endx;
        int startJ = patch->logical_extents[2] + it->second.starty;
        int endJ = patch->logical_extents[2] + it->second.endy;
        int startK = patch->logical_extents[4] + it->second.startz;
        int endK = patch->logical_extents[4] + it->second.endz;
        newpatch->logical_extents[0] = startI*refinement_ratio;
        newpatch->logical_extents[1] = (endI+1)*refinement_ratio-1;
        newpatch->logical_extents[2] = startJ*refinement_ratio;
        newpatch->logical_extents[3] = (endJ+1)*refinement_ratio-1;
        newpatch->logical_extents[4] = startK*refinement_ratio;
        newpatch->logical_extents[5] = (endK+1)*refinement_ratio-1;

        newpatch->nx = nx*refinement_ratio;
        newpatch->ny = ny*refinement_ratio;
        newpatch->nz = nz*refinement_ratio;

        // Blank the parent patch where this new patch sits.
        for(int k = it->second.startz; k <= it->second.endz; ++k)
        {
            int koffset = (k*patch->nx*patch->ny);
            for(int j = it->second.starty; j <= it->second.endy; ++j)
            {
                int offset = koffset + (j*patch->nx);
                for(int i = it->second.startx; i <= it->second.endx; ++i)
                    patch->blank[offset + i] = blank;
            }
        }

        // on to the next patch.
        newpatch++;
    }
}

// Edit these if we change the data payload on a patch.
void
patch_alloc_data(patch_t *patch, int nx, int ny, int nz)
{
    patch->nx = nx;
    patch->ny = ny;
    patch->nz = nz;
    patch->data = ALLOC(nx*ny*nz, float);
}

void
patch_free_data(patch_t *patch)
{
    FREE(patch->data);
}
