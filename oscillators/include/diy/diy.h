#ifndef DIY_DIY_H
#define DIY_DIY_H

#include "constants.h"
#include "types.h"

void DIY_Foreach(void (f*)(void* block, void* comm));

void DIY_Enqueue_item(void* comm, int nhbr);
void DIY_Enqueue_item_dir(void* comm, dir_t dir);
//void DIY_Enqueue_item_points(void* comm, ...);
void DIY_Enqueue_item_all(void* comm);

#endif
