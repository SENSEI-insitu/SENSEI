/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPriorityQueue.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPriorityQueue
 * @brief   a list of ids arranged in priority order
 *
 * svtkPriorityQueue is a general object for creating and manipulating lists
 * of object ids (e.g., point or cell ids). Object ids are sorted according
 * to a user-specified priority, where entries at the top of the queue have
 * the smallest values.
 *
 * This implementation provides a feature beyond the usual ability to insert
 * and retrieve (or pop) values from the queue. It is also possible to
 * pop any item in the queue given its id number. This allows you to delete
 * entries in the queue which can useful for reinserting an item into the
 * queue.
 *
 * @warning
 * This implementation is a variation of the priority queue described in
 * "Data Structures & Algorithms" by Aho, Hopcroft, Ullman. It creates
 * a balanced, partially ordered binary tree implemented as an ordered
 * array. This avoids the overhead associated with parent/child pointers,
 * and frequent memory allocation and deallocation.
 */

#ifndef svtkPriorityQueue_h
#define svtkPriorityQueue_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

#include "svtkIdTypeArray.h" // Needed for inline methods

class SVTKCOMMONCORE_EXPORT svtkPriorityQueue : public svtkObject
{
public:
  class Item
  {
  public:
    double priority;
    svtkIdType id;
  };

  /**
   * Instantiate priority queue with default size and extension size of 1000.
   */
  static svtkPriorityQueue* New();

  svtkTypeMacro(svtkPriorityQueue, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Allocate initial space for priority queue.
   */
  void Allocate(svtkIdType sz, svtkIdType ext = 1000);

  /**
   * Insert id with priority specified. The id is generally an
   * index like a point id or cell id.
   */
  void Insert(double priority, svtkIdType id);

  /**
   * Removes item at specified location from tree; then reorders and
   * balances tree. The location == 0 is the root of the tree. If queue
   * is exhausted, then a value < 0 is returned. (Note: the location
   * is not the same as deleting an id; id is mapped to location.)
   */
  svtkIdType Pop(svtkIdType location, double& priority);

  /**
   * Same as above but simplified for easier wrapping into interpreted
   * languages.
   */
  svtkIdType Pop(svtkIdType location = 0);

  /**
   * Peek into the queue without actually removing anything. Returns the
   * id and the priority.
   */
  svtkIdType Peek(svtkIdType location, double& priority);

  /**
   * Peek into the queue without actually removing anything. Returns the
   * id.
   */
  svtkIdType Peek(svtkIdType location = 0);

  /**
   * Delete entry in queue with specified id. Returns priority value
   * associated with that id; or SVTK_DOUBLE_MAX if not in queue.
   */
  double DeleteId(svtkIdType id);

  /**
   * Get the priority of an entry in the queue with specified id. Returns
   * priority value of that id or SVTK_DOUBLE_MAX if not in queue.
   */
  double GetPriority(svtkIdType id);

  /**
   * Return the number of items in this queue.
   */
  svtkIdType GetNumberOfItems() { return this->MaxId + 1; }

  /**
   * Empty the queue but without releasing memory. This avoids the
   * overhead of memory allocation/deletion.
   */
  void Reset();

protected:
  svtkPriorityQueue();
  ~svtkPriorityQueue() override;

  Item* Resize(const svtkIdType sz);

  svtkIdTypeArray* ItemLocation;
  Item* Array;
  svtkIdType Size;
  svtkIdType MaxId;
  svtkIdType Extend;

private:
  svtkPriorityQueue(const svtkPriorityQueue&) = delete;
  void operator=(const svtkPriorityQueue&) = delete;
};

inline double svtkPriorityQueue::DeleteId(svtkIdType id)
{
  double priority = SVTK_DOUBLE_MAX;
  svtkIdType loc;

  if (id <= this->ItemLocation->GetMaxId() && (loc = this->ItemLocation->GetValue(id)) != -1)
  {
    this->Pop(loc, priority);
  }
  return priority;
}

inline double svtkPriorityQueue::GetPriority(svtkIdType id)
{
  svtkIdType loc;

  if (id <= this->ItemLocation->GetMaxId() && (loc = this->ItemLocation->GetValue(id)) != -1)
  {
    return this->Array[loc].priority;
  }
  return SVTK_DOUBLE_MAX;
}

inline svtkIdType svtkPriorityQueue::Peek(svtkIdType location, double& priority)
{
  if (this->MaxId < 0)
  {
    return -1;
  }
  else
  {
    priority = this->Array[location].priority;
    return this->Array[location].id;
  }
}

inline svtkIdType svtkPriorityQueue::Peek(svtkIdType location)
{
  if (this->MaxId < 0)
  {
    return -1;
  }
  else
  {
    return this->Array[location].id;
  }
}

#endif
