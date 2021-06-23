/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkIdList.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkIdList
 * @brief   list of point or cell ids
 *
 * svtkIdList is used to represent and pass data id's between
 * objects. svtkIdList may represent any type of integer id, but
 * usually represents point and cell ids.
 */

#ifndef svtkIdList_h
#define svtkIdList_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

class SVTKCOMMONCORE_EXPORT svtkIdList : public svtkObject
{
public:
  //@{
  /**
   * Standard methods for instantiation, type information, and printing.
   */
  static svtkIdList* New();
  svtkTypeMacro(svtkIdList, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  /**
   * Release memory and restore to unallocated state.
   */
  void Initialize();

  /**
   * Allocate a capacity for sz ids in the list and
   * set the number of stored ids in the list to 0.
   * strategy is not used.
   */
  int Allocate(const svtkIdType sz, const int strategy = 0);

  /**
   * Return the number of id's in the list.
   */
  svtkIdType GetNumberOfIds() { return this->NumberOfIds; }

  /**
   * Return the id at location i.
   */
  svtkIdType GetId(const svtkIdType i) SVTK_EXPECTS(0 <= i && i < GetNumberOfIds())
  {
    return this->Ids[i];
  }

  /**
   * Find the location i of the provided id.
   */
  svtkIdType FindIdLocation(const svtkIdType id)
  {
    for (int i = 0; i < this->NumberOfIds; i++)
      if (this->Ids[i] == id)
        return i;
    return -1;
  }

  /**
   * Specify the number of ids for this object to hold. Does an
   * allocation as well as setting the number of ids.
   */
  void SetNumberOfIds(const svtkIdType number);

  /**
   * Set the id at location i. Doesn't do range checking so it's a bit
   * faster than InsertId. Make sure you use SetNumberOfIds() to allocate
   * memory prior to using SetId().
   */
  void SetId(const svtkIdType i, const svtkIdType svtkid) SVTK_EXPECTS(0 <= i && i < GetNumberOfIds())
  {
    this->Ids[i] = svtkid;
  }

  /**
   * Set the id at location i. Does range checking and allocates memory
   * as necessary.
   */
  void InsertId(const svtkIdType i, const svtkIdType svtkid) SVTK_EXPECTS(0 <= i);

  /**
   * Add the id specified to the end of the list. Range checking is performed.
   */
  svtkIdType InsertNextId(const svtkIdType svtkid);

  /**
   * If id is not already in list, insert it and return location in
   * list. Otherwise return just location in list.
   */
  svtkIdType InsertUniqueId(const svtkIdType svtkid);

  /**
   * Sort the ids in the list in ascending id order. This method uses
   * svtkSMPTools::Sort() so it can be sped up if built properly.
   */
  void Sort();

  /**
   * Get a pointer to a particular data index.
   */
  svtkIdType* GetPointer(const svtkIdType i) { return this->Ids + i; }

  /**
   * Get a pointer to a particular data index. Make sure data is allocated
   * for the number of items requested. Set MaxId according to the number of
   * data values requested.
   */
  svtkIdType* WritePointer(const svtkIdType i, const svtkIdType number);

  /**
   * Specify an array of svtkIdType to use as the id list. This replaces the
   * underlying array. This instance of svtkIdList takes ownership of the
   * array, meaning that it deletes it on destruction (using delete[]).
   */
  void SetArray(svtkIdType* array, svtkIdType size);

  /**
   * Reset to an empty state but retain previously allocated memory.
   */
  void Reset() { this->NumberOfIds = 0; }

  /**
   * Free any unused memory.
   */
  void Squeeze() { this->Resize(this->NumberOfIds); }

  /**
   * Copy an id list by explicitly copying the internal array.
   */
  void DeepCopy(svtkIdList* ids);

  /**
   * Delete specified id from list. Will remove all occurrences of id in list.
   */
  void DeleteId(svtkIdType svtkid);

  /**
   * Return -1 if id specified is not contained in the list; otherwise return
   * the position in the list.
   */
  svtkIdType IsId(svtkIdType svtkid);

  /**
   * Intersect this list with another svtkIdList. Updates current list according
   * to result of intersection operation.
   */
  void IntersectWith(svtkIdList* otherIds);

  /**
   * Adjust the size of the id list while maintaining its content (except
   * when being truncated).
   */
  svtkIdType* Resize(const svtkIdType sz);

  /**
   * Intersect one id list with another. This method should become legacy.
   */
  void IntersectWith(svtkIdList& otherIds) { this->IntersectWith(&otherIds); }

  //@{
  /**
   * To support range-based `for` loops
   */
  svtkIdType* begin() { return this->Ids; }
  svtkIdType* end() { return this->Ids + this->NumberOfIds; }
  const svtkIdType* begin() const { return this->Ids; }
  const svtkIdType* end() const { return this->Ids + this->NumberOfIds; }
  //@}
protected:
  svtkIdList();
  ~svtkIdList() override;

  svtkIdType NumberOfIds;
  svtkIdType Size;
  svtkIdType* Ids;

private:
  svtkIdList(const svtkIdList&) = delete;
  void operator=(const svtkIdList&) = delete;
};

// In-lined for performance
inline void svtkIdList::InsertId(const svtkIdType i, const svtkIdType svtkid)
{
  if (i >= this->Size)
  {
    this->Resize(i + 1);
  }
  this->Ids[i] = svtkid;
  if (i >= this->NumberOfIds)
  {
    this->NumberOfIds = i + 1;
  }
}

// In-lined for performance
inline svtkIdType svtkIdList::InsertNextId(const svtkIdType svtkid)
{
  if (this->NumberOfIds >= this->Size)
  {
    if (!this->Resize(2 * this->NumberOfIds + 1)) // grow by factor of 2
    {
      return this->NumberOfIds - 1;
    }
  }
  this->Ids[this->NumberOfIds++] = svtkid;
  return this->NumberOfIds - 1;
}

inline svtkIdType svtkIdList::IsId(svtkIdType svtkid)
{
  svtkIdType *ptr, i;
  for (ptr = this->Ids, i = 0; i < this->NumberOfIds; i++, ptr++)
  {
    if (svtkid == *ptr)
    {
      return i;
    }
  }
  return (-1);
}

#endif
