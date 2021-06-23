/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCellTypes.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCellTypes
 * @brief   object provides direct access to cells in svtkCellArray and type information
 *
 * This class is a supplemental object to svtkCellArray to allow random access
 * into cells as well as representing cell type information.  The "location"
 * field is the location in the svtkCellArray list in terms of an integer
 * offset.  An integer offset was used instead of a pointer for easy storage
 * and inter-process communication. The type information is defined in the
 * file svtkCellType.h.
 *
 * @warning
 * Sometimes this class is used to pass type information independent of the
 * random access (i.e., location) information. For example, see
 * svtkDataSet::GetCellTypes(). If you use the class in this way, you can use
 * a location value of -1.
 *
 * @sa
 * svtkCellArray svtkCellLinks
 */

#ifndef svtkCellTypes_h
#define svtkCellTypes_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

#include "svtkCellType.h"          // Needed for SVTK_EMPTY_CELL
#include "svtkIdTypeArray.h"       // Needed for inline methods
#include "svtkIntArray.h"          // Needed for inline methods
#include "svtkUnsignedCharArray.h" // Needed for inline methods

class SVTKCOMMONDATAMODEL_EXPORT svtkCellTypes : public svtkObject
{
public:
  static svtkCellTypes* New();
  svtkTypeMacro(svtkCellTypes, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Allocate memory for this array. Delete old storage only if necessary.
   */
  int Allocate(svtkIdType sz = 512, svtkIdType ext = 1000);

  /**
   * Add a cell at specified id.
   */
  void InsertCell(svtkIdType id, unsigned char type, svtkIdType loc);

  /**
   * Add a cell to the object in the next available slot.
   */
  svtkIdType InsertNextCell(unsigned char type, svtkIdType loc);

  /**
   * Specify a group of cell types.
   */
  void SetCellTypes(
    svtkIdType ncells, svtkUnsignedCharArray* cellTypes, svtkIdTypeArray* cellLocations);

  /**
   * Specify a group of cell types. This version is provided to maintain
   * backwards compatibility and does a copy of the cellLocations
   */
  void SetCellTypes(svtkIdType ncells, svtkUnsignedCharArray* cellTypes, svtkIntArray* cellLocations);

  /**
   * Return the location of the cell in the associated svtkCellArray.
   */
  svtkIdType GetCellLocation(svtkIdType cellId) { return this->LocationArray->GetValue(cellId); }

  /**
   * Delete cell by setting to nullptr cell type.
   */
  void DeleteCell(svtkIdType cellId) { this->TypeArray->SetValue(cellId, SVTK_EMPTY_CELL); }

  /**
   * Return the number of types in the list.
   */
  svtkIdType GetNumberOfTypes() { return (this->MaxId + 1); }

  /**
   * Return 1 if type specified is contained in list; 0 otherwise.
   */
  int IsType(unsigned char type);

  /**
   * Add the type specified to the end of the list. Range checking is performed.
   */
  svtkIdType InsertNextType(unsigned char type) { return this->InsertNextCell(type, -1); }

  /**
   * Return the type of cell.
   */
  unsigned char GetCellType(svtkIdType cellId) { return this->TypeArray->GetValue(cellId); }

  /**
   * Reclaim any extra memory.
   */
  void Squeeze();

  /**
   * Initialize object without releasing memory.
   */
  void Reset();

  /**
   * Return the memory in kibibytes (1024 bytes) consumed by this cell type array.
   * Used to support streaming and reading/writing data. The value
   * returned is guaranteed to be greater than or equal to the memory
   * required to actually represent the data represented by this object.
   * The information returned is valid only after the pipeline has
   * been updated.
   */
  unsigned long GetActualMemorySize();

  /**
   * Standard DeepCopy method.  Since this object contains no reference
   * to other objects, there is no ShallowCopy.
   */
  void DeepCopy(svtkCellTypes* src);

  /**
   * Given an int (as defined in svtkCellType.h) identifier for a class
   * return it's classname.
   */
  static const char* GetClassNameFromTypeId(int typeId);

  /**
   * Given a data object classname, return it's int identified (as
   * defined in svtkCellType.h)
   */
  static int GetTypeIdFromClassName(const char* classname);

  /**
   * This convenience method is a fast check to determine if a cell type
   * represents a linear or nonlinear cell.  This is generally much more
   * efficient than getting the appropriate svtkCell and checking its IsLinear
   * method.
   */
  static int IsLinear(unsigned char type);

  //@{
  /**
   * Methods for obtaining the arrays representing types and locations.
   */
  svtkUnsignedCharArray* GetCellTypesArray() { return this->TypeArray; }
  svtkIdTypeArray* GetCellLocationsArray() { return this->LocationArray; }
  //@}

protected:
  svtkCellTypes();
  ~svtkCellTypes() override;

  svtkUnsignedCharArray* TypeArray; // pointer to types array
  svtkIdTypeArray* LocationArray;   // pointer to array of offsets
  svtkIdType Size;                  // allocated size of data
  svtkIdType MaxId;                 // maximum index inserted thus far
  svtkIdType Extend;                // grow array by this point

private:
  svtkCellTypes(const svtkCellTypes&) = delete;
  void operator=(const svtkCellTypes&) = delete;
};

//----------------------------------------------------------------------------
inline int svtkCellTypes::IsType(unsigned char type)
{
  svtkIdType numTypes = this->GetNumberOfTypes();

  for (svtkIdType i = 0; i < numTypes; i++)
  {
    if (type == this->GetCellType(i))
    {
      return 1;
    }
  }
  return 0;
}

//-----------------------------------------------------------------------------
inline int svtkCellTypes::IsLinear(unsigned char type)
{
  return ((type <= 20) || (type == SVTK_CONVEX_POINT_SET) || (type == SVTK_POLYHEDRON));
}

#endif
