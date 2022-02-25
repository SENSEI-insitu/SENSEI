/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkStaticCellLinks.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkStaticCellLinks
 * @brief   object represents upward pointers from points
 * to list of cells using each point
 *
 * svtkStaticCellLinks is a supplemental object to svtkCellArray and
 * svtkCellTypes, enabling access from points to the cells using the
 * points. svtkStaticCellLinks is an array of links, each link represents a
 * list of cell ids using a particular point. The information provided by
 * this object can be used to determine cell neighbors and construct other
 * local topological information. This class is a faster implementation of
 * svtkCellLinks. However, it cannot be incrementally constructed; it is meant
 * to be constructed once (statically) and must be rebuilt if the cells
 * change.
 *
 * @warning
 * This is a drop-in replacement for svtkCellLinks using static link
 * construction. It uses the templated svtkStaticCellLinksTemplate class,
 * instantiating svtkStaticCellLinksTemplate with a svtkIdType template
 * parameter. Note that for best performance, the svtkStaticCellLinksTemplate
 * class may be used directly, instantiating it with the appropriate id
 * type. This class is also wrappable and can be used from an interpreted
 * language such as Python.
 *
 * @sa
 * svtkCellLinks svtkStaticCellLinksTemplate
 */

#ifndef svtkStaticCellLinks_h
#define svtkStaticCellLinks_h

#include "svtkAbstractCellLinks.h"
#include "svtkCommonDataModelModule.h"   // For export macro
#include "svtkStaticCellLinksTemplate.h" // For implementations

class svtkDataSet;
class svtkCellArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkStaticCellLinks : public svtkAbstractCellLinks
{
public:
  //@{
  /**
   * Standard methods for instantiation, type manipulation and printing.
   */
  static svtkStaticCellLinks* New();
  svtkTypeMacro(svtkStaticCellLinks, svtkAbstractCellLinks);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  /**
   * Build the link list array. Satisfy the superclass API.
   */
  void BuildLinks(svtkDataSet* ds) override
  {
    this->Impl->SetSequentialProcessing(this->SequentialProcessing);
    this->Impl->BuildLinks(ds);
  }

  /**
   * Get the number of cells using the point specified by ptId.
   */
  svtkIdType GetNumberOfCells(svtkIdType ptId) { return this->Impl->GetNumberOfCells(ptId); }

  /**
   * Get the number of cells using the point specified by ptId. This is an
   * alias for GetNumberOfCells(); consistent with the svtkCellLinks API.
   */
  svtkIdType GetNcells(svtkIdType ptId) { return this->Impl->GetNumberOfCells(ptId); }

  /**
   * Return a list of cell ids using the specified point.
   */
  svtkIdType* GetCells(svtkIdType ptId) { return this->Impl->GetCells(ptId); }

  /**
   * Make sure any previously created links are cleaned up.
   */
  void Initialize() override { this->Impl->Initialize(); }

  /**
   * Reclaim any unused memory.
   */
  void Squeeze() override {}

  /**
   * Reset to a state of no entries without freeing the memory.
   */
  void Reset() override {}

  /**
   * Return the memory in kibibytes (1024 bytes) consumed by this cell links array.
   * Used to support streaming and reading/writing data. The value
   * returned is guaranteed to be greater than or equal to the memory
   * required to actually represent the data represented by this object.
   * The information returned is valid only after the pipeline has
   * been updated.
   */
  unsigned long GetActualMemorySize() override { return this->Impl->GetActualMemorySize(); }

  /**
   * Standard DeepCopy method.  Since this object contains no reference
   * to other objects, there is no ShallowCopy.
   */
  void DeepCopy(svtkAbstractCellLinks* src) override { this->Impl->DeepCopy(src); }

protected:
  svtkStaticCellLinks();
  ~svtkStaticCellLinks() override;

  svtkStaticCellLinksTemplate<svtkIdType>* Impl;

private:
  svtkStaticCellLinks(const svtkStaticCellLinks&) = delete;
  void operator=(const svtkStaticCellLinks&) = delete;
};

#endif
