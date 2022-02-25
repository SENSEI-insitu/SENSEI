/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkStaticCellLinksTemplate.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkStaticCellLinksTemplate
 * @brief   object represents upward pointers from points
 * to list of cells using each point (template implementation)
 *
 *
 * svtkStaticCellLinksTemplate is a supplemental object to svtkCellArray and
 * svtkCellTypes, enabling access to the list of cells using each point.
 * svtkStaticCellLinksTemplate is an array of links, each link represents a
 * list of cell ids using a particular point. The information provided by
 * this object can be used to determine neighbors (e.g., face neighbors,
 * edge neighbors)and construct other local topological information. This
 * class is a faster implementation of svtkCellLinks. However, it cannot be
 * incrementally constructed; it is meant to be constructed once (statically)
 * and must be rebuilt if the cells change.
 *
 * This is a templated implementation for svtkStaticCellLinks. The reason for
 * the templating is to gain performance and reduce memory by using smaller
 * integral types to represent ids. For example, if the maximum id can be
 * represented by an int (as compared to a svtkIdType), it is possible to
 * reduce memory requirements by half and increase performance. This
 * templated class can be used directly; alternatively the
 * non-templated class svtkStaticCellLinks can be used for convenience;
 * although it uses svtkIdType and so will lose some speed and memory
 * advantages.
 *
 * @sa
 * svtkAbstractCellLinks svtkCellLinks svtkStaticCellLinks
 */

#ifndef svtkStaticCellLinksTemplate_h
#define svtkStaticCellLinksTemplate_h

class svtkDataSet;
class svtkPolyData;
class svtkUnstructuredGrid;
class svtkExplicitStructuredGrid;
class svtkCellArray;

#include "svtkAbstractCellLinks.h"

template <typename TIds>
class svtkStaticCellLinksTemplate
{
public:
  //@{
  /**
   * Instantiate and destructor methods.
   */
  svtkStaticCellLinksTemplate();
  ~svtkStaticCellLinksTemplate();
  //@}

  /**
   * Make sure any previously created links are cleaned up.
   */
  void Initialize();

  /**
   * Build the link list array for a general dataset. Slower than the
   * specialized methods that follow.
   */
  void BuildLinks(svtkDataSet* ds);

  /**
   * Build the link list array for svtkPolyData.
   */
  void BuildLinks(svtkPolyData* pd);

  /**
   * Build the link list array for svtkUnstructuredGrid.
   */
  void BuildLinks(svtkUnstructuredGrid* ugrid);

  /**
   * Build the link list array for svtkExplicitStructuredGrid.
   */
  void BuildLinks(svtkExplicitStructuredGrid* esgrid);

  /**
   * Specialized methods for building links from cell array.
   */
  void SerialBuildLinks(const svtkIdType numPts, const svtkIdType numCells, svtkCellArray* cellArray);
  void ThreadedBuildLinks(
    const svtkIdType numPts, const svtkIdType numCells, svtkCellArray* cellArray);

  //@{
  /**
   * Get the number of cells using the point specified by ptId.
   */
  TIds GetNumberOfCells(svtkIdType ptId) { return (this->Offsets[ptId + 1] - this->Offsets[ptId]); }
  svtkIdType GetNcells(svtkIdType ptId) { return (this->Offsets[ptId + 1] - this->Offsets[ptId]); }
  //@}

  /**
   * Return a list of cell ids using the point specified by ptId.
   */
  TIds* GetCells(svtkIdType ptId) { return (this->Links + this->Offsets[ptId]); }

  //@{
  /**
   * Support svtkAbstractCellLinks API.
   */
  unsigned long GetActualMemorySize();
  void DeepCopy(svtkAbstractCellLinks* src);
  //@}

  //@{
  /**
   * Control whether to thread or serial process.
   */
  void SetSequentialProcessing(svtkTypeBool seq) { this->SequentialProcessing = seq; }
  svtkTypeBool GetSequentialProcessing() { return this->SequentialProcessing; }
  //@}

protected:
  // The various templated data members
  TIds LinksSize;
  TIds NumPts;
  TIds NumCells;

  // These point to the core data structures
  TIds* Links;   // contiguous runs of cell ids
  TIds* Offsets; // offsets for each point into the links array

  // Support for execution
  int Type;
  svtkTypeBool SequentialProcessing;

private:
  svtkStaticCellLinksTemplate(const svtkStaticCellLinksTemplate&) = delete;
  void operator=(const svtkStaticCellLinksTemplate&) = delete;
};

#include "svtkStaticCellLinksTemplate.txx"

#endif
// SVTK-HeaderTest-Exclude: svtkStaticCellLinksTemplate.h
