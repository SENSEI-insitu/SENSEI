/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataObjectTreeInternals.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDataObjectTreeInternals
 *
 */

#ifndef svtkDataObjectTreeInternals_h
#define svtkDataObjectTreeInternals_h

#include "svtkDataObject.h"
#include "svtkInformation.h"
#include "svtkSmartPointer.h"

#include <vector>

//-----------------------------------------------------------------------------
// Item in the VectorOfDataObjects.
struct svtkDataObjectTreeItem
{
  svtkSmartPointer<svtkDataObject> DataObject;
  svtkSmartPointer<svtkInformation> MetaData;

  svtkDataObjectTreeItem(svtkDataObject* dobj = nullptr, svtkInformation* info = nullptr)
  {
    this->DataObject = dobj;
    this->MetaData = info;
  }
};

//-----------------------------------------------------------------------------
class svtkDataObjectTreeInternals
{
public:
  typedef std::vector<svtkDataObjectTreeItem> VectorOfDataObjects;
  typedef VectorOfDataObjects::iterator Iterator;
  typedef VectorOfDataObjects::reverse_iterator ReverseIterator;

  VectorOfDataObjects Children;
};

//-----------------------------------------------------------------------------
class svtkDataObjectTreeIndex : public std::vector<unsigned int>
{
  int IsValid() { return (this->size() > 0); }
};

#endif

// SVTK-HeaderTest-Exclude: svtkDataObjectTreeInternals.h
