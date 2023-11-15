/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataObjectToConduit.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class svtkDataObjectToConduit
 * @brief Convert VTK Data Object to Conduit Node
 */

#ifndef svtkDataObjectToConduit_h
#define svtkDataObjectToConduit_h

#include "svtkObject.h"

namespace conduit_cpp
{
class Node;
}

class svtkDataObject;

namespace svtkDataObjectToConduit
{
/**
 * Fill the given conduit node with the data from the data object.
 * The final structure is a valid blueprint mesh.
 *
 * At the moment, only svtkDataSet are supported.
 */
bool FillConduitNode(
  svtkDataObject* data_object, conduit_cpp::Node& conduit_node);
}

#endif
// VTK-HeaderTest-Exclude: vtkDataObjectToConduit.h
