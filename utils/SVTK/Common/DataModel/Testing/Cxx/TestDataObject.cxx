/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestVariant.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <cassert>

#include <svtkDataObject.h>

int TestGetAssociationTypeFromString()
{
  assert(svtkDataObject::GetAssociationTypeFromString("svtkDataObject::FIELD_ASSOCIATION_POINTS") ==
    svtkDataObject::FIELD_ASSOCIATION_POINTS);
  assert(svtkDataObject::GetAssociationTypeFromString("svtkDataObject::FIELD_ASSOCIATION_CELLS") ==
    svtkDataObject::FIELD_ASSOCIATION_CELLS);
  assert(svtkDataObject::GetAssociationTypeFromString("svtkDataObject::FIELD_ASSOCIATION_NONE") ==
    svtkDataObject::FIELD_ASSOCIATION_NONE);
  assert(svtkDataObject::GetAssociationTypeFromString(
           "svtkDataObject::FIELD_ASSOCIATION_POINTS_THEN_CELLS") ==
    svtkDataObject::FIELD_ASSOCIATION_POINTS_THEN_CELLS);
  assert(svtkDataObject::GetAssociationTypeFromString("svtkDataObject::FIELD_ASSOCIATION_VERTICES") ==
    svtkDataObject::FIELD_ASSOCIATION_VERTICES);
  assert(svtkDataObject::GetAssociationTypeFromString("svtkDataObject::FIELD_ASSOCIATION_EDGES") ==
    svtkDataObject::FIELD_ASSOCIATION_EDGES);
  assert(svtkDataObject::GetAssociationTypeFromString("svtkDataObject::FIELD_ASSOCIATION_ROWS") ==
    svtkDataObject::FIELD_ASSOCIATION_ROWS);

  assert(
    svtkDataObject::GetAssociationTypeFromString("svtkDataObject::POINT") == svtkDataObject::POINT);
  assert(svtkDataObject::GetAssociationTypeFromString("svtkDataObject::CELL") == svtkDataObject::CELL);
  assert(
    svtkDataObject::GetAssociationTypeFromString("svtkDataObject::FIELD") == svtkDataObject::FIELD);
  assert(svtkDataObject::GetAssociationTypeFromString("svtkDataObject::POINT_THEN_CELL") ==
    svtkDataObject::POINT_THEN_CELL);
  assert(
    svtkDataObject::GetAssociationTypeFromString("svtkDataObject::VERTEX") == svtkDataObject::VERTEX);
  assert(svtkDataObject::GetAssociationTypeFromString("svtkDataObject::EDGE") == svtkDataObject::EDGE);
  assert(svtkDataObject::GetAssociationTypeFromString("svtkDataObject::ROW") == svtkDataObject::ROW);

  assert(svtkDataObject::GetAssociationTypeFromString(nullptr) == -1);
  assert(svtkDataObject::GetAssociationTypeFromString("") == -1);
  assert(svtkDataObject::GetAssociationTypeFromString("INVALID") == -1);

  return 0;
}

int TestDataObject(int, char*[])
{
  return TestGetAssociationTypeFromString();
}
