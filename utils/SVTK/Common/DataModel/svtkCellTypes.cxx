/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCellTypes.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkCellTypes.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkCellTypes);

// This list should contain the cell class names in
// the same order as the enums in svtkCellType.h. Make sure
// this list is nullptr terminated.
static const char* svtkCellTypesStrings[] = { "svtkEmptyCell", "svtkVertex", "svtkPolyVertex",
  "svtkLine", "svtkPolyLine", "svtkTriangle", "svtkTriangleStrip", "svtkPolygon", "svtkPixel", "svtkQuad",
  "svtkTetra", "svtkVoxel", "svtkHexahedron", "svtkWedge", "svtkPyramid", "svtkPentagonalPrism",
  "svtkHexagonalPrism", "UnknownClass", "UnknownClass", "UnknownClass", "UnknownClass",
  "svtkQuadraticEdge", "svtkQuadraticTriangle", "svtkQuadraticQuad", "svtkQuadraticTetra",
  "svtkQuadraticHexahedron", "svtkQuadraticWedge", "svtkQuadraticPyramid", "svtkBiQuadraticQuad",
  "svtkTriQuadraticHexahedron", "svtkQuadraticLinearQuad", "svtkQuadraticLinearWedge",
  "svtkBiQuadraticQuadraticWedge", "svtkBiQuadraticQuadraticHexahedron", "svtkBiQuadraticTriangle",
  "svtkCubicLine", "svtkQuadraticPolygon", "UnknownClass", "UnknownClass", "UnknownClass",
  "UnknownClass", "svtkConvexPointSet", "svtkPolyhedron", "UnknownClass", "UnknownClass",
  "UnknownClass", "UnknownClass", "UnknownClass", "UnknownClass", "UnknownClass", "UnknownClass",
  "svtkParametricCurve", "svtkParametricSurface", "svtkParametricTriSurface",
  "svtkParametricQuadSurface", "svtkParametricTetraRegion", "svtkParametricHexRegion", "UnknownClass",
  "UnknownClass", "UnknownClass", "svtkHigherOrderEdge", "svtkHigherOrderTriangle",
  "svtkHigherOrderQuad", "svtkHigherOrderPolygon", "svtkHigherOrderTetrahedron", "svtkHigherOrderWedge",
  "svtkHigherOrderPyramid", "svtkHigherOrderHexahedron", "svtkLagrangeCurve",
  "svtkLagrangeQuadrilateral", "svtkLagrangeTriangle", "svtkLagrangeTetra", "svtkLagrangeHexahedron",
  "svtkLagrangeWedge", "svtkLagrangePyramid", "svtkBezierCurve", "svtkBezierQuadrilateral",
  "svtkBezierTriangle", "svtkBezierTetra", "svtkBezierHexahedron", "svtkBezierWedge",
  "svtkBezierPyramid", nullptr };

//----------------------------------------------------------------------------
const char* svtkCellTypes::GetClassNameFromTypeId(int type)
{
  static int numClasses = 0;

  // find length of table
  if (numClasses == 0)
  {
    while (svtkCellTypesStrings[numClasses] != nullptr)
    {
      numClasses++;
    }
  }

  if (type < numClasses)
  {
    return svtkCellTypesStrings[type];
  }
  else
  {
    return "UnknownClass";
  }
}

//----------------------------------------------------------------------------
int svtkCellTypes::GetTypeIdFromClassName(const char* classname)
{
  if (!classname)
  {
    return -1;
  }

  for (int idx = 0; svtkCellTypesStrings[idx] != nullptr; idx++)
  {
    if (strcmp(svtkCellTypesStrings[idx], classname) == 0)
    {
      return idx;
    }
  }

  return -1;
}

//----------------------------------------------------------------------------
svtkCellTypes::svtkCellTypes()
  : TypeArray(svtkUnsignedCharArray::New())
  , LocationArray(svtkIdTypeArray::New())
  , Size(0)
  , MaxId(-1)
  , Extend(1000)
{
  this->TypeArray->Register(this);
  this->TypeArray->Delete();

  this->LocationArray->Register(this);
  this->LocationArray->Delete();
}

//----------------------------------------------------------------------------
svtkCellTypes::~svtkCellTypes()
{
  if (this->TypeArray)
  {
    this->TypeArray->UnRegister(this);
  }

  if (this->LocationArray)
  {
    this->LocationArray->UnRegister(this);
  }
}

//----------------------------------------------------------------------------
// Allocate memory for this array. Delete old storage only if necessary.
int svtkCellTypes::Allocate(svtkIdType sz, svtkIdType ext)
{

  this->Size = (sz > 0 ? sz : 1);
  this->Extend = (ext > 0 ? ext : 1);
  this->MaxId = -1;

  if (this->TypeArray)
  {
    this->TypeArray->UnRegister(this);
  }
  this->TypeArray = svtkUnsignedCharArray::New();
  this->TypeArray->Allocate(sz, ext);
  this->TypeArray->Register(this);
  this->TypeArray->Delete();

  if (this->LocationArray)
  {
    this->LocationArray->UnRegister(this);
  }
  this->LocationArray = svtkIdTypeArray::New();
  this->LocationArray->Allocate(sz, ext);
  this->LocationArray->Register(this);
  this->LocationArray->Delete();

  return 1;
}

//----------------------------------------------------------------------------
// Add a cell at specified id.
void svtkCellTypes::InsertCell(svtkIdType cellId, unsigned char type, svtkIdType loc)
{
  svtkDebugMacro(<< "Insert Cell id: " << cellId << " at location " << loc);
  TypeArray->InsertValue(cellId, type);

  LocationArray->InsertValue(cellId, loc);

  if (cellId > this->MaxId)
  {
    this->MaxId = cellId;
  }
}

//----------------------------------------------------------------------------
// Add a cell to the object in the next available slot.
svtkIdType svtkCellTypes::InsertNextCell(unsigned char type, svtkIdType loc)
{
  svtkDebugMacro(<< "Insert Next Cell " << type << " location " << loc);
  this->InsertCell(++this->MaxId, type, loc);
  return this->MaxId;
}

//----------------------------------------------------------------------------
// Specify a group of cell types.
void svtkCellTypes::SetCellTypes(
  svtkIdType ncells, svtkUnsignedCharArray* cellTypes, svtkIntArray* cellLocations)
{
  svtkIdTypeArray* cellLocations64 = svtkIdTypeArray::New();
  cellLocations64->SetName(cellLocations->GetName());
  cellLocations64->SetNumberOfComponents(cellLocations->GetNumberOfComponents());
  cellLocations64->SetNumberOfTuples(cellLocations->GetNumberOfTuples());
  for (svtkIdType i = 0, iend = cellLocations->GetNumberOfValues(); i < iend; ++i)
  {
    cellLocations64->SetValue(i, cellLocations->GetValue(i));
  }
  this->SetCellTypes(ncells, cellTypes, cellLocations64);
  cellLocations64->Delete();
}

//----------------------------------------------------------------------------
// Specify a group of cell types.
void svtkCellTypes::SetCellTypes(
  svtkIdType ncells, svtkUnsignedCharArray* cellTypes, svtkIdTypeArray* cellLocations)
{
  this->Size = ncells;

  if (this->TypeArray)
  {
    this->TypeArray->Delete();
  }

  this->TypeArray = cellTypes;
  cellTypes->Register(this);

  if (this->LocationArray)
  {
    this->LocationArray->Delete();
  }
  this->LocationArray = cellLocations;
  cellLocations->Register(this);

  this->MaxId = ncells - 1;
}

//----------------------------------------------------------------------------
// Reclaim any extra memory.
void svtkCellTypes::Squeeze()
{
  this->TypeArray->Squeeze();
  this->LocationArray->Squeeze();
}

//----------------------------------------------------------------------------
// Initialize object without releasing memory.
void svtkCellTypes::Reset()
{
  this->MaxId = -1;
}

//----------------------------------------------------------------------------
unsigned long svtkCellTypes::GetActualMemorySize()
{
  unsigned long size = 0;

  if (this->TypeArray)
  {
    size += this->TypeArray->GetActualMemorySize();
  }

  if (this->LocationArray)
  {
    size += this->LocationArray->GetActualMemorySize();
  }

  return static_cast<unsigned long>(ceil(size / 1024.0)); // kibibytes
}

//----------------------------------------------------------------------------
void svtkCellTypes::DeepCopy(svtkCellTypes* src)
{
  if (this->TypeArray)
  {
    this->TypeArray->UnRegister(this);
    this->TypeArray = nullptr;
  }
  if (src->TypeArray)
  {
    this->TypeArray = svtkUnsignedCharArray::New();
    this->TypeArray->DeepCopy(src->TypeArray);
    this->TypeArray->Register(this);
    this->TypeArray->Delete();
  }

  if (this->LocationArray)
  {
    this->LocationArray->UnRegister(this);
    this->LocationArray = nullptr;
  }
  if (src->LocationArray)
  {
    this->LocationArray = svtkIdTypeArray::New();
    this->LocationArray->DeepCopy(src->LocationArray);
    this->LocationArray->Register(this);
    this->LocationArray->Delete();
  }
  this->Size = src->Size;
  this->Extend = src->Extend;
  this->MaxId = src->MaxId;
}

//----------------------------------------------------------------------------
void svtkCellTypes::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "TypeArray:\n";
  this->TypeArray->PrintSelf(os, indent.GetNextIndent());
  os << indent << "LocationArray:\n";
  this->LocationArray->PrintSelf(os, indent.GetNextIndent());

  os << indent << "Size: " << this->Size << "\n";
  os << indent << "MaxId: " << this->MaxId << "\n";
  os << indent << "Extend: " << this->Extend << "\n";
}
