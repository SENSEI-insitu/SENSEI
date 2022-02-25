/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGenericAdaptorCell.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkGenericAdaptorCell.h"

#include <cassert>

#include "svtkCellArray.h"
#include "svtkCellData.h"
#include "svtkContourValues.h"
#include "svtkDoubleArray.h"
#include "svtkGenericAttribute.h"
#include "svtkGenericAttributeCollection.h"
#include "svtkGenericCellTessellator.h"
#include "svtkHexahedron.h"
#include "svtkImplicitFunction.h"
#include "svtkIncrementalPointLocator.h"
#include "svtkLine.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkPyramid.h"
#include "svtkQuad.h"
#include "svtkTetra.h"
#include "svtkTriangle.h"
#include "svtkUnsignedCharArray.h"
#include "svtkVertex.h"
#include "svtkWedge.h"

svtkGenericAdaptorCell::svtkGenericAdaptorCell()
{
  this->Tetra = svtkTetra::New();
  this->Triangle = svtkTriangle::New();
  this->Line = svtkLine::New();
  this->Vertex = svtkVertex::New();
  this->Hexa = svtkHexahedron::New();
  this->Quad = svtkQuad::New();
  this->Wedge = svtkWedge::New();
  this->Pyramid = svtkPyramid::New();

  this->Scalars = svtkDoubleArray::New();
  this->Scalars->SetNumberOfTuples(8); // up to 8 points with a linear hexa
  this->PointData = svtkPointData::New();
  this->CellData = svtkCellData::New();

  // Internal array to avoid New/ Delete
  this->InternalPoints = svtkDoubleArray::New();
  this->InternalPoints->SetNumberOfComponents(3);
  this->InternalScalars = svtkDoubleArray::New();
  this->InternalCellArray = svtkCellArray::New();
  this->InternalIds = svtkIdList::New();

  this->PointDataScalars = svtkDoubleArray::New();
  this->PointData->SetScalars(this->PointDataScalars);
  this->PointDataScalars->Delete();

  this->Tuples = nullptr;
  this->TuplesCapacity = 0;
}

//----------------------------------------------------------------------------
svtkGenericAdaptorCell::~svtkGenericAdaptorCell()
{
  this->Tetra->Delete();
  this->Triangle->Delete();
  this->Line->Delete();
  this->Vertex->Delete();
  this->Hexa->Delete();
  this->Quad->Delete();
  this->Wedge->Delete();
  this->Pyramid->Delete();

  this->Scalars->Delete();
  this->PointData->Delete();
  this->CellData->Delete();

  this->InternalPoints->Delete();
  this->InternalScalars->Delete();
  this->InternalCellArray->Delete();
  this->InternalIds->Delete();

  delete[] this->Tuples;
}

//----------------------------------------------------------------------------

void svtkGenericAdaptorCell::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
//----------------------------------------------------------------------------
// Description:
// Does the cell have no higher-order interpolation for geometry?
// \post definition: result==(GetGeometryOrder()==1)
int svtkGenericAdaptorCell::IsGeometryLinear()
{
  return this->GetGeometryOrder() == 1;
}

//----------------------------------------------------------------------------
// Description:
// Return the index of the first point centered attribute with the highest
// order in `ac'.
// \pre ac_exists: ac!=0
// \post valid_result: result>=-1 && result<ac->GetNumberOfAttributes()
int svtkGenericAdaptorCell::GetHighestOrderAttribute(svtkGenericAttributeCollection* ac)
{
  assert("pre: ac_exists" && ac != nullptr);
  int result = -1;
  int highestOrder = -1;
  int order;
  svtkGenericAttribute* a;
  int c = ac->GetNumberOfAttributes();
  int i = 0;
  while (i < c)
  {
    a = ac->GetAttribute(i);
    if (a->GetCentering() == svtkPointCentered)
    {
      order = this->GetAttributeOrder(a);
      if (order > highestOrder)
      {
        highestOrder = order;
        result = i;
      }
    }
    ++i;
  }
  assert("post: valid_result" && result >= -1 && result < ac->GetNumberOfAttributes());
  return result;
}

//----------------------------------------------------------------------------
// Description:
// Does the attribute `a' have no higher-order interpolation for the cell?
// \pre a_exists: a!=0
// \post definition: result==(GetAttributeOrder()==1)
svtkTypeBool svtkGenericAdaptorCell::IsAttributeLinear(svtkGenericAttribute* a)
{
  return this->GetAttributeOrder(a) == 1;
}

//----------------------------------------------------------------------------
void svtkGenericAdaptorCell::GetBounds(double bounds[6])
{
#if 0
  double x[3];
  int i, numPts=this->GetNumberOfPoints();

  if (numPts)
  {
    this->GetPoints()->GetPoint(0, x);
    bounds[0] = x[0];
    bounds[2] = x[1];
    bounds[4] = x[2];
    bounds[1] = x[0];
    bounds[3] = x[1];
    bounds[5] = x[2];
    for (i=1; i<numPts; i++)
    {
      this->GetPoints()->GetPoint(i, x);
      bounds[0] = (x[0] < bounds[0] ? x[0] : bounds[0]);
      bounds[1] = (x[0] > bounds[1] ? x[0] : bounds[1]);
      bounds[2] = (x[1] < bounds[2] ? x[1] : bounds[2]);
      bounds[3] = (x[1] > bounds[3] ? x[1] : bounds[3]);
      bounds[4] = (x[2] < bounds[4] ? x[2] : bounds[4]);
      bounds[5] = (x[2] > bounds[5] ? x[2] : bounds[5]);
    }
  }
#endif
  memset(bounds, 0, sizeof(double));
  svtkErrorMacro("TO BE DONE");
}

//----------------------------------------------------------------------------
double* svtkGenericAdaptorCell::GetBounds()
{
  this->GetBounds(this->Bounds);

  return this->Bounds;
}

//----------------------------------------------------------------------------
// Description:
// Return the bounding box diagonal squared of the current cell.
// \post positive_result: result>=0
double svtkGenericAdaptorCell::GetLength2()
{
  double diff, l = 0.0;
  int i;

  this->GetBounds(this->Bounds);
  for (i = 0; i < 3; i++)
  {
    diff = this->Bounds[2 * i + 1] - this->Bounds[2 * i];
    l += diff * diff;
  }
  return l;
}

//----------------------------------------------------------------------------
// Reset
void svtkGenericAdaptorCell::Reset()
{
  this->InternalPoints->Reset();
  this->InternalCellArray->Reset();
  this->InternalScalars->Reset();
}

//----------------------------------------------------------------------------
void svtkGenericAdaptorCell::Contour(svtkContourValues* contourValues, svtkImplicitFunction* f,
  svtkGenericAttributeCollection* attributes, svtkGenericCellTessellator* tess,
  svtkIncrementalPointLocator* locator, svtkCellArray* verts, svtkCellArray* lines,
  svtkCellArray* polys, svtkPointData* outPd, svtkCellData* outCd, svtkPointData* internalPd,
  svtkPointData* secondaryPd, svtkCellData* secondaryCd)
{
  assert("pre: values_exist" &&
    ((contourValues != nullptr && f == nullptr) || (contourValues != nullptr && f != nullptr)));
  assert("pre: attributes_exist" && attributes != nullptr);
  assert("pre: tessellator_exists" && tess != nullptr);
  assert("pre: locator_exists" && locator != nullptr);
  assert("pre: verts_exist" && verts != nullptr);
  assert("pre: lines_exist" && lines != nullptr);
  assert("pre: polys_exist" && polys != nullptr);
  assert("pre: internalPd_exists" && internalPd != nullptr);
  assert("pre: secondaryPd_exists" && secondaryPd != nullptr);
  assert("pre: secondaryCd_exists" && secondaryCd != nullptr);

  int i;
  int j;
  int c;
  double range[2];
  double contVal = -1000;
  double* values;

  svtkCell* linearCell = nullptr;
  svtkIdType ptsCount = 0;

  range[0] = 0; // to fix warning
  range[1] = 0; // on some compilers

  this->Reset();

  // for each cell-centered attribute: copy the value in the secondary
  // cell data.
  secondaryCd->Reset();
  int attrib = 0;
  while (attrib < attributes->GetNumberOfAttributes())
  {
    if (attributes->GetAttribute(attrib)->GetCentering() == svtkCellCentered)
    {
      svtkDataArray* array = secondaryCd->GetArray(attributes->GetAttribute(attrib)->GetName());
      values = attributes->GetAttribute(attrib)->GetTuple(this);
      array->InsertNextTuple(values);
    }
    attrib++;
  }

  int attribute = this->GetHighestOrderAttribute(attributes);
  if (this->IsGeometryLinear() &&
    (attribute == -1 || this->IsAttributeLinear(attributes->GetAttribute(attribute))))
  {
    // linear case
    switch (this->GetType())
    {
      case SVTK_HIGHER_ORDER_TRIANGLE:
        linearCell = this->Triangle;
        ptsCount = 3;
        break;
      case SVTK_HIGHER_ORDER_QUAD:
        linearCell = this->Quad;
        ptsCount = 4;
        break;
      case SVTK_HIGHER_ORDER_TETRAHEDRON:
        linearCell = this->Tetra;
        ptsCount = 4;
        break;
      case SVTK_HIGHER_ORDER_HEXAHEDRON:
        linearCell = this->Hexa;
        ptsCount = 8;
        break;
      case SVTK_HIGHER_ORDER_WEDGE:
        linearCell = this->Wedge;
        ptsCount = 6;
        break;
      case SVTK_HIGHER_ORDER_PYRAMID:
        linearCell = this->Pyramid;
        ptsCount = 5;
        break;
      default:
        assert("check: impossible case" && 0);
        return;
    }
    int currComp = attributes->GetActiveComponent();

    double* locals = this->GetParametricCoords();
    double point[3];

    svtkGenericAttribute* a;
    int count = attributes->GetNumberOfAttributes();
    int attribute_idx;

    values = contourValues->GetValues();
    int numContours = contourValues->GetNumberOfContours();

    this->AllocateTuples(attributes->GetMaxNumberOfComponents());

    int activeAttributeIdx = attributes->GetActiveAttribute();

    // build the cell
    i = 0;
    while (i < ptsCount)
    {
      this->EvaluateLocation(0, locals, point);
      linearCell->PointIds->SetId(i, i);
      linearCell->Points->SetPoint(i, point);

      // for each point-centered attribute
      secondaryPd->Reset();
      attribute_idx = 0;
      j = 0;
      while (attribute_idx < count)
      {
        a = attributes->GetAttribute(attribute_idx);
        if (a->GetCentering() == svtkPointCentered)
        {
          this->InterpolateTuple(a, locals, this->Tuples);
          secondaryPd->GetArray(j)->InsertTuple(i, this->Tuples);
          if (attribute_idx == activeAttributeIdx)
          {
            if (f == nullptr)
            {
              contVal = this->Tuples[currComp];
            }
          }
          ++j;
        }
        ++attribute_idx;
      }

      if (f)
      {
        contVal = f->FunctionValue(point);
      }
      this->Scalars->SetTuple1(i, contVal); // value at point i of the
      // current linear cell.
      if (i == 0)
      {
        range[0] = range[1] = contVal;
      }
      else
      {
        range[0] = range[0] < contVal ? range[0] : contVal;
        range[1] = range[1] > contVal ? range[1] : contVal;
      }

      ++i;
      locals = locals + 3;
    }

    // call contour on each value
    for (int vv = 0; vv < numContours; vv++)
    {
      if (values[vv] >= range[0] && values[vv] <= range[1])
      {
        linearCell->Contour(values[vv], this->Scalars, locator, verts, lines, polys, secondaryPd,
          outPd, secondaryCd, 0, outCd);
      }
    }

    return;
  }
  // not linear case
  internalPd->Reset();

  switch (this->GetDimension())
  {
    case 3:
      tess->Tessellate(this, attributes, this->InternalPoints, this->InternalCellArray, internalPd);
      linearCell = this->Tetra;
      ptsCount = 4;
      break;
    case 2:
      tess->Triangulate(
        this, attributes, this->InternalPoints, this->InternalCellArray, internalPd);
      linearCell = this->Triangle;
      ptsCount = 3;
      break;
    default:
      assert("TODO: dimension 1 and 0" && 0);
      return;
  }

  svtkIdType npts;
  const svtkIdType* pts = nullptr;
  double* point = this->InternalPoints->GetPointer(0);

  svtkDataArray* scalars = internalPd->GetArray(attributes->GetActiveAttribute());
  int currComp = attributes->GetActiveComponent();

  values = contourValues->GetValues();
  int numContours = contourValues->GetNumberOfContours();

  c = internalPd->GetNumberOfArrays();
  int dataIndex = 0;

  // for each linear sub-tetra, Build it and its pointdata
  // then contour it.
  for (this->InternalCellArray->InitTraversal(); this->InternalCellArray->GetNextCell(npts, pts);)
  {
    assert("check: valid number of points" && npts == ptsCount);
    range[1] = range[0] = scalars->GetComponent(dataIndex, currComp);
    secondaryPd->Reset();
    for (i = 0; i < ptsCount; i++, point += 3)
    {
      linearCell->PointIds->SetId(i, pts[i]);
      linearCell->Points->SetPoint(i, point);
      if (f)
      {
        contVal = f->FunctionValue(point);
      }
      else
      {
        contVal = scalars->GetComponent(dataIndex, currComp);
      }
      this->Scalars->SetTuple1(i, contVal); // value at point i of the
      // current linear simplex.

      range[0] = range[0] < contVal ? range[0] : contVal;
      range[1] = range[1] > contVal ? range[1] : contVal;
      // for each point-centered attribute
      j = 0;
      while (j < c)
      {
        secondaryPd->GetArray(j)->InsertTuple(pts[i], internalPd->GetArray(j)->GetTuple(dataIndex));
        ++j;
      }
      ++dataIndex;
    }
    for (int vv = 0; vv < numContours; vv++)
    {
      if (values[vv] >= range[0] && values[vv] <= range[1])
      {
        linearCell->Contour(values[vv], this->Scalars, locator, verts, lines, polys, secondaryPd,
          outPd, secondaryCd, 0, outCd);
      }
    }
  }
}

//----------------------------------------------------------------------------
void svtkGenericAdaptorCell::Clip(double value, svtkImplicitFunction* f,
  svtkGenericAttributeCollection* attributes, svtkGenericCellTessellator* tess, int insideOut,
  svtkIncrementalPointLocator* locator, svtkCellArray* connectivity, svtkPointData* outPd,
  svtkCellData* outCd, svtkPointData* internalPd, svtkPointData* secondaryPd, svtkCellData* secondaryCd)
{
  assert("pre: attributes_exist" && attributes != nullptr);
  assert("pre: tessellator_exists" && tess != nullptr);
  assert("pre: locator_exists" && locator != nullptr);
  assert("pre: connectivity_exist" && connectivity != nullptr);
  assert("pre: internalPd_exists" && internalPd != nullptr);
  assert("pre: secondaryPd_exists" && secondaryPd != nullptr);
  assert("pre: secondaryCd_exists" && secondaryCd != nullptr);

  int i;
  int j;
  int c;
  double contVal = -1000;
  double* values;

  svtkCell* linearCell = nullptr;
  svtkIdType ptsCount = 0;

  this->Reset();

  // for each cell-centered attribute: copy the value in the secondary
  // cell data.
  secondaryCd->Reset();
  int attrib = 0;
  while (attrib < attributes->GetNumberOfAttributes())
  {
    if (attributes->GetAttribute(attrib)->GetCentering() == svtkCellCentered)
    {
      svtkDataArray* array = secondaryCd->GetArray(attributes->GetAttribute(attrib)->GetName());
      values = attributes->GetAttribute(attrib)->GetTuple(this);
      array->InsertNextTuple(values);
    }
    attrib++;
  }

  int attribute = this->GetHighestOrderAttribute(attributes);
  if (this->IsGeometryLinear() &&
    (attribute == -1 || this->IsAttributeLinear(attributes->GetAttribute(attribute))))
  {
    // linear case
    switch (this->GetType())
    {
      case SVTK_HIGHER_ORDER_TRIANGLE:
        linearCell = this->Triangle;
        ptsCount = 3;
        break;
      case SVTK_HIGHER_ORDER_QUAD:
        linearCell = this->Quad;
        ptsCount = 4;
        break;
      case SVTK_HIGHER_ORDER_TETRAHEDRON:
        linearCell = this->Tetra;
        ptsCount = 4;
        break;
      case SVTK_HIGHER_ORDER_HEXAHEDRON:
        linearCell = this->Hexa;
        ptsCount = 8;
        break;
      case SVTK_HIGHER_ORDER_WEDGE:
        linearCell = this->Wedge;
        ptsCount = 6;
        break;
      case SVTK_HIGHER_ORDER_PYRAMID:
        linearCell = this->Pyramid;
        ptsCount = 5;
        break;
      default:
        assert("check: impossible case" && 0);
        return;
    }
    int currComp = attributes->GetActiveComponent();

    double* locals = this->GetParametricCoords();
    double point[3];

    svtkGenericAttribute* a;
    int count = attributes->GetNumberOfAttributes();
    int attribute_idx;

    this->AllocateTuples(attributes->GetMaxNumberOfComponents());

    int activeAttributeIdx = attributes->GetActiveAttribute();

    // build the cell
    i = 0;
    while (i < ptsCount)
    {
      this->EvaluateLocation(0, locals, point);
      linearCell->PointIds->SetId(i, i);
      linearCell->Points->SetPoint(i, point);

      // for each point-centered attribute
      secondaryPd->Reset();
      attribute_idx = 0;
      j = 0;
      while (attribute_idx < count)
      {
        a = attributes->GetAttribute(attribute_idx);
        if (a->GetCentering() == svtkPointCentered)
        {
          this->InterpolateTuple(a, locals, this->Tuples);
          secondaryPd->GetArray(j)->InsertTuple(i, this->Tuples);
          if (attribute_idx == activeAttributeIdx)
          {
            if (f == nullptr)
            {
              contVal = this->Tuples[currComp];
            }
          }
          ++j;
        }
        ++attribute_idx;
      }

      if (f)
      {
        contVal = f->FunctionValue(point);
      }
      this->Scalars->SetTuple1(i, contVal); // value at point i of the
      // current linear cell.

      ++i;
      locals = locals + 3;
    }

    linearCell->Clip(value, this->Scalars, locator, connectivity, secondaryPd, outPd, secondaryCd,
      0, outCd, insideOut);
    return;
  }

  // Not linear case
  internalPd->Reset();

  switch (this->GetDimension())
  {
    case 3:
      tess->Tessellate(this, attributes, this->InternalPoints, this->InternalCellArray, internalPd);
      linearCell = this->Tetra;
      ptsCount = 4;
      break;
    case 2:
      tess->Triangulate(
        this, attributes, this->InternalPoints, this->InternalCellArray, internalPd);
      linearCell = this->Triangle;
      ptsCount = 3;
      break;
    default:
      assert("TODO: dimension 1 and 0" && 0);
      return;
  }

  svtkIdType npts;
  const svtkIdType* pts = nullptr;
  double* point = this->InternalPoints->GetPointer(0);

  svtkDataArray* scalars = internalPd->GetArray(attributes->GetActiveAttribute());
  int currComp = attributes->GetActiveComponent();

  c = internalPd->GetNumberOfArrays();
  int dataIndex = 0;

  // for each linear sub-tetra, Build it and its pointdata
  // then contour it.
  for (this->InternalCellArray->InitTraversal(); this->InternalCellArray->GetNextCell(npts, pts);)
  {
    assert("check: valid number of points" && npts == ptsCount);
    secondaryPd->Reset();
    for (i = 0; i < ptsCount; i++, point += 3)
    {
      linearCell->PointIds->SetId(i, pts[i]);
      linearCell->Points->SetPoint(i, point);
      if (f)
      {
        contVal = f->FunctionValue(point);
      }
      else
      {
        contVal = scalars->GetComponent(dataIndex, currComp);
      }
      this->Scalars->SetTuple1(i, contVal); // value at point i of the
      // current linear simplex.

      // for each point-centered attribute
      j = 0;
      while (j < c)
      {
        secondaryPd->GetArray(j)->InsertTuple(pts[i], internalPd->GetArray(j)->GetTuple(dataIndex));
        ++j;
      }
      ++dataIndex;
    }
    linearCell->Clip(value, this->Scalars, locator, connectivity, secondaryPd, outPd, secondaryCd,
      0, outCd, insideOut);
  }
}

//----------------------------------------------------------------------------
// Description:
// Tessellate the cell if it is not linear or if at least one attribute of
// `attributes' is not linear. The output are linear cells of the same
// dimension than than cell. If the cell is linear and all attributes are
// linear, the output is just a copy of the current cell.
// `points', `cellArray', `pd' and `cd' are cumulative output data arrays
// over cell iterations: they store the result of each call to Tessellate().
// If it is not null, `types' is fill with the types of the linear cells.
// `types' is null when it is called from svtkGenericGeometryFilter and not
// null when it is called from svtkGenericDatasetTessellator.
// \pre attributes_exist: attributes!=0
// \pre points_exist: points!=0
// \pre cellArray_exists: cellArray!=0
// // \pre scalars_exists: scalars!=0
// \pre pd_exist: pd!=0
// \pre cd_exists: cd!=0
void svtkGenericAdaptorCell::Tessellate(svtkGenericAttributeCollection* attributes,
  svtkGenericCellTessellator* tess, svtkPoints* points, svtkIncrementalPointLocator* locator,
  svtkCellArray* cellArray, svtkPointData* internalPd, svtkPointData* pd, svtkCellData* cd,
  svtkUnsignedCharArray* types)
{
  assert("pre: attributes_exist" && attributes != nullptr);
  assert("pre: tessellator_exists" && tess != nullptr);
  assert("pre: points_exist" && points != nullptr);
  assert("pre: cellArray_exists" && cellArray != nullptr);
  assert("pre: internalPd_exists" && internalPd != nullptr);
  assert("pre: pd_exist" && pd != nullptr);
  assert("pre: cd_exist" && cd != nullptr);

  int i;
  int j;
#ifndef NDEBUG
  svtkIdType valid_npts = 0; // for the check assertion
#endif
  int linearCellType = SVTK_EMPTY_CELL;
  this->Reset();

  assert("check: TODO: Tessellate only works with 2D and 3D cells" &&
    (this->GetDimension() == 3 || this->GetDimension() == 2));

  int attribute = this->GetHighestOrderAttribute(attributes);
  if (this->IsGeometryLinear() &&
    (attribute == -1 || this->IsAttributeLinear(attributes->GetAttribute(attribute))))
  {
    // linear case
    this->AllocateTuples(attributes->GetMaxNumberOfComponents());

    // for each cell-centered attribute: copy the value
    int attrib = 0;
    while (attrib < attributes->GetNumberOfAttributes())
    {
      if (attributes->GetAttribute(attrib)->GetCentering() == svtkCellCentered)
      {
        svtkDataArray* array = cd->GetArray(attributes->GetAttribute(attrib)->GetName());
        double* values = attributes->GetAttribute(attrib)->GetTuple(this);
        array->InsertNextTuple(values);
      }
      attrib++;
    }
    int numVerts = 0;

    switch (this->GetType())
    {
      case SVTK_HIGHER_ORDER_TRIANGLE:
        linearCellType = SVTK_TRIANGLE;
        numVerts = 3;
        break;
      case SVTK_HIGHER_ORDER_QUAD:
        linearCellType = SVTK_QUAD;
        numVerts = 4;
        break;
      case SVTK_HIGHER_ORDER_TETRAHEDRON:
        linearCellType = SVTK_TETRA;
        numVerts = 4;
        break;
      case SVTK_HIGHER_ORDER_HEXAHEDRON:
        linearCellType = SVTK_HEXAHEDRON;
        numVerts = 8;
        break;
      case SVTK_HIGHER_ORDER_WEDGE:
        linearCellType = SVTK_WEDGE;
        numVerts = 6;
        break;
      case SVTK_HIGHER_ORDER_PYRAMID:
        linearCellType = SVTK_PYRAMID;
        numVerts = 5;
        break;
      default:
        assert("check: impossible case" && 0);
        return;
    }
    double* locals = this->GetParametricCoords();
    this->InternalIds->Reset();
    double point[3];
    svtkIdType ptId;
    i = 0;

    svtkGenericAttribute* a;
    int count = attributes->GetNumberOfAttributes();
    int attribute_idx;
    int newpoint = 1;

    while (i < numVerts)
    {
      this->EvaluateLocation(0, locals, point);
      if (locator == nullptr) // no merging
      {
        ptId = points->InsertNextPoint(point);
      }
      else // merging
      {
        newpoint = locator->InsertUniquePoint(point, ptId);
      }
      this->InternalIds->InsertId(i, ptId);
      if (newpoint)
      {
        // for each point-centered attribute
        attribute_idx = 0;
        j = 0;
        while (attribute_idx < count)
        {
          a = attributes->GetAttribute(attribute_idx);
          if (a->GetCentering() == svtkPointCentered)
          {
            this->InterpolateTuple(a, locals, this->Tuples);
            pd->GetArray(j)->InsertTuple(ptId, this->Tuples);
            ++j;
          }
          ++attribute_idx;
        }
      }
      ++i;
      locals = locals + 3;
    }

    cellArray->InsertNextCell(this->InternalIds);
    if (types != nullptr)
    {
      types->InsertNextValue(linearCellType);
    }
  }
  else // not linear
  {
    if (this->GetDimension() == 3)
    {
      internalPd->Reset();
      tess->Tessellate(this, attributes, this->InternalPoints, this->InternalCellArray, internalPd);
      linearCellType = SVTK_TETRA;
#ifndef NDEBUG
      valid_npts = 4;
#endif
    }
    else
    {
      if (this->GetDimension() == 2)
      {
        internalPd->Reset();
        tess->Triangulate(
          this, attributes, this->InternalPoints, this->InternalCellArray, internalPd);
        linearCellType = SVTK_TRIANGLE;
#ifndef NDEBUG
        valid_npts = 3;
#endif
      }
      else
      {
        linearCellType = 0; // for compiler warning
      }
    }

    svtkIdType npts = 0;
    const svtkIdType* pts = nullptr;
    double* point = this->InternalPoints->GetPointer(0);

    // for each cell-centered attribute: copy the value
    int c = this->InternalCellArray->GetNumberOfCells();
    int attrib = 0;
    while (attrib < attributes->GetNumberOfAttributes())
    {
      if (attributes->GetAttribute(attrib)->GetCentering() == svtkCellCentered)
      {
        svtkDataArray* array = cd->GetArray(attributes->GetAttribute(attrib)->GetName());
        double* values = attributes->GetAttribute(attrib)->GetTuple(this);
        i = 0;
        while (i < c)
        {
          array->InsertNextTuple(values);
          ++i;
        }
      }
      attrib++;
    }

    c = internalPd->GetNumberOfArrays(); // same as pd->GetNumberOfArrays();

    int dataIndex = 0;
    int newpoint = 1;

    for (this->InternalCellArray->InitTraversal(); this->InternalCellArray->GetNextCell(npts, pts);)
    {
      assert("check: is_a_simplex" && npts == valid_npts);
      this->InternalIds->Reset();

      for (i = 0; i < npts; i++, point += 3)
      {
        svtkIdType ptId;
        if (locator == nullptr) // no merging
        {
          ptId = points->InsertNextPoint(point);
        }
        else // merging
        {
          newpoint = locator->InsertUniquePoint(point, ptId);
        }
        this->InternalIds->InsertId(i, ptId);
        if (newpoint)
        {
          // for each point-centered attribute
          j = 0;
          while (j < c)
          {
            pd->GetArray(j)->InsertTuple(ptId, internalPd->GetArray(j)->GetTuple(dataIndex));
            ++j;
          }
        }
        ++dataIndex;
      }
      cellArray->InsertNextCell(this->InternalIds);
      if (types != nullptr)
      {
        types->InsertNextValue(linearCellType);
      }
    }
  }
}

//----------------------------------------------------------------------------
void svtkGenericAdaptorCell::TriangulateFace(svtkGenericAttributeCollection* attributes,
  svtkGenericCellTessellator* tess, int index, svtkPoints* points,
  svtkIncrementalPointLocator* locator, svtkCellArray* cellArray, svtkPointData* internalPd,
  svtkPointData* pd, svtkCellData* cd)
{
  assert("pre: cell_is_3d" && this->GetDimension() == 3);
  assert("pre: attributes_exist" && attributes != nullptr);
  assert("pre: tessellator_exists" && tess != nullptr);
  assert("pre: valid_face" && index >= 0 && index < this->GetNumberOfBoundaries(2));
  assert("pre: points_exist" && points != nullptr);
  assert("pre: cellArray_exists" && cellArray != nullptr);
  assert("pre: internalPd_exists" && internalPd != nullptr);
  assert("pre: pd_exist" && pd != nullptr);
  assert("pre: cd_exists" && cd != nullptr);

  int i;
  int j;
  int c;

  this->Reset();

  internalPd->Reset();

  // if simplex (tetra) just one sub-tetra [0,1,2,3]
  // otherwise build sub-tetra: HOW?

  int attribute = this->GetHighestOrderAttribute(attributes);
  if (this->IsGeometryLinear() &&
    (attribute == -1 || this->IsAttributeLinear(attributes->GetAttribute(attribute))))
  {
    // LINEAR CASE
    // the cell is linear both in geometry and attributes
    // just create a linear cell of the same type and return
    // the relevant face: basically, do what the Graphics/svtkGeometryFilter
    // do with the linear cells
    this->AllocateTuples(attributes->GetMaxNumberOfComponents());

    // for each cell-centered attribute: copy the value
    int attrib = 0;
    while (attrib < attributes->GetNumberOfAttributes())
    {
      if (attributes->GetAttribute(attrib)->GetCentering() == svtkCellCentered)
      {
        svtkDataArray* array = cd->GetArray(attributes->GetAttribute(attrib)->GetName());
        double* values = attributes->GetAttribute(attrib)->GetTuple(this);
        array->InsertNextTuple(values);
      }
      attrib++;
    }
    svtkGenericAttribute* a;
    int count = attributes->GetNumberOfAttributes();

    int attribute_idx;

    this->InternalIds->Reset();
    const svtkIdType* faceVerts = this->GetFaceArray(index);
    int numVerts = this->GetNumberOfVerticesOnFace(index);
    double* locals = this->GetParametricCoords();
    double* local;
    double point[3];
    svtkIdType ptId;

    i = 0;
    int newpoint = 1;
    while (i < numVerts)
    {
      local = locals + 3 * faceVerts[i];
      this->EvaluateLocation(0, local, point);
      if (locator == nullptr) // no merging
      {
        ptId = points->InsertNextPoint(point);
      }
      else // merging
      {
        newpoint = locator->InsertUniquePoint(point, ptId);
      }
      this->InternalIds->InsertId(i, ptId);
      if (newpoint)
      {
        // for each point-centered attribute
        attribute_idx = 0;
        j = 0;
        while (attribute_idx < count)
        {
          a = attributes->GetAttribute(attribute_idx);
          if (a->GetCentering() == svtkPointCentered)
          {
            this->InterpolateTuple(a, local, this->Tuples);
            pd->GetArray(j)->InsertTuple(ptId, this->Tuples);
            ++j;
          }
          ++attribute_idx;
        }
      }
      ++i;
    }
    cellArray->InsertNextCell(this->InternalIds);
    return;
  }

  // NOT LINEAR
  tess->TessellateFace(
    this, attributes, index, this->InternalPoints, this->InternalCellArray, internalPd);

  svtkIdType npts = 0;
  const svtkIdType* pts = nullptr;
  double* point = this->InternalPoints->GetPointer(0);

  // for each cell-centered attribute: copy the value
  c = this->InternalCellArray->GetNumberOfCells();
  int attrib = 0;
  while (attrib < attributes->GetNumberOfAttributes())
  {
    if (attributes->GetAttribute(attrib)->GetCentering() == svtkCellCentered)
    {
      svtkDataArray* array = cd->GetArray(attributes->GetAttribute(attrib)->GetName());
      double* values = attributes->GetAttribute(attrib)->GetTuple(this);
      i = 0;
      while (i < c)
      {
        array->InsertNextTuple(values);
        ++i;
      }
    }
    attrib++;
  }

  c = internalPd->GetNumberOfArrays();
  int dataIndex = 0;

  int newpoint = 1;

  for (this->InternalCellArray->InitTraversal(); this->InternalCellArray->GetNextCell(npts, pts);)
  {
    assert("check: is_a_triangle" && npts == 3);
    this->InternalIds->Reset();

    for (i = 0; i < npts; i++, point += 3)
    {
      svtkIdType ptId;
      if (locator == nullptr) // no merging
      {
        ptId = points->InsertNextPoint(point);
      }
      else // merging
      {
        newpoint = locator->InsertUniquePoint(point, ptId);
      }
      this->InternalIds->InsertId(i, ptId);
      if (newpoint)
      {
        // for each point-centered attribute
        j = 0;
        while (j < c)
        {
          pd->GetArray(j)->InsertTuple(ptId, internalPd->GetArray(j)->GetTuple(dataIndex));
          ++j;
        }
      }
      ++dataIndex;
    }
    cellArray->InsertNextCell(this->InternalIds);
  }
}

//----------------------------------------------------------------------------
// Description:
// Allocate some memory if Tuples does not exist or is smaller than size.
// \pre positive_size: size>0
void svtkGenericAdaptorCell::AllocateTuples(int size)
{
  assert("pre: positive_size" && size > 0);

  if (this->TuplesCapacity < size)
  {
    delete[] this->Tuples;
    this->Tuples = new double[size];
    this->TuplesCapacity = size;
  }
}
