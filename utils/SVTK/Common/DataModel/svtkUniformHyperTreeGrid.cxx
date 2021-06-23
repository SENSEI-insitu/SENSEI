/*=========================================================================

Program:   Visualization Toolkit
Module:    svtkUniformHyperTreeGrid.cxx

Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkUniformHyperTreeGrid.h"

#include "svtkDoubleArray.h"
#include "svtkGenericDataArray.h"
#include "svtkHyperTree.h"
#include "svtkHyperTreeGrid.h"
#include "svtkHyperTreeGridScales.h"

#include <deque>

svtkStandardNewMacro(svtkUniformHyperTreeGrid);

// Helper macros to quickly fetch a HT at a given index or iterator
#define GetHyperTreeFromOtherMacro(_obj_, _index_)                                                 \
  (static_cast<svtkHyperTree*>(_obj_->HyperTrees.find(_index_) != _obj_->HyperTrees.end()           \
      ? _obj_->HyperTrees[_index_]                                                                 \
      : nullptr))
#define GetHyperTreeFromThisMacro(_index_) GetHyperTreeFromOtherMacro(this, _index_)

//-----------------------------------------------------------------------------
svtkUniformHyperTreeGrid::svtkUniformHyperTreeGrid()
{
  // Default dimension
  this->Dimension = 3;

  // Default grid origin
  this->Origin[0] = 0.;
  this->Origin[1] = 0.;
  this->Origin[2] = 0.;

  // Default element sizes
  this->GridScale[0] = 1.;
  this->GridScale[1] = 1.;
  this->GridScale[2] = 1.;

  this->WithCoordinates = false;

  // Coordinates have not been computed yet
  this->ComputedXCoordinates = false;
  this->ComputedYCoordinates = false;
  this->ComputedZCoordinates = false;
}

//-----------------------------------------------------------------------------
svtkUniformHyperTreeGrid::~svtkUniformHyperTreeGrid() {}

//-----------------------------------------------------------------------------
svtkHyperTree* svtkUniformHyperTreeGrid::GetTree(svtkIdType index, bool create)
{
  // Wrap convenience macro for outside use
  svtkHyperTree* tree = GetHyperTreeFromThisMacro(index);

  // Create a new cursor if only required to do so
  if (create && !tree)
  {
    tree = svtkHyperTree::CreateInstance(this->BranchFactor, this->Dimension);
    tree->SetTreeIndex(index);
    this->HyperTrees[index] = tree;
    tree->Delete();

    // JB pour initialiser le scales au niveau de HT
    // Esperons qu'aucun HT n'est cree hors de l'appel a cette methode
    // Ce service ne devrait pas exister ou etre visible car c'est au niveau d'un HT ou d'un
    // cursor que cet appel est fait
    if (!tree->HasScales())
    {
      if (!this->Scales)
      {
        this->Scales =
          std::make_shared<svtkHyperTreeGridScales>(this->BranchFactor, this->GridScale);
      }
      tree->SetScales(this->Scales);
    }
  }

  return tree;
}

//-----------------------------------------------------------------------------
void svtkUniformHyperTreeGrid::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Origin: " << this->Origin[0] << "," << this->Origin[1] << "," << this->Origin[2]
     << endl;

  os << indent << "GridScale: " << this->GridScale[0] << "," << this->GridScale[1] << ","
     << this->GridScale[2] << endl;

  os << indent << "ComputedXCoordinates: " << this->ComputedXCoordinates << endl;
  os << indent << "ComputedYCoordinates: " << this->ComputedYCoordinates << endl;
  os << indent << "ComputedZCoordinates: " << this->ComputedZCoordinates << endl;

  os << indent << "Scales:" << this->Scales << endl;
  if (this->Scales)
  {
    for (unsigned int ilevel = 0; ilevel < this->Scales->GetCurrentFailLevel(); ++ilevel)
    {
      os << " #" << ilevel << " (" << this->Scales->GetScaleX(ilevel) << " ,"
         << this->Scales->GetScaleY(ilevel) << " ," << this->Scales->GetScaleZ(ilevel) << ")";
    }
  }
}

//-----------------------------------------------------------------------------
void svtkUniformHyperTreeGrid::SetGridScale(double h0, double h1, double h2)
{
  // Call superclass
  this->GridScale[0] = h0;
  this->GridScale[1] = h1;
  this->GridScale[2] = h2;

  // Update modification time
  this->Modified();
}

//-----------------------------------------------------------------------------
void svtkUniformHyperTreeGrid::SetGridScale(double* h)
{
  // No range check is performed
  this->SetGridScale(h[0], h[1], h[2]);
}

//-----------------------------------------------------------------------------
void svtkUniformHyperTreeGrid::SetGridScale(double h)
{
  // Set uniform grid scale depending on dimensionality
  switch (this->Dimension)
  {
    case 1:
    {
      switch (this->GetOrientation())
      {
        case 0:
          this->SetGridScale(h, 0., 0.);
          break;
        case 1:
          this->SetGridScale(0., h, 0.);
          break;
        case 2:
          this->SetGridScale(0., 0., h);
          break;
      } // switch ( this->GetOrientation() )
      break;
    }
    case 2:
    {
      switch (this->GetOrientation())
      {
        case 0:
          this->SetGridScale(0., h, h);
          break;
        case 1:
          this->SetGridScale(h, 0., h);
          break;
        case 2:
          this->SetGridScale(h, h, 0.);
          break;
      } // switch ( this->GetOrientation() )
      break;
    }
    case 3:
    {
      this->SetGridScale(h, h, h);
      break;
    }
  }
}

//----------------------------------------------------------------------------
void svtkUniformHyperTreeGrid::SetXCoordinates(svtkDataArray* m_XCoordinates)
{
  std::cerr << "Bad to call svtkUniformHyperTreeGrid::SetXCoordinates" << std::endl;
  bool isConform = true;
  // TODO JB Verifier la conformite a un UHTG
  if (!isConform)
  {
    throw std::domain_error("Cannot use SetXCoordinates on UniformHyperTreeGrid");
  }
  this->Origin[0] = m_XCoordinates->GetTuple1(0);
  this->GridScale[0] =
    (m_XCoordinates->GetTuple1(m_XCoordinates->GetNumberOfTuples() - 1) - this->Origin[0]) /
    (m_XCoordinates->GetNumberOfTuples() - 1);
}

svtkDataArray* svtkUniformHyperTreeGrid::GetXCoordinates()
{
  std::cerr << "Bad to call svtkUniformHyperTreeGrid::GetXCoordinates" << std::endl;
  // Compute coordinates only if necessary
  if (!this->ComputedXCoordinates)
  {
    // Compute and store actual coordinates
    unsigned int np = this->GetDimensions()[0];
    this->XCoordinates->SetNumberOfTuples(np);
    double x = this->Origin[0];
    for (unsigned int i = 0; i < np; ++i, x += this->GridScale[0])
    {
      this->XCoordinates->SetTuple1(i, x);
    }

    // Keep track that computation was done
    this->ComputedXCoordinates = true;
  } // if ( ! this->ComputedCoordinates )

  // Return coordinates
  return this->XCoordinates;
}

//----------------------------------------------------------------------------
void svtkUniformHyperTreeGrid::SetYCoordinates(svtkDataArray* m_YCoordinates)
{
  std::cerr << "Bad to call svtkUniformHyperTreeGrid::SetYCoordinates" << std::endl;
  bool isConform = true;
  // TODO JB Verifier la conformite a un UHTG
  if (!isConform)
  {
    throw std::domain_error("Cannot use SetYCoordinates on UniformHyperTreeGrid");
  }
  this->Origin[1] = m_YCoordinates->GetTuple1(0);
  this->GridScale[1] =
    (m_YCoordinates->GetTuple1(m_YCoordinates->GetNumberOfTuples() - 1) - this->Origin[1]) /
    (m_YCoordinates->GetNumberOfTuples() - 1);
}

svtkDataArray* svtkUniformHyperTreeGrid::GetYCoordinates()
{
  std::cerr << "Bad to call svtkUniformHyperTreeGrid::GetYCoordinates" << std::endl;
  // Compute coordinates only if necessary
  if (!this->ComputedYCoordinates)
  {
    // Compute and store actual coordinates
    unsigned int np = this->GetDimensions()[1];
    this->YCoordinates->SetNumberOfTuples(np);
    double y = this->Origin[1];
    for (unsigned int i = 0; i < np; ++i, y += this->GridScale[1])
    {
      this->YCoordinates->SetTuple1(i, y);
    }

    // Keep track that computation was done
    this->ComputedYCoordinates = true;
  } // if ( ! this->ComputedCoordinates )

  // Return coordinates
  return this->YCoordinates;
}

//----------------------------------------------------------------------------
void svtkUniformHyperTreeGrid::SetZCoordinates(svtkDataArray* m_ZCoordinates)
{
  std::cerr << "Bad to call svtkUniformHyperTreeGrid::SetZCoordinates" << std::endl;
  bool isConform = true;
  // TODO JB Verifier la conformite a un UHTG
  if (!isConform)
  {
    throw std::domain_error("Cannot use SetZCoordinates on UniformHyperTreeGrid");
  }
  this->Origin[2] = m_ZCoordinates->GetTuple1(0);
  this->GridScale[2] =
    (m_ZCoordinates->GetTuple1(m_ZCoordinates->GetNumberOfTuples() - 1) - this->Origin[2]) /
    (m_ZCoordinates->GetNumberOfTuples() - 1);
};

svtkDataArray* svtkUniformHyperTreeGrid::GetZCoordinates()
{
  std::cerr << "Bad to call svtkUniformHyperTreeGrid::GetZCoordinates" << std::endl;
  // Compute coordinates only if necessary
  if (!this->ComputedZCoordinates)
  {
    // Compute and store actual coordinates
    unsigned int np = this->GetDimensions()[2];
    this->ZCoordinates->SetNumberOfTuples(np);
    double z = this->Origin[2];
    for (unsigned int i = 0; i < np; ++i, z += this->GridScale[2])
    {
      this->ZCoordinates->SetTuple1(i, z);
    }

    // Keep track that computation was done
    this->ComputedZCoordinates = true;
  } // if ( ! this->ComputedCoordinates )

  // Return coordinates
  return this->ZCoordinates;
}

void svtkUniformHyperTreeGrid::CopyCoordinates(const svtkHyperTreeGrid* outputHTG)
{
  svtkUniformHyperTreeGrid* output =
    svtkUniformHyperTreeGrid::SafeDownCast(const_cast<svtkHyperTreeGrid*>(outputHTG));
  this->SetOrigin(output->GetOrigin());
  this->SetGridScale(output->GetGridScale());
}

void svtkUniformHyperTreeGrid::SetFixedCoordinates(unsigned int axis, double value)
{
  assert("pre: invalid_axis" && axis < 3);
  this->GridScale[axis] = value;
}

//-----------------------------------------------------------------------------
void svtkUniformHyperTreeGrid::GetLevelZeroOriginAndSizeFromIndex(
  svtkIdType treeindex, double* m_Origin, double* m_Size)
{
  // Retrieve Cartesian coordinates of root tree
  unsigned int i, j, k;
  this->GetLevelZeroCoordinatesFromIndex(treeindex, i, j, k);

  // Compute origin of the cursor
  const double* origin = this->GetOrigin();
  const double* scale = this->GetGridScale();
  m_Origin[0] = origin[0] + i * scale[0];
  m_Origin[1] = origin[1] + j * scale[1];
  m_Origin[2] = origin[2] + k * scale[2];
  m_Size[0] = scale[0];
  m_Size[1] = scale[1];
  m_Size[2] = scale[2];
}

//-----------------------------------------------------------------------------
void svtkUniformHyperTreeGrid::Initialize()
{
  this->Superclass::Initialize();
  // Default dimension
  this->Dimension = 3;

  // Default grid origin
  this->Origin[0] = 0.;
  this->Origin[1] = 0.;
  this->Origin[2] = 0.;

  // Default element sizes
  this->GridScale[0] = 1.;
  this->GridScale[1] = 1.;
  this->GridScale[2] = 1.;

  this->WithCoordinates = false;

  // Coordinates have not been computed yet
  this->ComputedXCoordinates = false;
  this->ComputedYCoordinates = false;
  this->ComputedZCoordinates = false;
}

//-----------------------------------------------------------------------------
void svtkUniformHyperTreeGrid::GetLevelZeroOriginFromIndex(svtkIdType treeindex, double* m_Origin)
{
  // Retrieve Cartesian coordinates of root tree
  unsigned int i, j, k;
  this->GetLevelZeroCoordinatesFromIndex(treeindex, i, j, k);

  // Compute origin of the cursor
  const double* origin = this->GetOrigin();
  const double* scale = this->GetGridScale();
  m_Origin[0] = origin[0] + i * scale[0];
  m_Origin[1] = origin[1] + j * scale[1];
  m_Origin[2] = origin[2] + k * scale[2];
}

//-----------------------------------------------------------------------------
void svtkUniformHyperTreeGrid::CopyStructure(svtkDataObject* ds)
{
  assert("pre: ds_exists" && ds != nullptr);
  svtkUniformHyperTreeGrid* uhtg = svtkUniformHyperTreeGrid::SafeDownCast(ds);
  assert("pre: same_type" && uhtg != nullptr);

  // Call superclass
  this->Superclass::CopyStructure(ds);

  // Copy uniform grid parameters
  memcpy(this->Origin, uhtg->GetOrigin(), 3 * sizeof(double));
  memcpy(this->GridScale, uhtg->GetGridScale(), 3 * sizeof(double));
}

//----------------------------------------------------------------------------
double* svtkUniformHyperTreeGrid::GetBounds()
{
  // Recompute each call
  // Compute bounds from uniform grid parameters
  for (unsigned int i = 0; i < 3; ++i)
  {
    unsigned int di = 2 * i;
    unsigned int dip = di + 1;
    this->Bounds[di] = this->Origin[i];
    if (this->GetDimensions()[i] == 1)
    {
      this->Bounds[dip] = this->Origin[i];
    }
    else
    {
      this->Bounds[dip] = this->Origin[i] + this->GetCellDims()[i] * this->GridScale[i];
    }

    // Ensure that the bounds are increasing
    if (this->Bounds[di] > this->Bounds[dip])
    {
      std::swap(this->Bounds[di], this->Bounds[dip]);
    }
  }

  return this->Bounds;
}

//-----------------------------------------------------------------------------
void svtkUniformHyperTreeGrid::ShallowCopy(svtkDataObject* src)
{
  svtkHyperTreeGrid* uhtg = svtkUniformHyperTreeGrid::SafeDownCast(src);
  assert("src_same_type" && uhtg);

  // Copy member variables
  this->CopyStructure(uhtg);

  // Call superclass
  this->Superclass::ShallowCopy(src);
}

//-----------------------------------------------------------------------------
void svtkUniformHyperTreeGrid::DeepCopy(svtkDataObject* src)
{
  svtkHyperTreeGrid* uhtg = svtkUniformHyperTreeGrid::SafeDownCast(src);
  assert("src_same_type" && uhtg);

  // Copy member variables
  this->CopyStructure(uhtg);

  // Call superclass
  this->Superclass::DeepCopy(src);
}

//----------------------------------------------------------------------------
unsigned long svtkUniformHyperTreeGrid::GetActualMemorySizeBytes()
{
  unsigned long size = 0; // in bytes

  size += (this->svtkDataObject::GetActualMemorySize() << 10);

  // Iterate over all trees in grid
  svtkHyperTreeGridIterator it;
  it.Initialize(this);
  while (svtkHyperTree* tree = it.GetNextTree())
  {
    size += tree->GetActualMemorySizeBytes();
  }

  // Size of root cells sizes
  size += 3 * sizeof(double);

  return size;
}
