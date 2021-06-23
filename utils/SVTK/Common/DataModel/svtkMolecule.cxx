/*=========================================================================

Program:   Visualization Toolkit
Module:    svtkMolecule.cxx
Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkMolecule.h"

#include "svtkAbstractElectronicData.h"
#include "svtkDataSetAttributes.h"
#include "svtkEdgeListIterator.h"
#include "svtkFloatArray.h"
#include "svtkGraphInternals.h"
#include "svtkIdTypeArray.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkMatrix3x3.h"
#include "svtkNew.h"
#include "svtkObjectFactory.h"
#include "svtkPlane.h"
#include "svtkPoints.h"
#include "svtkUnsignedCharArray.h"
#include "svtkUnsignedShortArray.h"
#include "svtkVector.h"
#include "svtkVectorOperators.h"

#include <cassert>

//----------------------------------------------------------------------------
svtkStandardNewMacro(svtkMolecule);

//----------------------------------------------------------------------------
svtkMolecule::svtkMolecule()
  : ElectronicData(nullptr)
  , Lattice(nullptr)
  , LatticeOrigin(0., 0., 0.)
  , AtomGhostArray(nullptr)
  , BondGhostArray(nullptr)
  , AtomicNumberArrayName(nullptr)
  , BondOrdersArrayName(nullptr)
{
  this->Initialize();
}

//----------------------------------------------------------------------------
void svtkMolecule::Initialize()
{
  // Reset underlying data structure
  this->Superclass::Initialize();

  // Setup vertex data
  svtkDataSetAttributes* vertData = this->GetVertexData();
  vertData->AllocateArrays(1); // atomic nums

  // Atomic numbers
  this->SetAtomicNumberArrayName("Atomic Numbers");
  svtkNew<svtkUnsignedShortArray> atomicNums;
  atomicNums->SetNumberOfComponents(1);
  atomicNums->SetName(this->GetAtomicNumberArrayName());
  vertData->SetScalars(atomicNums);

  // Nuclear coordinates
  svtkPoints* points = svtkPoints::New();
  this->SetPoints(points);
  points->Delete();

  // Setup edge data
  svtkDataSetAttributes* edgeData = this->GetEdgeData();
  edgeData->AllocateArrays(1); // Bond orders

  this->SetBondOrdersArrayName("Bond Orders");
  svtkNew<svtkUnsignedShortArray> bondOrders;
  bondOrders->SetNumberOfComponents(1);
  bondOrders->SetName(this->GetBondOrdersArrayName());
  edgeData->SetScalars(bondOrders);

  this->UpdateBondList();

  // Electronic data
  this->SetElectronicData(nullptr);

  this->Modified();
}

//----------------------------------------------------------------------------
svtkMolecule::~svtkMolecule()
{
  this->SetElectronicData(nullptr);
  delete[] this->AtomicNumberArrayName;
  delete[] this->BondOrdersArrayName;
}

//----------------------------------------------------------------------------
void svtkMolecule::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  svtkIndent subIndent = indent.GetNextIndent();

  os << indent << "Atoms:\n";
  for (svtkIdType i = 0; i < this->GetNumberOfAtoms(); ++i)
  {
    this->GetAtom(i).PrintSelf(os, subIndent);
  }

  os << indent << "Bonds:\n";
  for (svtkIdType i = 0; i < this->GetNumberOfBonds(); ++i)
  {
    os << subIndent << "===== Bond " << i << ": =====\n";
    this->GetBond(i).PrintSelf(os, subIndent);
  }

  os << indent << "Lattice:\n";
  if (this->HasLattice())
  {
    double* m = this->Lattice->GetData();
    os << subIndent << "a: " << m[0] << " " << m[3] << " " << m[6] << "\n";
    os << subIndent << "b: " << m[1] << " " << m[4] << " " << m[7] << "\n";
    os << subIndent << "c: " << m[2] << " " << m[5] << " " << m[8] << "\n";
    os << subIndent << "origin: " << this->LatticeOrigin[0] << " " << this->LatticeOrigin[1] << " "
       << this->LatticeOrigin[2] << "\n";
  }

  os << indent << "Electronic Data:\n";
  if (this->ElectronicData)
  {
    this->ElectronicData->PrintSelf(os, subIndent);
  }
  else
  {
    os << subIndent << "Not set.\n";
  }

  os << indent << "Atomic number array name : " << this->GetAtomicNumberArrayName() << "\n";
  os << indent << "Bond orders array name : " << this->GetBondOrdersArrayName();
}

//----------------------------------------------------------------------------
svtkAtom svtkMolecule::AppendAtom(unsigned short atomicNumber, double x, double y, double z)
{
  svtkUnsignedShortArray* atomicNums = this->GetAtomicNumberArray();

  assert(atomicNums);

  svtkIdType id;
  this->AddVertexInternal(nullptr, &id);

  atomicNums->InsertValue(id, atomicNumber);
  svtkIdType coordID = this->Points->InsertNextPoint(x, y, z);
  (void)coordID;
  assert("point ids synced with vertex ids" && coordID == id);

  this->Modified();
  return svtkAtom(this, id);
}

//----------------------------------------------------------------------------
svtkAtom svtkMolecule::GetAtom(svtkIdType atomId)
{
  assert(atomId >= 0 && atomId < this->GetNumberOfAtoms());

  svtkAtom atom(this, atomId);
  return atom;
}

//----------------------------------------------------------------------------
unsigned short svtkMolecule::GetAtomAtomicNumber(svtkIdType id)
{
  assert(id >= 0 && id < this->GetNumberOfAtoms());

  svtkUnsignedShortArray* atomicNums = this->GetAtomicNumberArray();

  return atomicNums->GetValue(id);
}

//----------------------------------------------------------------------------
void svtkMolecule::SetAtomAtomicNumber(svtkIdType id, unsigned short atomicNum)
{
  assert(id >= 0 && id < this->GetNumberOfAtoms());

  svtkUnsignedShortArray* atomicNums = this->GetAtomicNumberArray();

  atomicNums->SetValue(id, atomicNum);
  this->Modified();
}

//----------------------------------------------------------------------------
void svtkMolecule::SetAtomPosition(svtkIdType id, const svtkVector3f& pos)
{
  assert(id >= 0 && id < this->GetNumberOfAtoms());
  this->Points->SetPoint(id, pos.GetData());
  this->Modified();
}

//----------------------------------------------------------------------------
void svtkMolecule::SetAtomPosition(svtkIdType id, double x, double y, double z)
{
  assert(id >= 0 && id < this->GetNumberOfAtoms());
  this->Points->SetPoint(id, x, y, z);
  this->Modified();
}

//----------------------------------------------------------------------------
svtkVector3f svtkMolecule::GetAtomPosition(svtkIdType id)
{
  assert(id >= 0 && id < this->GetNumberOfAtoms());
  svtkFloatArray* positions = svtkArrayDownCast<svtkFloatArray>(this->Points->GetData());
  assert(positions != nullptr);
  float* data = positions->GetPointer(id * 3);
  return svtkVector3f(data);
}

//----------------------------------------------------------------------------
void svtkMolecule::GetAtomPosition(svtkIdType id, float pos[3])
{
  svtkVector3f position = this->GetAtomPosition(id);
  pos[0] = position.GetX();
  pos[1] = position.GetY();
  pos[2] = position.GetZ();
}

//----------------------------------------------------------------------------
void svtkMolecule::GetAtomPosition(svtkIdType id, double pos[3])
{
  this->Points->GetPoint(id, pos);
}

//----------------------------------------------------------------------------
svtkIdType svtkMolecule::GetNumberOfAtoms()
{
  return this->GetNumberOfVertices();
}

//----------------------------------------------------------------------------
svtkBond svtkMolecule::AppendBond(
  const svtkIdType atom1, const svtkIdType atom2, const unsigned short order)
{
  svtkUnsignedShortArray* bondOrders = this->GetBondOrdersArray();

  assert(bondOrders);

  svtkEdgeType edgeType;
  this->AddEdgeInternal(atom1, atom2, false, nullptr, &edgeType);
  this->SetBondListDirty();

  svtkIdType id = edgeType.Id;
  bondOrders->InsertValue(id, order);
  this->Modified();
  return svtkBond(this, id, atom1, atom2);
}

//----------------------------------------------------------------------------
svtkBond svtkMolecule::GetBond(svtkIdType bondId)
{
  assert(bondId >= 0 && bondId < this->GetNumberOfBonds());

  svtkIdTypeArray* bonds = this->GetBondList();
  // An array with two components holding the bonded atom's ids
  svtkIdType* ids = bonds->GetPointer(2 * bondId);
  return svtkBond(this, bondId, ids[0], ids[1]);
}

//----------------------------------------------------------------------------
void svtkMolecule::SetBondOrder(svtkIdType bondId, unsigned short order)
{
  assert(bondId >= 0 && bondId < this->GetNumberOfBonds());

  svtkUnsignedShortArray* bondOrders = this->GetBondOrdersArray();
  assert(bondOrders);

  this->Modified();
  bondOrders->InsertValue(bondId, order);
}

//----------------------------------------------------------------------------
unsigned short svtkMolecule::GetBondOrder(svtkIdType bondId)
{
  assert(bondId >= 0 && bondId < this->GetNumberOfBonds());

  svtkUnsignedShortArray* bondOrders = this->GetBondOrdersArray();

  return bondOrders ? bondOrders->GetValue(bondId) : 0;
}

//----------------------------------------------------------------------------
double svtkMolecule::GetBondLength(svtkIdType bondId)
{
  assert(bondId >= 0 && bondId < this->GetNumberOfBonds());

  // Get list of bonds
  svtkIdTypeArray* bonds = this->GetBondList();
  // An array of length two holding the bonded atom's ids
  svtkIdType* ids = bonds->GetPointer(bondId);

  // Get positions
  svtkVector3f pos1 = this->GetAtomPosition(ids[0]);
  svtkVector3f pos2 = this->GetAtomPosition(ids[1]);

  return (pos2 - pos1).Norm();
}

//----------------------------------------------------------------------------
svtkPoints* svtkMolecule::GetAtomicPositionArray()
{
  return this->Points;
}

//----------------------------------------------------------------------------
svtkUnsignedShortArray* svtkMolecule::GetAtomicNumberArray()
{
  svtkUnsignedShortArray* atomicNums = svtkArrayDownCast<svtkUnsignedShortArray>(
    this->GetVertexData()->GetScalars(this->GetAtomicNumberArrayName()));

  assert(atomicNums);

  return atomicNums;
}

//----------------------------------------------------------------------------
svtkUnsignedShortArray* svtkMolecule::GetBondOrdersArray()
{
  return svtkArrayDownCast<svtkUnsignedShortArray>(
    this->GetBondData()->GetScalars(this->GetBondOrdersArrayName()));
}

//----------------------------------------------------------------------------
svtkIdType svtkMolecule::GetNumberOfBonds()
{
  return this->GetNumberOfEdges();
}

//----------------------------------------------------------------------------
svtkCxxSetObjectMacro(svtkMolecule, ElectronicData, svtkAbstractElectronicData);

//----------------------------------------------------------------------------
void svtkMolecule::ShallowCopy(svtkDataObject* obj)
{
  svtkMolecule* m = svtkMolecule::SafeDownCast(obj);
  if (!m)
  {
    svtkErrorMacro("Can only shallow copy from svtkMolecule or subclass.");
    return;
  }
  this->ShallowCopyStructure(m);
  this->ShallowCopyAttributes(m);
}

//----------------------------------------------------------------------------
void svtkMolecule::DeepCopy(svtkDataObject* obj)
{
  svtkMolecule* m = svtkMolecule::SafeDownCast(obj);
  if (!m)
  {
    svtkErrorMacro("Can only deep copy from svtkMolecule or subclass.");
    return;
  }
  this->DeepCopyStructure(m);
  this->DeepCopyAttributes(m);
}

//----------------------------------------------------------------------------
bool svtkMolecule::CheckedShallowCopy(svtkGraph* g)
{
  bool result = this->Superclass::CheckedShallowCopy(g);
  this->BondListIsDirty = true;
  return result;
}

//----------------------------------------------------------------------------
bool svtkMolecule::CheckedDeepCopy(svtkGraph* g)
{
  bool result = this->Superclass::CheckedDeepCopy(g);
  this->BondListIsDirty = true;
  return result;
}

//----------------------------------------------------------------------------
void svtkMolecule::ShallowCopyStructure(svtkMolecule* m)
{
  this->CopyStructureInternal(m, false);
}

//----------------------------------------------------------------------------
void svtkMolecule::DeepCopyStructure(svtkMolecule* m)
{
  this->CopyStructureInternal(m, true);
}

//----------------------------------------------------------------------------
void svtkMolecule::ShallowCopyAttributes(svtkMolecule* m)
{
  this->CopyAttributesInternal(m, false);
}

//----------------------------------------------------------------------------
void svtkMolecule::DeepCopyAttributes(svtkMolecule* m)
{
  this->CopyAttributesInternal(m, true);
}

//----------------------------------------------------------------------------
void svtkMolecule::CopyStructureInternal(svtkMolecule* m, bool deep)
{
  // Call superclass
  if (deep)
  {
    this->Superclass::DeepCopy(m);
  }
  else
  {
    this->Superclass::ShallowCopy(m);
  }

  if (!m->HasLattice())
  {
    this->ClearLattice();
  }
  else
  {
    if (deep)
    {
      svtkNew<svtkMatrix3x3> newLattice;
      newLattice->DeepCopy(m->Lattice);
      this->SetLattice(newLattice);
    }
    else
    {
      this->SetLattice(m->Lattice);
    }
    this->LatticeOrigin = m->LatticeOrigin;
  }

  this->BondListIsDirty = true;
}

//----------------------------------------------------------------------------
void svtkMolecule::CopyAttributesInternal(svtkMolecule* m, bool deep)
{
  if (deep)
  {
    if (m->ElectronicData)
      this->ElectronicData->DeepCopy(m->ElectronicData);
  }
  else
  {
    this->SetElectronicData(m->ElectronicData);
  }
}

//----------------------------------------------------------------------------
void svtkMolecule::UpdateBondList()
{
  this->BuildEdgeList();
  this->BondListIsDirty = false;
}

//----------------------------------------------------------------------------
svtkIdTypeArray* svtkMolecule::GetBondList()
{
  // Create the edge list if it doesn't exist, or is marked as dirty.
  svtkIdTypeArray* edgeList = this->BondListIsDirty ? nullptr : this->GetEdgeList();
  if (!edgeList)
  {
    this->UpdateBondList();
    edgeList = this->GetEdgeList();
  }
  assert(edgeList != nullptr);
  return edgeList;
}

//----------------------------------------------------------------------------
bool svtkMolecule::GetPlaneFromBond(const svtkBond& bond, const svtkVector3f& normal, svtkPlane* plane)
{
  return svtkMolecule::GetPlaneFromBond(bond.GetBeginAtom(), bond.GetEndAtom(), normal, plane);
}

//----------------------------------------------------------------------------
bool svtkMolecule::GetPlaneFromBond(
  const svtkAtom& atom1, const svtkAtom& atom2, const svtkVector3f& normal, svtkPlane* plane)
{
  if (plane == nullptr)
  {
    return false;
  }

  svtkVector3f v(atom1.GetPosition() - atom2.GetPosition());

  svtkVector3f n_i(normal);
  svtkVector3f unitV(v.Normalized());

  // Check if vectors are (nearly) parallel
  if (unitV.Compare(n_i.Normalized(), 1e-7))
  {
    return false;
  }

  // calculate projection of n_i onto v
  // TODO Remove or restore this when scalar mult. is supported again
  // svtkVector3d proj (unitV * n_i.Dot(unitV));
  double n_iDotUnitV = n_i.Dot(unitV);
  svtkVector3f proj(unitV[0] * n_iDotUnitV, unitV[1] * n_iDotUnitV, unitV[2] * n_iDotUnitV);
  // end svtkVector reimplementation TODO

  // Calculate actual normal:
  svtkVector3f realNormal(n_i - proj);

  // Create plane:
  svtkVector3f pos(atom1.GetPosition());
  plane->SetOrigin(pos.Cast<double>().GetData());
  plane->SetNormal(realNormal.Cast<double>().GetData());
  return true;
}

//------------------------------------------------------------------------------
bool svtkMolecule::HasLattice()
{
  return this->Lattice != nullptr;
}

//------------------------------------------------------------------------------
void svtkMolecule::ClearLattice()
{
  this->SetLattice(nullptr);
}

//------------------------------------------------------------------------------
void svtkMolecule::SetLattice(svtkMatrix3x3* matrix)
{
  if (!matrix)
  {
    if (this->Lattice)
    {
      // If we're clearing a matrix, zero out the origin:
      this->LatticeOrigin = svtkVector3d(0., 0., 0.);
      this->Lattice = nullptr;
      this->Modified();
    }
  }
  else if (this->Lattice != matrix)
  {
    this->Lattice = matrix;
    this->Modified();
  }
}

//------------------------------------------------------------------------------
void svtkMolecule::SetLattice(const svtkVector3d& a, const svtkVector3d& b, const svtkVector3d& c)
{
  if (this->Lattice == nullptr)
  {
    this->Lattice.TakeReference(svtkMatrix3x3::New());
    this->Modified();
  }

  double* mat = this->Lattice->GetData();
  if (mat[0] != a[0] || mat[1] != b[0] || mat[2] != c[0] || mat[3] != a[1] || mat[4] != b[1] ||
    mat[5] != c[1] || mat[6] != a[2] || mat[7] != b[2] || mat[8] != c[2])
  {
    mat[0] = a[0];
    mat[1] = b[0];
    mat[2] = c[0];
    mat[3] = a[1];
    mat[4] = b[1];
    mat[5] = c[1];
    mat[6] = a[2];
    mat[7] = b[2];
    mat[8] = c[2];
    this->Modified();
  }
}

//------------------------------------------------------------------------------
svtkMatrix3x3* svtkMolecule::GetLattice()
{
  return this->Lattice;
}

//------------------------------------------------------------------------------
void svtkMolecule::GetLattice(svtkVector3d& a, svtkVector3d& b, svtkVector3d& c)
{
  if (this->Lattice)
  {
    double* mat = this->Lattice->GetData();
    a[0] = mat[0];
    a[1] = mat[3];
    a[2] = mat[6];
    b[0] = mat[1];
    b[1] = mat[4];
    b[2] = mat[7];
    c[0] = mat[2];
    c[1] = mat[5];
    c[2] = mat[8];
  }
  else
  {
    a = b = c = svtkVector3d(0., 0., 0.);
  }
}

//------------------------------------------------------------------------------
void svtkMolecule::GetLattice(svtkVector3d& a, svtkVector3d& b, svtkVector3d& c, svtkVector3d& origin)
{
  if (this->Lattice)
  {
    double* mat = this->Lattice->GetData();
    a[0] = mat[0];
    a[1] = mat[3];
    a[2] = mat[6];
    b[0] = mat[1];
    b[1] = mat[4];
    b[2] = mat[7];
    c[0] = mat[2];
    c[1] = mat[5];
    c[2] = mat[8];
    origin = this->LatticeOrigin;
  }
  else
  {
    a = b = c = origin = svtkVector3d(0., 0., 0.);
  }
}

//----------------------------------------------------------------------------
svtkUnsignedCharArray* svtkMolecule::GetAtomGhostArray()
{
  return svtkArrayDownCast<svtkUnsignedCharArray>(
    this->GetVertexData()->GetArray(svtkDataSetAttributes::GhostArrayName()));
}

//----------------------------------------------------------------------------
void svtkMolecule::AllocateAtomGhostArray()
{
  if (this->GetAtomGhostArray() == nullptr)
  {
    svtkNew<svtkUnsignedCharArray> ghosts;
    ghosts->SetName(svtkDataSetAttributes::GhostArrayName());
    ghosts->SetNumberOfComponents(1);
    ghosts->SetNumberOfTuples(this->GetNumberOfAtoms());
    ghosts->FillComponent(0, 0);
    this->GetVertexData()->AddArray(ghosts);
  }
  else
  {
    this->GetAtomGhostArray()->SetNumberOfTuples(this->GetNumberOfAtoms());
  }
}

//----------------------------------------------------------------------------
svtkUnsignedCharArray* svtkMolecule::GetBondGhostArray()
{
  return svtkArrayDownCast<svtkUnsignedCharArray>(
    this->GetEdgeData()->GetArray(svtkDataSetAttributes::GhostArrayName()));
}

//----------------------------------------------------------------------------
void svtkMolecule::AllocateBondGhostArray()
{
  if (this->GetBondGhostArray() == nullptr)
  {
    svtkNew<svtkUnsignedCharArray> ghosts;
    ghosts->SetName(svtkDataSetAttributes::GhostArrayName());
    ghosts->SetNumberOfComponents(1);
    ghosts->SetNumberOfTuples(this->GetNumberOfBonds());
    ghosts->FillComponent(0, 0);
    this->GetEdgeData()->AddArray(ghosts);
  }
  else
  {
    this->GetBondGhostArray()->SetNumberOfTuples(this->GetNumberOfBonds());
  }
}

//----------------------------------------------------------------------------
int svtkMolecule::Initialize(
  svtkPoints* atomPositions, svtkDataArray* atomicNumberArray, svtkDataSetAttributes* atomData)
{
  // Start with default initialization the molecule
  this->Initialize();

  // if no atomicNumberArray given, look for one in atomData
  if (!atomicNumberArray && atomData)
  {
    atomicNumberArray = atomData->GetArray(this->GetAtomicNumberArrayName());
  }

  if (!atomPositions && !atomicNumberArray)
  {
    svtkDebugMacro(<< "Atom position and atomic numbers were not found: skip atomic data.");
    return 1;
  }

  if (!atomPositions || !atomicNumberArray)
  {
    svtkDebugMacro(<< "Empty atoms or atomic numbers.");
    return 0;
  }

  // ensure it is a short array
  svtkNew<svtkUnsignedShortArray> newAtomicNumberShortArray;
  if (svtkUnsignedShortArray::SafeDownCast(atomicNumberArray))
  {
    newAtomicNumberShortArray->ShallowCopy(atomicNumberArray);
  }
  else
  {
    svtkIdType nbPoints = atomicNumberArray->GetNumberOfTuples();
    newAtomicNumberShortArray->SetNumberOfComponents(1);
    newAtomicNumberShortArray->SetNumberOfTuples(nbPoints);
    newAtomicNumberShortArray->SetName(atomicNumberArray->GetName());
    for (svtkIdType i = 0; i < nbPoints; i++)
    {
      newAtomicNumberShortArray->SetTuple1(i, atomicNumberArray->GetTuple1(i));
    }
  }

  int nbAtoms = atomPositions->GetNumberOfPoints();
  if (nbAtoms != newAtomicNumberShortArray->GetNumberOfTuples())
  {
    svtkErrorMacro(<< "Number of atoms not equal to number of atomic numbers.");
    return 0;
  }
  if (atomData && nbAtoms != atomData->GetNumberOfTuples())
  {
    svtkErrorMacro(<< "Number of atoms not equal to number of atom properties.");
    return 0;
  }

  static const std::string atomicNumberName = this->GetAtomicNumberArrayName();

  // update atoms positions
  this->ForceOwnership();
  this->Internals->Adjacency.resize(nbAtoms, svtkVertexAdjacencyList());
  this->SetPoints(atomPositions);

  // if atom properties exists, copy it in VertexData and look for an atomic number in its arrays.
  if (atomData)
  {
    this->GetVertexData()->ShallowCopy(atomData);

    // if atomData contains an array with the atomic number name,
    // copy this array with a new name as we will overwrite it.
    svtkDataArray* otherArray = atomData->GetArray(atomicNumberName.c_str());
    if (otherArray && otherArray != static_cast<svtkAbstractArray*>(atomicNumberArray))
    {
      this->GetVertexData()->RemoveArray(atomicNumberName.c_str());

      // create a new name for the copied array.
      std::string newName = std::string("Original ") + atomicNumberName;

      // if the new name is available create a copy of the array with another name, and add it
      // else no backup is done, array will be replaced.
      if (!atomData->GetArray(newName.c_str()))
      {
        svtkDataArray* otherArrayCopy = otherArray->NewInstance();
        otherArrayCopy->ShallowCopy(otherArray);
        otherArrayCopy->SetName(newName.c_str());
        this->GetVertexData()->AddArray(otherArrayCopy);
        otherArrayCopy->Delete();
      }
      else
      {
        svtkWarningMacro(<< "Array '" << atomicNumberName << "' was replaced.");
      }
    }
  }

  // add atomic number array: if given array has the expected name, add it directly
  // (it will replace the current one). Else copy it and add it with atomic number name.
  if (atomicNumberName == newAtomicNumberShortArray->GetName())
  {
    this->GetVertexData()->AddArray(newAtomicNumberShortArray);
  }
  else
  {
    svtkNew<svtkUnsignedShortArray> atomicNumberArrayCopy;
    atomicNumberArrayCopy->ShallowCopy(newAtomicNumberShortArray);
    atomicNumberArrayCopy->SetName(atomicNumberName.c_str());
    this->GetVertexData()->AddArray(atomicNumberArrayCopy.Get());
  }

  this->Modified();
  return 1;
}

//----------------------------------------------------------------------------
int svtkMolecule::Initialize(svtkMolecule* molecule)
{
  if (molecule == nullptr)
  {
    this->Initialize();
    return 1;
  }

  return this->Initialize(
    molecule->GetPoints(), molecule->GetAtomicNumberArray(), molecule->GetVertexData());
}

//----------------------------------------------------------------------------
svtkMolecule* svtkMolecule::GetData(svtkInformation* info)
{
  return info ? svtkMolecule::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkMolecule* svtkMolecule::GetData(svtkInformationVector* v, int i)
{
  return svtkMolecule::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
unsigned long svtkMolecule::GetActualMemorySize()
{
  unsigned long size = this->Superclass::GetActualMemorySize();
  if (this->ElectronicData)
  {
    size += this->ElectronicData->GetActualMemorySize();
  }
  if (this->AtomGhostArray)
  {
    size += this->AtomGhostArray->GetActualMemorySize();
  }
  if (this->BondGhostArray)
  {
    size += this->BondGhostArray->GetActualMemorySize();
  }
  return size;
}
