/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMolecule.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkMolecule
 * @brief   class describing a molecule
 *
 *
 *
 * svtkMolecule and the convenience classes svtkAtom and svtkBond
 * describe the geometry and connectivity of a molecule. The molecule
 * can be constructed using the AppendAtom() and AppendBond() methods in one
 * of two ways; either by fully specifying the atom/bond in a single
 * call, or by incrementally setting the various attributes using the
 * convenience svtkAtom and svtkBond classes:
 *
 * Single call:
 * \code
 * svtkMolecule *mol = svtkMolecule::New();
 * svtkAtom h1 = mol->AppendAtom(1, 0.0, 0.0, -0.5);
 * svtkAtom h2 = mol->AppendAtom(1, 0.0, 0.0,  0.5);
 * svtkBond b  = mol->AppendBond(h1, h2, 1);
 * \endcode
 *
 * Incremental:
 * \code
 * svtkMolecule *mol = svtkMolecule::New();
 *
 * svtkAtom h1 = mol->AppendAtom();
 * h1.SetAtomicNumber(1);
 * h1.SetPosition(0.0, 0.0, -0.5);
 *
 * svtkAtom h2 = mol->AppendAtom();
 * h2.SetAtomicNumber(1);
 * svtkVector3d displacement (0.0, 0.0, 1.0);
 * h2.SetPosition(h1.GetPositionAsVector3d() + displacement);
 *
 * svtkBond b  = mol->AppendBond(h1, h2, 1);
 * \endcode
 *
 * Both of the above methods will produce the same molecule, two
 * hydrogens connected with a 1.0 Angstrom single bond, aligned to the
 * z-axis. The second example also demonstrates the use of SVTK's
 * svtkVector class, which is fully supported by the Chemistry kit.
 *
 * The svtkMolecule object is intended to be used with the
 * svtkMoleculeMapper class for visualizing molecular structure using
 * common rendering techniques.
 *
 * \warning While direct use of the underlying svtkUndirectedGraph
 * structure is possible due to svtkMolecule's public inheritance, this
 * should not be relied upon and may change in the future.
 *
 * @sa
 * svtkAtom svtkBond svtkMoleculeMapper svtkPeriodicTable
 */

#ifndef svtkMolecule_h
#define svtkMolecule_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkSmartPointer.h"          // For svtkSmartPointer
#include "svtkUndirectedGraph.h"

#include "svtkAtom.h" // Simple proxy class dependent on svtkMolecule
#include "svtkBond.h" // Simple proxy class dependent on svtkMolecule

#include "svtkVector.h" // Small templated vector convenience class

class svtkAbstractElectronicData;
class svtkDataArray;
class svtkInformation;
class svtkInformationVector;
class svtkMatrix3x3;
class svtkPlane;
class svtkPoints;
class svtkUnsignedCharArray;
class svtkUnsignedShortArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkMolecule : public svtkUndirectedGraph
{
public:
  static svtkMolecule* New();
  svtkTypeMacro(svtkMolecule, svtkUndirectedGraph);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  void Initialize() override;

  /**
   * Return what type of dataset this is.
   */
  int GetDataObjectType() override { return SVTK_MOLECULE; }

  /**
   * Add new atom with atomic number 0 (dummy atom) at origin. Return
   * a svtkAtom that refers to the new atom.
   */
  svtkAtom AppendAtom() { return this->AppendAtom(0, 0., 0., 0.); }

  //@{
  /**
   * Add new atom with the specified atomic number and position. Return a
   * svtkAtom that refers to the new atom.
   */
  svtkAtom AppendAtom(unsigned short atomicNumber, double x, double y, double z);
  svtkAtom AppendAtom(unsigned short atomicNumber, const svtkVector3f& pos)
  {
    return this->AppendAtom(atomicNumber, pos[0], pos[1], pos[2]);
  }

  svtkAtom AppendAtom(unsigned short atomicNumber, double pos[3])
  {
    return this->AppendAtom(atomicNumber, pos[0], pos[1], pos[2]);
  }
  //@}

  /**
   * Return a svtkAtom that refers to the atom with the specified id.
   */
  svtkAtom GetAtom(svtkIdType atomId);

  /**
   * Return the number of atoms in the molecule.
   */
  svtkIdType GetNumberOfAtoms();

  //@{
  /**
   * Add a bond between the specified atoms, optionally setting the
   * bond order (default: 1). Return a svtkBond object referring to the
   * new bond.
   */
  svtkBond AppendBond(svtkIdType atom1, svtkIdType atom2, unsigned short order = 1);
  svtkBond AppendBond(const svtkAtom& atom1, const svtkAtom& atom2, unsigned short order = 1)
  {
    return this->AppendBond(atom1.Id, atom2.Id, order);
  }
  //@}

  /**
   * Return a svtkAtom that refers to the bond with the specified id.
   */
  svtkBond GetBond(svtkIdType bondId);

  /**
   * Return the number of bonds in the molecule.
   */
  svtkIdType GetNumberOfBonds();

  /**
   * Return the atomic number of the atom with the specified id.
   */
  unsigned short GetAtomAtomicNumber(svtkIdType atomId);

  /**
   * Set the atomic number of the atom with the specified id.
   */
  void SetAtomAtomicNumber(svtkIdType atomId, unsigned short atomicNum);

  //@{
  /**
   * Set the position of the atom with the specified id.
   */
  void SetAtomPosition(svtkIdType atomId, const svtkVector3f& pos);
  void SetAtomPosition(svtkIdType atomId, double x, double y, double z);
  void SetAtomPosition(svtkIdType atomId, double pos[3])
  {
    this->SetAtomPosition(atomId, pos[0], pos[1], pos[2]);
  }
  //@}

  //@{
  /**
   * Get the position of the atom with the specified id.
   */
  svtkVector3f GetAtomPosition(svtkIdType atomId);
  void GetAtomPosition(svtkIdType atomId, float pos[3]);
  void GetAtomPosition(svtkIdType atomId, double pos[3]);
  //@}

  //@{
  /**
   * Get/Set the bond order of the bond with the specified id
   */
  void SetBondOrder(svtkIdType bondId, unsigned short order);
  unsigned short GetBondOrder(svtkIdType bondId);
  //@}

  /**
   * Get the bond length of the bond with the specified id

   * \note If the associated svtkBond object is already available,
   * svtkBond::GetBondLength is potentially much faster than this
   * function, as a list of all bonds may need to be constructed to
   * locate the appropriate bond.
   * \sa UpdateBondList()
   */
  double GetBondLength(svtkIdType bondId);

  //@{
  /**
   * Access the raw arrays used in this svtkMolecule instance
   */
  svtkPoints* GetAtomicPositionArray();
  svtkUnsignedShortArray* GetAtomicNumberArray();
  svtkUnsignedShortArray* GetBondOrdersArray();
  //@}

  //@{
  /**
   * Set/Get the AbstractElectronicData-subclassed object for this molecule.
   */
  svtkGetObjectMacro(ElectronicData, svtkAbstractElectronicData);
  virtual void SetElectronicData(svtkAbstractElectronicData*);
  //@}

  /**
   * Performs the same operation as ShallowCopy(),
   * but instead of reporting an error for an incompatible graph,
   * returns false.
   */
  bool CheckedShallowCopy(svtkGraph* g) override;

  /**
   * Performs the same operation as DeepCopy(),
   * but instead of reporting an error for an incompatible graph,
   * returns false.
   */
  bool CheckedDeepCopy(svtkGraph* g) override;

  /**
   * Shallow copies the data object into this molecule.
   */
  void ShallowCopy(svtkDataObject* obj) override;

  /**
   * Deep copies the data object into this molecule.
   */
  void DeepCopy(svtkDataObject* obj) override;

  /**
   * Shallow copies the atoms and bonds from @a m into @a this.
   */
  virtual void ShallowCopyStructure(svtkMolecule* m);

  /**
   * Deep copies the atoms and bonds from @a m into @a this.
   */
  virtual void DeepCopyStructure(svtkMolecule* m);

  /**
   * Shallow copies attributes (i.e. everything besides atoms and bonds) from
   * @a m into @a this.
   */
  virtual void ShallowCopyAttributes(svtkMolecule* m);

  /**
   * Deep copies attributes (i.e. everything besides atoms and bonds) from
   * @a m into @a this.
   */
  virtual void DeepCopyAttributes(svtkMolecule* m);

  //@{
  /**
   * Obtain the plane that passes through the indicated bond with the given
   * normal. If the plane is set successfully, the function returns true.

   * If the normal is not orthogonal to the bond, a new normal will be
   * constructed in such a way that the plane will be orthogonal to
   * the plane spanned by the bond vector and the input normal vector.

   * This ensures that the plane passes through the bond, and the
   * normal is more of a "hint" indicating the orientation of the plane.

   * The new normal (n) is defined as the input normal vector (n_i) minus
   * the projection of itself (proj[n_i]_v) onto the bond vector (v):

   * @verbatim
   * v ^
   * |  n = (n_i - proj[n_j]_v)
   * proj[n_i]_v ^  |----x
   * |  |   /
   * |  |  / n_i
   * |  | /
   * |  |/
   * @endverbatim

   * If n_i is parallel to v, a warning will be printed and no plane will be
   * added. Obviously, n_i must not be parallel to v.
   */
  static bool GetPlaneFromBond(const svtkBond& bond, const svtkVector3f& normal, svtkPlane* plane);
  static bool GetPlaneFromBond(
    const svtkAtom& atom1, const svtkAtom& atom2, const svtkVector3f& normal, svtkPlane* plane);
  //@}

  /**
   * Return true if a unit cell lattice is defined.
   */
  bool HasLattice();

  /**
   * Remove any unit cell lattice information from the molecule.
   */
  void ClearLattice();

  //@{
  /**
   * The unit cell vectors. The matrix is stored using a row-major layout, with
   * the vectors encoded as columns.
   */
  void SetLattice(svtkMatrix3x3* matrix);
  void SetLattice(const svtkVector3d& a, const svtkVector3d& b, const svtkVector3d& c);
  //@}

  /**
   * Get the unit cell lattice vectors. The matrix is stored using a row-major
   * layout, with the vectors encoded as columns. Will return nullptr if no
   * unit cell information is available.
   * @sa GetLatticeOrigin
   */
  svtkMatrix3x3* GetLattice();

  //@{
  /**
   * Get the unit cell lattice vectors, and optionally, the origin.
   */
  void GetLattice(svtkVector3d& a, svtkVector3d& b, svtkVector3d& c);
  void GetLattice(svtkVector3d& a, svtkVector3d& b, svtkVector3d& c, svtkVector3d& origin);
  //@}

  //@{
  /**
   * Get the unit cell origin (for rendering purposes).
   */
  svtkGetMacro(LatticeOrigin, svtkVector3d);
  svtkSetMacro(LatticeOrigin, svtkVector3d);
  //@}

  /**
   * Get the array that defines the ghost type of each atom.
   */
  svtkUnsignedCharArray* GetAtomGhostArray();

  /**
   * Allocate ghost array for atoms.
   */
  void AllocateAtomGhostArray();

  /**
   * Get the array that defines the ghost type of each bond.
   */
  svtkUnsignedCharArray* GetBondGhostArray();

  /**
   * Allocate ghost array for bonds.
   */
  void AllocateBondGhostArray();

  /**
   * Initialize a molecule with an atom per input point.
   * Parameters atomPositions and atomicNumberArray should have the same size.
   */
  int Initialize(
    svtkPoints* atomPositions, svtkDataArray* atomicNumberArray, svtkDataSetAttributes* atomData);

  /**
   * Overloads Initialize method.
   */
  int Initialize(svtkPoints* atomPositions, svtkDataSetAttributes* atomData)
  {
    return this->Initialize(atomPositions, nullptr, atomData);
  }

  /**
   * Use input molecule points, atomic number and atomic data to initialize the new molecule.
   */
  int Initialize(svtkMolecule* molecule);

  //@{
  /**
   * Retrieve a molecule from an information vector.
   */
  static svtkMolecule* GetData(svtkInformation* info);
  static svtkMolecule* GetData(svtkInformationVector* v, int i = 0);
  //@}

  /**
   * Return the VertexData of the underlying graph
   */
  svtkDataSetAttributes* GetAtomData() { return this->GetVertexData(); }

  /**
   * Return the EdgeData of the underlying graph
   */
  svtkDataSetAttributes* GetBondData() { return this->GetEdgeData(); }

  /**
   * Return the edge id from the underlying graph.
   */
  svtkIdType GetBondId(svtkIdType a, svtkIdType b) { return this->GetEdgeId(a, b); }

  //@{
  /**
   * Get/Set the atomic number array name.
   */
  svtkSetStringMacro(AtomicNumberArrayName);
  svtkGetStringMacro(AtomicNumberArrayName);
  //@}

  //@{
  /**
   * Get/Set the bond orders array name.
   */
  svtkSetStringMacro(BondOrdersArrayName);
  svtkGetStringMacro(BondOrdersArrayName);
  //@}

  /**
   * Return the actual size of the data in kibibytes (1024 bytes). This number
   * is valid only after the pipeline has updated. The memory size
   * returned is guaranteed to be greater than or equal to the
   * memory required to represent the data (e.g., extra space in
   * arrays, etc. are not included in the return value).
   */
  unsigned long GetActualMemorySize() override;

protected:
  svtkMolecule();
  ~svtkMolecule() override;

  /**
   * Copy bonds and atoms.
   */
  virtual void CopyStructureInternal(svtkMolecule* m, bool deep);

  /**
   * Copy everything but bonds and atoms.
   */
  virtual void CopyAttributesInternal(svtkMolecule* m, bool deep);

  //@{
  /**
   * The graph superclass does not provide fast random access to the
   * edge (bond) data. All random access is performed using a lookup
   * table that must be rebuilt periodically. These allow for lazy
   * building of the lookup table
   */
  bool BondListIsDirty;
  void SetBondListDirty() { this->BondListIsDirty = true; }
  void UpdateBondList();
  svtkIdTypeArray* GetBondList();
  //@}

  friend class svtkAtom;
  friend class svtkBond;

  svtkAbstractElectronicData* ElectronicData;
  svtkSmartPointer<svtkMatrix3x3> Lattice;
  svtkVector3d LatticeOrigin;

  svtkUnsignedCharArray* AtomGhostArray;
  svtkUnsignedCharArray* BondGhostArray;

  char* AtomicNumberArrayName;
  char* BondOrdersArrayName;

private:
  svtkMolecule(const svtkMolecule&) = delete;
  void operator=(const svtkMolecule&) = delete;
};

#endif
