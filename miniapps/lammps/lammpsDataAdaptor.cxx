#include "lammpsDataAdaptor.h"
#include "Error.h"
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkCellArray.h>

static
std::array<int,2> getArrayRange(unsigned long nSize, int *x) {
  int xmin = std::numeric_limits<int>::max(); 
  int xmax = std::numeric_limits<int>::lowest();
  for(int i=0; i<nSize; ++i) {
    xmin = std::min(xmin, x[i]);
    xmax = std::max(xmax, x[i]);
  }

  return {xmin, xmax};
}

static
std::array<double,2> getArrayRange(unsigned long nSize,double *x) {
  double xmin = std::numeric_limits<double>::max(); 
  double xmax = std::numeric_limits<double>::lowest();
  for(int i=0; i<nSize; ++i) {
    xmin = std::min(xmin, x[i]);
    xmax = std::max(xmax, x[i]);
  }

  return {xmin, xmax};
}

static 
void getBounds(const sdiy::DiscreteBounds &db, double *ext) 
{
  ext[0] = db.min[0];
  ext[1] = db.max[0];
  ext[2] = db.min[1];
  ext[3] = db.max[1];
  ext[4] = db.min[2];
  ext[5] = db.max[2];
}

namespace senseiLammps
{

struct lammpsDataAdaptor::DInternals
{
  vtkSmartPointer<vtkMultiBlockDataSet> mesh;
  vtkSmartPointer<vtkDoubleArray> AtomPositions;
  vtkSmartPointer<vtkIntArray> AtomTypes;
  vtkSmartPointer<vtkIntArray> AtomIDs;
  vtkSmartPointer<vtkCellArray> vertices;
  sdiy::DiscreteBounds DomainBounds;
  sdiy::DiscreteBounds BlockBounds;
  sdiy::DiscreteBounds typeRange;
  sdiy::DiscreteBounds idRange;
  int nlocal, nghost;
  double **x;
  int *type;
  int *id;
};

//-----------------------------------------------------------------------------
senseiNewMacro(lammpsDataAdaptor);

//-----------------------------------------------------------------------------
lammpsDataAdaptor::lammpsDataAdaptor() :
  Internals(new lammpsDataAdaptor::DInternals())
{
}

//-----------------------------------------------------------------------------
lammpsDataAdaptor::~lammpsDataAdaptor()
{
  delete this->Internals;
}

//-----------------------------------------------------------------------------
void lammpsDataAdaptor::Initialize()
{  
  this->ReleaseData();
}

//-----------------------------------------------------------------------------
void lammpsDataAdaptor::AddLAMMPSData( long ntimestep, int nlocal, int *id, 
                                      int nghost, int *type, double **x, 
                                      double xsublo, double xsubhi, 
                                      double ysublo, double ysubhi, 
                                      double zsublo, double zsubhi)
{
  DInternals& internals = (*this->Internals);

  if(!internals.AtomPositions)
    {
    internals.AtomPositions = vtkSmartPointer<vtkDoubleArray>::New();
    }

  if(!internals.AtomTypes)
    {
    internals.AtomTypes = vtkSmartPointer<vtkIntArray>::New();
    }

  if(!internals.AtomIDs)
    {
    internals.AtomIDs = vtkSmartPointer<vtkIntArray>::New();
    }

  if(!internals.vertices)
    {
    internals.vertices = vtkSmartPointer<vtkCellArray>::New();
    }

  // atom coordinates
  if (internals.AtomPositions)
    {
    //long nvals = nlocal + nghost;
    long nvals = nlocal;

    internals.AtomPositions->SetNumberOfComponents(3);
    internals.AtomPositions->SetArray(*x, nvals*3, 1); 
    internals.AtomPositions->SetName("positions");

    internals.x = x;
    }
  else 
    {
    SENSEI_ERROR("Error. Internal AtomPositions structure not initialized")
    }

  // atom types
  if (internals.AtomTypes)
    {
    //long nvals = nlocal + nghost;
    long nvals = nlocal;

    internals.AtomTypes->SetNumberOfComponents(1);
    internals.AtomTypes->SetArray(type, nvals, 1);
    internals.AtomTypes->SetName("type");

    internals.type = type;  
    }
  else 
    {
    SENSEI_ERROR("Error. Internal AtomTypes structure not initialized")
    }

  // atom IDs
  if (internals.AtomIDs)
    {
    //long nvals = nlocal + nghost;
    long nvals = nlocal;

    internals.AtomIDs->SetNumberOfComponents(1);
    internals.AtomIDs->SetArray(id, nvals, 1);
    internals.AtomIDs->SetName("id");

    internals.id = id;  
    }
  else 
    {
    SENSEI_ERROR("Error. Internal AtomIDs structure not initialized")
    }

  // vertices
  if (internals.vertices)
    {
    vtkIdType pid[1] = {0};

    //for( int i=0; i < nlocal+nghost; i++) {
    for( int i=0; i < nlocal; i++) 
      {
      internals.vertices->InsertNextCell (1, pid);
      pid[0]++;
      }
    }

  // number of atoms
  internals.nlocal = nlocal;
  internals.nghost = nghost;

  std::array<double,2> x_range = getArrayRange(nlocal, x[0]);
	std::array<double,2> y_range = getArrayRange(nlocal, x[1]);
	std::array<double,2> z_range = getArrayRange(nlocal, x[2]);

  // bounding box
  this->SetDomainBounds(xsublo, xsubhi, ysublo, ysubhi, zsublo, zsubhi);
  this->SetBlockBounds(
      x_range[0], x_range[1],
      y_range[0], y_range[1],
      z_range[0], z_range[1]);

  /// XXX Set type and id range
  this->Internals->typeRange.min[0] = std::numeric_limits<int>::max();
  this->Internals->typeRange.max[0] = std::numeric_limits<int>::min();
  this->Internals->idRange.min[0] = std::numeric_limits<int>::max();
  this->Internals->idRange.max[0] = std::numeric_limits<int>::min();

  // timestep
  this->SetDataTimeStep(ntimestep);

}

void lammpsDataAdaptor::SetBlockBounds(double *x, int nelem) {
  this-Internals->  
}

void lammpsDataAdaptor::SetDomainBounds(double xmin, double xmax,
    double ymin, double ymax, double zmin, double zmax) {
  this->Internals->DomainBounds.min[0] = xmin;
  this->Internals->DomainBounds.min[1] = ymin;
  this->Internals->DomainBounds.min[2] = zmin;

  this->Internals->DomainBounds.max[0] = xmax;
  this->Internals->DomainBounds.max[1] = ymax;
  this->Internals->DomainBounds.max[2] = zmax;
}

void lammpsDataAdaptor::GetN ( int &nlocal, int &nghost )
{
  DInternals& internals = (*this->Internals);
  
  nlocal = internals.nlocal;
  nghost = internals.nghost;
}

void lammpsDataAdaptor::GetPointers ( double **&x, int *&type)
{
  DInternals& internals = (*this->Internals);
  
  x = internals.x;
  type = internals.type;
}

//-----------------------------------------------------------------------------
void lammpsDataAdaptor::GetAtoms ( vtkDoubleArray *&atoms)
{
  DInternals& internals = (*this->Internals);

  if (internals.AtomPositions)
    atoms = internals.AtomPositions;
  else
    SENSEI_ERROR("Trying to get atom position array before setting it")
}

void lammpsDataAdaptor::GetTypes ( vtkIntArray *&types)
{
  DInternals& internals = (*this->Internals);

  if (internals.AtomTypes)
    types = internals.AtomTypes;
  else
    SENSEI_ERROR("Trying to get atom type array before setting it")
}

void lammpsDataAdaptor::GetIDs ( vtkIntArray *&ids)
{
  DInternals& internals = (*this->Internals);

  if (internals.AtomIDs)
    ids = internals.AtomIDs;
  else
    SENSEI_ERROR("Trying to get atom ID array before setting it")
}

//-----------------------------------------------------------------------------
int lammpsDataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  numMeshes = 1;
  return 0;
}

//-----------------------------------------------------------------------------
int lammpsDataAdaptor::GetMesh(const std::string &meshName, bool structureOnly,
    vtkDataObject *&mesh)
{
  if (meshName != "atoms") 
    {  
    SENSEI_ERROR("No mesh \"" << meshName << "\"")
    return -1;
    }

  DInternals& internals = (*this->Internals);

  if (!internals.mesh)
    {
    vtkSmartPointer<vtkPolyData> pd = vtkSmartPointer<vtkPolyData>::New();

    if(!structureOnly)
      {
      vtkSmartPointer<vtkPoints> pts = vtkSmartPointer<vtkPoints>::New();
      pts->SetData(internals.AtomPositions);
      pd->SetPoints(pts);
      }

    pd->SetVerts( internals.vertices );

    int rank, size; 
    MPI_Comm comm;
  	
    comm = GetCommunicator();
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size); 

    internals.mesh = vtkSmartPointer<vtkMultiBlockDataSet>::New();
    internals.mesh->SetNumberOfBlocks(size);
    internals.mesh->SetBlock(rank, pd);
    }

  mesh = internals.mesh;

  return 0;
}

//-----------------------------------------------------------------------------
int lammpsDataAdaptor::AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName)
{
  if (meshName != "atoms")
    {
    SENSEI_ERROR("No mesh \"" << meshName << "\"")
    return -1;
    }

  if (association != vtkDataObject::FIELD_ASSOCIATION_POINTS)
    {
    SENSEI_ERROR("No cell data on mesh")
    return -1;
    }

  if (arrayName == "type")
    {  	
    DInternals& internals = (*this->Internals);
    vtkMultiBlockDataSet* md = vtkMultiBlockDataSet::SafeDownCast(mesh);
    vtkSmartPointer<vtkPolyData> pd = vtkSmartPointer<vtkPolyData>::New();
  
    int rank;
    MPI_Comm comm;

    comm = GetCommunicator();
    MPI_Comm_rank(comm, &rank);

    pd = vtkPolyData::SafeDownCast(md->GetBlock(rank));
    pd->GetPointData()->AddArray(internals.AtomTypes);
    }

  if (arrayName == "id")
    {	
    DInternals& internals = (*this->Internals);
    vtkMultiBlockDataSet* md = vtkMultiBlockDataSet::SafeDownCast(mesh);
    vtkSmartPointer<vtkPolyData> pd = vtkSmartPointer<vtkPolyData>::New();
  
    int rank;
    MPI_Comm comm;

    comm = GetCommunicator();
    MPI_Comm_rank(comm, &rank);

    pd = vtkPolyData::SafeDownCast(md->GetBlock(rank));
    pd->GetPointData()->AddArray(internals.AtomIDs);
    }

  return 0;  
}

// not implemented
//----------------------------------------------------------------------------
int lammpsDataAdaptor::AddGhostCellsArray(vtkDataObject *mesh, const std::string &meshName)
{
  (void) mesh;
  (void) meshName;
  return 0;
}

//-----------------------------------------------------------------------------
int lammpsDataAdaptor::GetMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &metadata)
{

  if (id > 0)
    {
    SENSEI_ERROR("invalid mesh id " << id)
    return -1;
    }

  int rank, nRanks;
  MPI_Comm comm;

  comm = GetCommunicator();
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nRanks);


  int nBlocks = 1; // One block per rank
  metadata->MeshName = "atoms";
  metadata->MeshType = VTK_MULTIBLOCK_DATA_SET;
  metadata->BlockType = VTK_POLY_DATA;
  metadata->CoordinateType = VTK_DOUBLE;
  metadata->NumBlocks = nRanks;
  metadata->NumBlocksLocal = {nBlocks};
  metadata->NumGhostCells = this->Internals->nghost;
  metadata->NumArrays = 2;
  metadata->ArrayName = {"type", "id"};
  metadata->ArrayCentering = {vtkDataObject::POINT, vtkDataObject::POINT};
  metadata->ArrayComponents = {1, 1};
  metadata->ArrayType = {VTK_INT, VTK_INT};
  metadata->StaticMesh = 0;

  if (metadata->Flags.BlockExtentsSet())
    {
    SENSEI_WARNING("lammps data adaptor. Flags.BlockExtentsSet()")
    // There should be no extent for a PolyData, but ADIOS2 needs this
    std::array<int,6> ext = { 0, 0, 0, 0, 0, 0};
    metadata->Extent = std::move(ext);

    metadata->BlockExtents.reserve(nBlocks);
    metadata->BlockExtents.emplace_back(std::move(ext));
    }

  if (metadata->Flags.BlockBoundsSet())
    {
    SENSEI_WARNING("lammps data adaptor. Flags.BlockBoundsSet()")
    std::array<double, 6> bounds;
    getBounds(this->Internals->DomainBounds, bounds.data());
    metadata->Bounds = std::move(bounds);
    
    metadata->BlockBounds.reserve(nBlocks);
    
    getBounds(this->Internals->BlockBounds, bounds.data());
    metadata->BlockBounds.emplace_back(std::move(bounds));
    }

  if (metadata->Flags.BlockSizeSet())
    {
    int nCells = nlocal;
    SENSEI_WARNING("lammps data adaptor. Flags.BlockSizeSet()")
    metadata->BlockNumCells.push_back(nCells);
    metadata->BlockNumPoints.push_back(nCells);
    metadata->BlockCellArraySize.push_back(2 * nCells); // XXX- VTK_POINTS
    }

  if (metadata->Flags.BlockDecompSet())
    {
    metadata->BlockOwner.push_back(rank);
    metadata->BlockIds.push_back(rank);
    }

  if (metadata->Flags.BlockArrayRangeSet())
    {
    SENSEI_WARNING("lammps data adaptor. Flags.BlockArrayRangeSet()")
    
    std::array<int,2> typeBlockRange = getArrayRange(nvals, this->Internals->type);
    std::array<int,2> idBlockRange = getArrayRange(nvals, this->Internals->id);
    metadata->BlockArrayRange.push_back({typeBlockRange, idBlockRange});

		std::array<int,2> typeRange = { this->Internals->typeRange.min[0], this->Internals->typeRange.max[0] };
    std::array<int,2> idRange = { this->Internals->idRange.min[0], this->Internals->idRange.max[0] };
    metadata->ArrayRange.push_back(typeRange); 
    metadata->ArrayRange.push_back(idRange); 
}

  return 0;
}



//-----------------------------------------------------------------------------
int lammpsDataAdaptor::ReleaseData()
{
  DInternals& internals = (*this->Internals);

  internals.mesh = NULL;
  internals.AtomPositions = NULL;
  internals.AtomTypes = NULL;
  internals.AtomIDs = NULL;
  internals.nlocal = 0;
  internals.nghost = 0;

  return 0;
}

} //senseiLammps


