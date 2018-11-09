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

namespace senseiLammps
{

struct lammpsDataAdaptor::DInternals
{
  vtkSmartPointer<vtkMultiBlockDataSet> mesh;
  vtkSmartPointer<vtkDoubleArray> AtomPositions;
  vtkSmartPointer<vtkIntArray> AtomTypes;
  vtkSmartPointer<vtkIntArray> AtomIDs;
  vtkSmartPointer<vtkCellArray> vertices;
  double xsublo, ysublo, zsublo, xsubhi, ysubhi, zsubhi;
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
    for( int i=0; i < nlocal; i++) {
	internals.vertices->InsertNextCell (1, pid);
	pid[0]++;
    }
  }

  // number of atoms
  internals.nlocal = nlocal;
  internals.nghost = nghost;

  // bounding box
  internals.xsublo = xsublo;
  internals.ysublo = ysublo;
  internals.zsublo = zsublo;
  internals.xsubhi = xsubhi;
  internals.ysubhi = ysubhi;
  internals.zsubhi = zsubhi;

  // timestep
  this->SetDataTimeStep(ntimestep);

}

//-----------------------------------------------------------------------------
void lammpsDataAdaptor::GetBounds ( double &xsublo, double &xsubhi, 
                                   double &ysublo, double &ysubhi, 
                                   double &zsublo, double &zsubhi)
{
  DInternals& internals = (*this->Internals);

  xsublo = internals.xsublo;
  ysublo = internals.ysublo;
  zsublo = internals.zsublo;
  xsubhi = internals.xsubhi;
  ysubhi = internals.ysubhi;
  zsubhi = internals.zsubhi;
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
int lammpsDataAdaptor::GetMeshName(unsigned int id, std::string &meshName)
{
  if (id == 0) {
    meshName = "atoms";
    return 0;
  }

  SENSEI_ERROR("Failed to get mesh name")
  return -1;
}

//-----------------------------------------------------------------------------
int lammpsDataAdaptor::GetMesh(const std::string &meshName, bool structureOnly,
    vtkDataObject *&mesh)
{
  if (meshName != "atoms") {  
    SENSEI_ERROR("No mesh \"" << meshName << "\"")
    return -1;
  }

  DInternals& internals = (*this->Internals);

  if (!internals.mesh){
  	vtkSmartPointer<vtkPolyData> pd = vtkSmartPointer<vtkPolyData>::New();

  	if(!structureOnly){
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
  if (meshName != "atoms"){
    SENSEI_ERROR("No mesh \"" << meshName << "\"")
    return -1;
  }

  if (association != vtkDataObject::FIELD_ASSOCIATION_POINTS){
    SENSEI_ERROR("No cell data on mesh")
    return -1;
  }

  if (arrayName == "type"){
  	
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

  if (arrayName == "id"){
  	
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



//-----------------------------------------------------------------------------
int lammpsDataAdaptor::GetNumberOfArrays(const std::string &meshName,
  int association, unsigned int &numberOfArrays)
{
  numberOfArrays = 0;

  if (meshName != "atoms") {
    SENSEI_ERROR("No mesh \"" << meshName << "\"")
    return -1;
    }

  if (association == vtkDataObject::POINT) {
    numberOfArrays = 2;
  }

  return 0;
}


//-----------------------------------------------------------------------------
int lammpsDataAdaptor::GetArrayName(const std::string &meshName, int association,
  unsigned int index, std::string &arrayName)
{
  arrayName = "";

  if (meshName != "atoms") {
    SENSEI_ERROR("No mesh \"" << meshName << "\"")
    return -1;
  }

  if (association != vtkDataObject::POINT) {
    SENSEI_ERROR("No cell data on mesh \"" << meshName << "\"")
    return -1;
  }

  switch (index)
  {
    case 0:
      arrayName = "type";
      break;
    case 1:
      arrayName = "id";
      break;
    default:
      SENSEI_ERROR("Array index " << index << " out of bounds")
      return -1;
  }

  return 0;
}

//----------------------------------------------------------------------------
int lammpsDataAdaptor::GetMeshHasGhostCells(const std::string &/*meshName*/, 
  int &nLayers)
{
  nLayers = 0;
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
int lammpsDataAdaptor::ReleaseData()
{
  DInternals& internals = (*this->Internals);

  internals.mesh = NULL;
  internals.AtomPositions = NULL;
  internals.AtomTypes = NULL;
  internals.AtomIDs = NULL;
  internals.nlocal = 0;
  internals.nghost = 0;
  internals.xsublo = 0;
  internals.ysublo = 0;
  internals.zsublo = 0;
  internals.xsubhi = 0;
  internals.ysubhi = 0;
  internals.zsubhi = 0;

  return 0;
}

}	// senseiLammps


