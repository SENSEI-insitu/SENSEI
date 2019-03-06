#include "ConduitDataAdaptor.h"
#include <conduit.hpp>
#include <conduit_blueprint.hpp>

#include "vtkCharArray.h"
#include "vtkUnsignedCharArray.h"
#include "vtkShortArray.h"
#include "vtkUnsignedShortArray.h"
#include "vtkIntArray.h"
#include "vtkUnsignedIntArray.h"
#include "vtkLongArray.h"
#include "vtkUnsignedLongArray.h"
#include "vtkLongLongArray.h"
#include "vtkUnsignedLongLongArray.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"

#include <vtkDataSetAttributes.h>
#include <vtkImageData.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkVector.h>
#include <vtkVectorOperators.h>
#include "vtkCellArray.h"
#include "vtkCellType.h"
#include "vtkCellData.h"
#include "vtkIdTypeArray.h"
#include "vtkPoints.h"
#include "vtkPointData.h"

#include "vtkRectilinearGrid.h"
#include "vtkStructuredGrid.h"
#include "vtkUnstructuredGrid.h"

#include <Error.h>
#include <sensei/DataAdaptor.h>
#include <cassert>
#include <sstream>
#include <diy/master.hpp>

namespace sensei 
{

//-----------------------------------------------------------------------------
senseiNewMacro(ConduitDataAdaptor);
//-----------------------------------------------------------------------------
ConduitDataAdaptor::ConduitDataAdaptor()
{
}

//-----------------------------------------------------------------------------
ConduitDataAdaptor::~ConduitDataAdaptor()
{
}

//-----------------------------------------------------------------------------
static int
ElementShapeNameToVTKCellType(const std::string &shape_name)
{
  if (shape_name == "point") return VTK_VERTEX;
  if (shape_name == "line")  return VTK_LINE;
  if (shape_name == "tri")   return VTK_TRIANGLE;
  if (shape_name == "quad")  return VTK_QUAD;
  if (shape_name == "hex")   return VTK_HEXAHEDRON;
  if (shape_name == "tet")   return VTK_TETRA;
  SENSEI_ERROR("Warning: Unsupported Element Shape: " << shape_name)
  return 0;
}

//-----------------------------------------------------------------------------

static int
VTKCellTypeSize(int cell_type)
{
  if (cell_type == VTK_VERTEX)     return 1;
  if (cell_type == VTK_LINE)       return 2;
  if (cell_type == VTK_TRIANGLE)   return 3;
  if (cell_type == VTK_QUAD)       return 4;
  if (cell_type == VTK_HEXAHEDRON) return 8;
  if (cell_type == VTK_TETRA)      return 4;
  return 0;
}

//-----------------------------------------------------------------------------

template<typename T> void 
Blueprint_MultiCompArray_To_VTKDataArray(const conduit::Node &n,
                                         int ncomps,
                                         int ntuples,
                                         vtkDataArray *darray)
{
        // vtk reqs us to set number of comps before number of tuples
  if( ncomps == 2) // we need 3 comps for vectors
    darray->SetNumberOfComponents(3);
  else
    darray->SetNumberOfComponents(ncomps);
 
  // set number of tuples
  darray->SetNumberOfTuples(ntuples);
        
  // handle multi-component case
  if(n.number_of_children() > 0)
  {
    for(vtkIdType c=0; c < ncomps; c++)
    {
      conduit::DataArray<T> vals_array = n[c].value();;

      for (vtkIdType i = 0; i < ntuples; i++)
      {
        darray->SetComponent(i, c, (double) vals_array[i]);

        if(ncomps == 2)
        {
          darray->SetComponent(i, 2, 0.0);
        }
      }
    }
  }
  // single array case
  else
  {
    conduit::DataArray<T> vals_array = n.value();
    for (vtkIdType i = 0; i < ntuples; i++)
    {
      darray->SetComponent(i,0, (double) vals_array[i]);
    }
  }
}


//-----------------------------------------------------------------------------

vtkDataArray *
ConduitArrayToVTKDataArray(const conduit::Node &n)
{
  vtkDataArray *retval = NULL;
  
  int nchildren = n.number_of_children();
  int ntuples = 0;
  int ncomps  = 1;


  conduit::DataType vals_dtype;

  if(nchildren > 0) // n is a mcarray w/ children that hold the vals
  {
    conduit::Node v_info;
    if(!conduit::blueprint::mcarray::verify(n,v_info))
    {
      SENSEI_ERROR("Node is not a mcarray " << v_info.to_json())
    }
        
    // in this case, each child is a component of the array
    ncomps = nchildren;
    // This assumes all children have the same leaf type
    vals_dtype = n[0].dtype();
  }
  else // n is an array, holds the vals
  {
    vals_dtype = n.dtype();
  }
    
  // get the number of tuples
  ntuples = (int) vals_dtype.number_of_elements();
    
  std::cout<< "VTKDataArray num_tuples = " << ntuples << " " 
                  << " num_comps = " << ncomps << std::endl;;

  if (vals_dtype.is_unsigned_char())
  {
    retval = vtkUnsignedCharArray::New();
    Blueprint_MultiCompArray_To_VTKDataArray<CONDUIT_NATIVE_UNSIGNED_CHAR>(n,
                                                                           ncomps,
                                                                           ntuples,
                                                                           retval);
  }
  else if (vals_dtype.is_unsigned_short())
  {
    retval = vtkUnsignedShortArray::New();
    Blueprint_MultiCompArray_To_VTKDataArray<CONDUIT_NATIVE_UNSIGNED_SHORT>(n,
                                                                            ncomps,
                                                                            ntuples,
                                                                            retval);
  }
  else if (vals_dtype.is_unsigned_int())
  {
    retval = vtkUnsignedIntArray::New();
    Blueprint_MultiCompArray_To_VTKDataArray<CONDUIT_NATIVE_UNSIGNED_INT>(n,
                                                                          ncomps,
                                                                          ntuples,
                                                                          retval);
  }
  else if (vals_dtype.is_char())
  {
    retval = vtkCharArray::New();
    Blueprint_MultiCompArray_To_VTKDataArray<CONDUIT_NATIVE_CHAR>(n,
                                                                  ncomps,
                                                                  ntuples,
                                                                  retval);

  }
  else if (vals_dtype.is_short())
  {
    retval = vtkShortArray::New();
    Blueprint_MultiCompArray_To_VTKDataArray<CONDUIT_NATIVE_SHORT>(n,
                                                                   ncomps,
                                                                   ntuples,
                                                                   retval);
  }
  else if (vals_dtype.is_int())
  {
    retval = vtkIntArray::New();
    Blueprint_MultiCompArray_To_VTKDataArray<CONDUIT_NATIVE_INT>(n,
                                                                 ncomps,
                                                                 ntuples,
                                                                 retval);
  }
  else if (vals_dtype.is_long())
  {
    retval = vtkLongArray::New();
    Blueprint_MultiCompArray_To_VTKDataArray<CONDUIT_NATIVE_LONG>(n,
                                                                  ncomps,
                                                                  ntuples,
                                                                  retval);
  }
  else if (vals_dtype.is_float())
  {
    retval = vtkFloatArray::New();
    Blueprint_MultiCompArray_To_VTKDataArray<CONDUIT_NATIVE_FLOAT>(n,
                                                                   ncomps,
                                                                   ntuples,
                                                                   retval);
  }
  else if (vals_dtype.is_double())
  {
    retval = vtkDoubleArray::New();
    Blueprint_MultiCompArray_To_VTKDataArray<CONDUIT_NATIVE_DOUBLE>(n,
                                                                    ncomps,
                                                                    ntuples,
                                                                    retval);
  }
  else
  {
    SENSEI_ERROR("Conduit Array to VTK Data Array:  unsupported data type: " << n.dtype().name())
  }
  return retval;
}

//-----------------------------------------------------------------------------

vtkCellArray *
HomogeneousShapeTopologyToVTKCellArray(const conduit::Node &n_topo,
                                       int npts)
{
  vtkCellArray *ca = vtkCellArray::New();
  vtkIdTypeArray *ida = vtkIdTypeArray::New();

  int ctype = ElementShapeNameToVTKCellType(n_topo["elements/shape"].as_string());
  int csize = VTKCellTypeSize(ctype);
  int ncells = n_topo["elements/connectivity"].dtype().number_of_elements() / csize;

    conduit::int_array topo_conn;
    ida->SetNumberOfTuples(ncells * (csize + 1));
    for (int i = 0 ; i < ncells; i++)
    {
      conduit::Node n_tmp;
      if(n_topo["elements/connectivity"].dtype().is_int())
      {
        topo_conn = n_topo["elements/connectivity"].as_int_array();
      }
      else
      {
        n_topo["elements/connectivity"].to_int_array(n_tmp);
        topo_conn = n_tmp.as_int_array();
      }
            
      ida->SetComponent((csize+1)*i, 0, csize);
      for (int j = 0; j < csize; j++)
      {
        ida->SetComponent((csize+1)*i+j+1, 0,topo_conn[i*csize+j]);
      }
  }
  ca->SetCells(ncells, ida);
  ida->Delete();
  return ca;
}

//-----------------------------------------------------------------------------

vtkPoints *
ExplicitCoordsToVTKPoints(const conduit::Node &coords)
{
  vtkPoints *points = vtkPoints::New();
        
  const conduit::Node &vals = coords["values"];
    
  // We always use doubles

  int npts = (int) vals["x"].dtype().number_of_elements();
    
  conduit::double_array x_vals;
  conduit::double_array y_vals;
  conduit::double_array z_vals;
    
  bool have_y = false;
  bool have_z = false;
    
  conduit::Node vals_double;
    
  if(!vals["x"].dtype().is_double())
  {
    vals["x"].to_double_array(vals_double["x"]);
    x_vals = vals_double["x"].value();
  }
  else
  {
    x_vals = vals["x"].value();
  }
    
    
  if(vals.has_child("y"))
  {
    have_y = true;
        
    if(!vals["y"].dtype().is_double())
    {
      vals["y"].to_double_array(vals_double["y"]);
      y_vals = vals_double["y"].value();
    }
    else
    {
      y_vals = vals["y"].value();
    }
  }
    
  if(vals.has_child("z"))
  {
    have_z = true;
       
    if(!vals["z"].dtype().is_double())
    {
      vals["z"].to_double_array(vals_double["z"]);
      z_vals = vals_double["z"].value();
    }
    else
    {
      z_vals = vals["z"].value();
    }
  }


  points->SetDataTypeToDouble();
  points->SetNumberOfPoints(npts);

  //TODO: we could describe the VTK data array via 
  // and push the conversion directly into its memory. 

  for (vtkIdType i = 0; i < npts; i++)
  {
    double x = x_vals[i];
    double y = have_y ? y_vals[i] : 0;
    double z = have_z ? z_vals[i] : 0;
    points->SetPoint(i, x, y, z);
  }

  return points;
}


//-----------------------------------------------------------------------------
vtkDataSet* StructuredMesh(const conduit::Node* node)
{
  vtkStructuredGrid *sgrid = vtkStructuredGrid::New();
  const conduit::Node &coords = (*node)["coordsets"][0];
  const conduit::Node &topo   = (*node)["topologies"][0];

  int dims[3];
  dims[0] = topo.has_path("elements/dims/i") ? topo["elements/dims/i"].to_int()+1 : 1;
  dims[1] = topo.has_path("elements/dims/j") ? topo["elements/dims/j"].to_int()+1 : 1;
  dims[2] = topo.has_path("elements/dims/k") ? topo["elements/dims/k"].to_int()+1 : 1;
  sgrid->SetDimensions(dims);

  vtkPoints *points = ExplicitCoordsToVTKPoints(coords);
  sgrid->SetPoints(points);
  points->Delete();

  return sgrid;
}

//-----------------------------------------------------------------------------

vtkDataSet* UnstructuredMesh(const conduit::Node* node)
{
    
  const conduit::Node &coords = (*node)["coordsets"][0];
  const conduit::Node &topo   = (*node)["topologies"][0];
  
  vtkPoints *points = ExplicitCoordsToVTKPoints(coords);

  vtkUnstructuredGrid *ugrid = vtkUnstructuredGrid::New();
  ugrid->SetPoints(points);
  points->Delete();

  //
  // Now, add explicit topology
  //
  vtkCellArray *ca = HomogeneousShapeTopologyToVTKCellArray(topo, points->GetNumberOfPoints());
  ugrid->SetCells(ElementShapeNameToVTKCellType(topo["elements/shape"].as_string()), ca);
  ca->Delete();
    
  return ugrid;

}


//-----------------------------------------------------------------------------

vtkDataSet* RectilinearMesh(const conduit::Node* node)
{
  vtkRectilinearGrid *rectgrid = vtkRectilinearGrid::New();

  const conduit::Node &coords         = (*node)["coordsets"][0];
  const conduit::Node &coords_values  = coords["values"];

  int dims[3] = {1,1,1};

  dims[0] = coords_values["x"].dtype().number_of_elements();
  if (coords_values.has_child("y"))
    dims[1] = coords_values["y"].dtype().number_of_elements();
  if (coords_values.has_child("z"))
    dims[2] = coords_values["z"].dtype().number_of_elements();
  rectgrid->SetDimensions(dims);

  vtkDataArray *vtk_coords[3] = {0,0,0};
  vtk_coords[0] = ConduitArrayToVTKDataArray(coords_values["x"]);
  if (coords_values.has_child("y"))
    vtk_coords[1] = ConduitArrayToVTKDataArray(coords_values["y"]);
  else
  {
    vtk_coords[1] = vtk_coords[0]->NewInstance();
    vtk_coords[1]->SetNumberOfTuples(1);
    vtk_coords[1]->SetComponent(0,0,0);
  }
  if (coords_values.has_child("z"))
    vtk_coords[2] = ConduitArrayToVTKDataArray(coords_values["z"]);
  else
  {
    vtk_coords[2] = vtk_coords[0]->NewInstance();
    vtk_coords[2]->SetNumberOfTuples(1);
    vtk_coords[2]->SetComponent(0,0,0);
  }

  rectgrid->SetXCoordinates(vtk_coords[0]);
  rectgrid->SetYCoordinates(vtk_coords[1]);
  rectgrid->SetZCoordinates(vtk_coords[2]);

  vtk_coords[0]->Delete();
  vtk_coords[1]->Delete();
  vtk_coords[2]->Delete();

  return rectgrid;

}

//-----------------------------------------------------------------------------

vtkDataSet* UniformMesh(const conduit::Node* node)
{
  vtkRectilinearGrid *rectgrid = vtkRectilinearGrid::New();
  const conduit::Node &coords = (*node)["coordsets"][0];
  int dims[3];
  dims[0]  = coords["dims"].has_child("i") ? coords["dims/i"].to_int() : 1;
  dims[1]  = coords["dims"].has_child("j") ? coords["dims/j"].to_int() : 1;
  dims[2]  = coords["dims"].has_child("k") ? coords["dims/k"].to_int() : 1;
  rectgrid->SetDimensions(dims);

  double spacing[3];
  spacing[0] = coords["spacing"].has_child("dx") ? coords["spacing/dx"].to_double(): 0; 
  spacing[1] = coords["spacing"].has_child("dy") ? coords["spacing/dy"].to_double(): 0; 
  spacing[2] = coords["spacing"].has_child("dz") ? coords["spacing/dz"].to_double(): 0; 
  double origin[3];
  origin[0]  = coords["origin"].has_child("x") ? coords["origin/x"].to_double() : 0;
  origin[1]  = coords["origin"].has_child("y") ? coords["origin/y"].to_double() : 0;
  origin[2]  = coords["origin"].has_child("z") ? coords["origin/z"].to_double() : 0;

  for (int i = 0; i < 3; i++)
  {
     vtkDataArray *da = NULL;
     conduit::DataType dt = conduit::DataType::c_double();
        // we have we origin, we can infer type from it
    if(coords.has_path("origin/x"))
    {
      dt = coords["origin"]["x"].dtype();
    }

    // since vtk uses the c-native style types
    // only need to check for native types in conduit
    if (dt.is_unsigned_char())
      da = vtkUnsignedCharArray::New();
    else if (dt.is_unsigned_short())
      da = vtkUnsignedShortArray::New();
    else if (dt.is_unsigned_int())
      da = vtkUnsignedIntArray::New();
    else if (dt.is_char())
      da = vtkCharArray::New();
    else if (dt.is_short())
      da = vtkShortArray::New();
    else if (dt.is_int())
      da = vtkIntArray::New();
    else if (dt.is_long())
      da = vtkLongArray::New();
    else if (dt.is_float())
      da = vtkFloatArray::New();
    else if (dt.is_double())
      da = vtkDoubleArray::New();
    else
    {
      SENSEI_ERROR("Conduit Blueprint to Rectilinear Grid coordinates, unsupported data type: " << dt.name())
    }

    da->SetNumberOfTuples(dims[i]);
    double x = origin[i];
    for (int j = 0; j < dims[i]; j++, x += spacing[i])
      da->SetComponent(j, 0, x);

    if (i == 0) rectgrid->SetXCoordinates(da);
    if (i == 1) rectgrid->SetYCoordinates(da);
    if (i == 2) rectgrid->SetZCoordinates(da);
    da->Delete();
  }
  return rectgrid;
}


//-----------------------------------------------------------------------------
void ConduitDataAdaptor::Initialize(conduit::Node* in_node)
{
  this->Node = in_node;
  ConduitDataAdaptor::UpdateFields();
  return;
}


//-----------------------------------------------------------------------------
int ConduitDataAdaptor::GetNumberOfArrays(const std::string &meshName,
                                          int association,
                                          unsigned int &numberOfArrays)
{

  auto search = this->FieldNames.find(meshName);
  if(search == this->FieldNames.end())
  {
    SENSEI_ERROR("GetNumberOfArrays: Mesh " << meshName << " Cannot Be Found")
    return 1;
  }
 
  std::vector<std::string> vec = search->second;

  numberOfArrays = vec.size();
 
  return 0;
}

//-----------------------------------------------------------------------------

int ConduitDataAdaptor::GetArrayName(const std::string &meshName,
                                     int association, 
                                     unsigned int index,
                                     std::string &arrayName)
{

  auto search = this->FieldNames.find(meshName);
  if(search == this->FieldNames.end())
  {
    SENSEI_ERROR("GetArrayName: Mesh " << meshName << " Cannot Be Found")
    return -1;
  }

  std::vector<std::string> vec = search->second;

  if(index > (unsigned int) vec.size())
    return -1;

  arrayName = vec.at(index);

  return 0;
}

//-----------------------------------------------------------------------------

void ConduitDataAdaptor::SetNode(conduit::Node* node)
{
  this->Node = node;
  ConduitDataAdaptor::UpdateFields();
  return;
}


//-----------------------------------------------------------------------------

void ConduitDataAdaptor::UpdateFields()
{
  //in_node->print();
  this->FieldNames.clear();
  if(conduit::blueprint::mesh::is_multi_domain(*this->Node))
  {
    conduit::NodeConstIterator doms_itr = (this->Node)->children();
    while(doms_itr.has_next())
    {
      const conduit::Node& d_node = doms_itr.next();
      conduit::NodeConstIterator field_itr = d_node["fields"].children();
      while(field_itr.has_next())
      {
        const conduit::Node& field = field_itr.next();
        std::string field_name = field_itr.name();
                
        //TODO: There is no formal protocol for naming meshes within the node
        //This is a placeholder until there is a path in a node that 
        //distinguishes what mesh the data belongs to. 
        //For now, the meshName will be hardcoded as "mesh".
        //std::string meshName = field["mesh"].as_string();
        //std::string meshName = field["topology"].as_string();

        std::string meshName = "mesh";
        auto searchMesh = this->FieldNames.find(meshName);
        if(searchMesh != this->FieldNames.end())
        {
          std::vector<std::string> vec = searchMesh->second;
          std::vector<std::string>::iterator itr;
          itr = find(vec.begin(), vec.end(), field_name);
     
          if(itr == vec.end())
          {
             vec.push_back(field_name);
             this->FieldNames.erase(meshName);
             this->FieldNames.insert(std::pair<std::string, std::vector<std::string>>(meshName, vec));
          }
        }
        else
        {
           std::vector<std::string> vec;
           vec.push_back(field_name);
           this->FieldNames.insert(std::pair<std::string, std::vector<std::string>>(meshName, vec));

        }
      }
    }
  }
  else
  {
    const conduit::Node& fields = (*this->Node)["fields"];
    conduit::NodeConstIterator fields_itr = fields.children();

    while(fields_itr.has_next())
    {   
      const conduit::Node& field = fields_itr.next();
      std::string field_name = fields_itr.name();
      std::string meshName = field["topology"].as_string();

      auto searchMesh = this->FieldNames.find(meshName);
      if(searchMesh != this->FieldNames.end())
      {
        std::vector<std::string> vec = searchMesh->second;
        std::vector<std::string>::iterator itr;
        itr = find(vec.begin(), vec.end(), field_name);
     
        if(itr == vec.end())
        {
           vec.push_back(field_name);
           this->FieldNames.erase(meshName);
           this->FieldNames.insert(std::pair<std::string, std::vector<std::string>>(meshName, vec));
        }
      }
      else
      {
        std::vector<std::string> vec;
        vec.push_back(field_name);
        this->FieldNames.insert(std::pair<std::string, std::vector<std::string>>(meshName, vec));
      }
    }
  }
  return;
}
//-----------------------------------------------------------------------------

int ConduitDataAdaptor::GetMesh(const std::string &meshName,
                                bool structureOnly, 
                                vtkDataObject *&mesh)
{   
  auto search = this->FieldNames.find(meshName);
  if(search == this->FieldNames.end())
  {
    SENSEI_ERROR("GetMesh: Mesh " << meshName << " Cannot Be Found")
    return -1;
  }

  vtkMultiBlockDataSet *mb_mesh = vtkMultiBlockDataSet::New();

  int start = 0, total_blocks = 0;
  int size, rank;
  MPI_Comm_size(this->GetCommunicator(), &size);
  MPI_Comm_rank(this->GetCommunicator(), &rank);

  int global_blocks[size] = {};

  if(conduit::blueprint::mesh::is_multi_domain(*this->Node))
  {
    int local_blocks[size]  = {};
    local_blocks[rank] = this->Node->number_of_children();
    MPI_Allreduce(&local_blocks, &global_blocks, size, MPI_INT, MPI_SUM, this->GetCommunicator()); 

    this->GlobalBlockDistribution = (int*)malloc(sizeof(int)*size);
    memcpy(this->GlobalBlockDistribution, global_blocks, sizeof(int)*size);
        
    for(int i = 0; i < size; i++)
    {
      total_blocks += global_blocks[i];
      if(i < rank)
        start += global_blocks[i];
    }


    mb_mesh->SetNumberOfBlocks(total_blocks);
    int domain = 0;
    conduit::NodeConstIterator domain_itr = this->Node->children();
       
    while(domain_itr.has_next())
    {
      int block = domain + start;
      const conduit::Node &d_node = domain_itr.next();
      const conduit::Node &coords = d_node["coordsets"][0];
      const conduit::Node &topo   = d_node["topologies"][0];
            
      if (coords["type"].as_string() == "uniform")
      {
        mb_mesh->SetBlock(block, UniformMesh(&d_node));
      }
      else if (coords["type"].as_string() == "rectilinear")
      {
        mb_mesh->SetBlock(block, RectilinearMesh(&d_node));
      }   
      else if (coords["type"].as_string() == "explicit")
      {
        if (topo["type"].as_string() == "structured")
        {
          mb_mesh->SetBlock(block, StructuredMesh(&d_node)); 
        }
        else
        {
          mb_mesh->SetBlock(block, UnstructuredMesh(&d_node));
        }
      }
      domain++;
    }
  }
  else
  {
    int local_blocks[size]  = {};
    local_blocks[rank] = 1;
    MPI_Allreduce(&local_blocks, &global_blocks, size, MPI_INT, MPI_SUM, this->GetCommunicator()); 
        
    this->GlobalBlockDistribution = (int*)malloc(sizeof(int)*size);
    memcpy(this->GlobalBlockDistribution, global_blocks, sizeof(int)*size);
   
    for(int i = 0; i < size; i++)
    {
      total_blocks += global_blocks[i];
      if(i < rank)
        start += global_blocks[i];
    }

    mb_mesh->SetNumberOfBlocks(total_blocks);

    int block = start;
    const conduit::Node &coords = (*this->Node)["coordsets"][0];
    const conduit::Node &topo   = (*this->Node)["topologies"][0];
    if (coords["type"].as_string() == "uniform")
    {
       mb_mesh->SetBlock(block, UniformMesh(this->Node));
    }
    else if (coords["type"].as_string() == "rectilinear")
    {
      mb_mesh->SetBlock(block, RectilinearMesh(this->Node));
    }   
    else if (coords["type"].as_string() == "explicit")
    {
      if (topo["type"].as_string() == "structured")
      {
        mb_mesh->SetBlock(block, StructuredMesh(this->Node)); 
      }
      else
      {
        mb_mesh->SetBlock(block, UnstructuredMesh(this->Node));
      }
    }
  }
  mesh = mb_mesh;
  return 0;
}


//-----------------------------------------------------------------------------
int ConduitDataAdaptor::GetNumberOfMeshes(unsigned int &numberOfMeshes)
{
  //Once multiple meshes are present in the same node; you can use 
  //this->FieldNames.size(). The size of the map corresponds to how many 
  //different meshes. 
  numberOfMeshes = 1;
  return 0;
}


//-----------------------------------------------------------------------------
int ConduitDataAdaptor::GetMeshName(unsigned int id, std::string &meshName)
{
  //this->FieldNames can be used to find the mesh name, once that becomes 
  //supported in the conduit node. 
  meshName = "mesh";
  return 0;
}

//-----------------------------------------------------------------------------
int ConduitDataAdaptor::AddArray(vtkDataObject* mesh,
                                 const std::string &meshName, 
                                 int association,
                                 const std::string& arrayname)
{

  auto search = this->FieldNames.find(meshName);
  if(search == this->FieldNames.end())
  {
    SENSEI_ERROR("AddArray: Mesh " << meshName << " Cannot Be Found")
  }

  std::vector<std::string> vec = search->second;
  int flag = 1;
  for(int i = 0; i < (int) vec.size(); i++)
  {
    if(arrayname.compare(vec.at(i)) == 0)
    {
      flag = 0;
      break;
    }
    else
    {
      flag = 1; 
    }
  }
  if(flag)
  {
    SENSEI_ERROR("ERROR: field " << arrayname << " does not reside on Mesh ")
    return -1;
  }

  int rank, start = 0;
  MPI_Comm_rank(this->GetCommunicator(), &rank);
  
  for(int i = 0; i < rank; i++)
    start += this->GlobalBlockDistribution[i];

  vtkMultiBlockDataSet *mb = dynamic_cast<vtkMultiBlockDataSet*>(mesh);
  
  if(!mb)
  {
    SENSEI_ERROR("MultiBlockDataSet is NULl")
    return -1;
  }
  if(conduit::blueprint::mesh::is_multi_domain(*this->Node))
  {
    conduit::NodeConstIterator dom_itr = this->Node->children();
    int domain = 0;
    while(dom_itr.has_next())
    {
      const conduit::Node& d_node = dom_itr.next();
      const conduit::Node& fields  = d_node["fields"];
      const conduit::Node& field   = fields[arrayname];
      const conduit::Node& values  = field["values"];
            
      vtkSmartPointer<vtkDataArray> array = ConduitArrayToVTKDataArray(values);
      array->SetName(arrayname.c_str());
       
      vtkDataObject *block = mb->GetBlock(start + domain);

      std::stringstream ss;
      ss << "fields/" << arrayname << "/association";
      std::string association_path  = ss.str();
      std::string field_association = d_node[association_path].as_string();

      std::stringstream rr;
      rr << "fields/" << arrayname << "/values";
      std::string values_path = rr.str();
      int nchildren = d_node[values_path].number_of_children();

      if(field_association == "vertex")
      {
        if(nchildren > 0)
        {
          vtkDataSetAttributes *attr = block->GetAttributes(vtkDataObject::POINT);
          attr->AddArray(array);
        }
        else
        {
          vtkDataSetAttributes *attr = block->GetAttributes(vtkDataObject::POINT);
          attr->AddArray(array);
        }
      }  
      else if (field_association == "element")
      {
        if(nchildren > 0)
        {
          vtkDataSetAttributes *attr = block->GetAttributes(vtkDataObject::CELL);
          attr->AddArray(array);
        } 
        else
        {
          vtkDataSetAttributes *attr = block->GetAttributes(vtkDataObject::CELL);
          attr->AddArray(array);
        }
      }
      else
      {
        SENSEI_ERROR("ERROR: association of type " << field_association << " incompatible")
        return -1;
      }
      domain++;
    }
  }  
  else
  {
    const conduit::Node& fields  = (*this->Node)["fields"];
    const conduit::Node& field   = fields[arrayname];
    const conduit::Node& values  = field["values"];
    vtkSmartPointer<vtkDataArray> array = ConduitArrayToVTKDataArray(values);
    array->SetName(arrayname.c_str());
       
    vtkDataObject *block = mb->GetBlock(start);

    std::stringstream ss;
    ss << "fields/" << arrayname << "/association";
    std::string association_path  = ss.str();
    std::string field_association = (*this->Node)[association_path].as_string();

    std::stringstream rr;
    rr << "fields/" << arrayname << "/values";
    std::string values_path = rr.str();
    int nchildren = (*this->Node)[values_path].number_of_children();

    if(field_association == "vertex")
    {
      if(nchildren > 0)
      {
        vtkDataSetAttributes *attr = block->GetAttributes(vtkDataObject::POINT);
        attr->AddArray(array);
      }  
      else
      {
        vtkDataSetAttributes *attr = block->GetAttributes(vtkDataObject::POINT);
        attr->AddArray(array);
      }
    }
    else if (field_association == "element")
    {
      if(nchildren > 0)
      {
        vtkDataSetAttributes *attr = block->GetAttributes(vtkDataObject::CELL);
        attr->AddArray(array);
      } 
      else
      {
        vtkDataSetAttributes *attr = block->GetAttributes(vtkDataObject::CELL);
        attr->AddArray(array);
      }  
    }
    else
    {
      SENSEI_ERROR("ERROR: association of type " << field_association << " incompatible")
      return -1;
    }
  } 
  return 0;
}

//-----------------------------------------------------------------------------
int ConduitDataAdaptor::ReleaseData()
{
  this->Node = NULL;
  this->FieldNames.clear();
  free(this->GlobalBlockDistribution);

  return 0;
}

}
