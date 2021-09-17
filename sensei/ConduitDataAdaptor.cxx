#include <cassert>
#include <sstream>

#include <conduit_blueprint.hpp>

#include <svtkCharArray.h>
#include <svtkUnsignedCharArray.h>
#include <svtkShortArray.h>
#include <svtkUnsignedShortArray.h>
#include <svtkIntArray.h>
#include <svtkUnsignedIntArray.h>
#include <svtkLongArray.h>
#include <svtkUnsignedLongArray.h>
#include <svtkLongLongArray.h>
#include <svtkUnsignedLongLongArray.h>
#include <svtkFloatArray.h>
#include <svtkDoubleArray.h>

#include <svtkDataSetAttributes.h>
#include <svtkImageData.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkObjectFactory.h>
#include <svtkSmartPointer.h>
#include <svtkVector.h>
#include <svtkVectorOperators.h>
#include <svtkCellArray.h>
#include <svtkCellType.h>
#include <svtkCellData.h>
#include <svtkIdTypeArray.h>
#include <svtkPoints.h>
#include <svtkPointData.h>

#include <svtkRectilinearGrid.h>
#include <svtkStructuredGrid.h>
#include <svtkUnstructuredGrid.h>

#include "ConduitDataAdaptor.h"
#include "Error.h"
// compile error Timer.h not found 4/17/2021, wes removed it
// Timer.h shows up under clang (11.0.X), and vtk-m
// #include "Timer.h"

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
void ConduitDataAdaptor::PrintSelf(ostream &os, vtkIndent indent)
{
  os << indent << indent << "Internal conduit node: " << this->Node->to_json_default() << std::endl;
  Superclass::PrintSelf(os, indent);
}

//-----------------------------------------------------------------------------
static inline int ElementShapeNameToVTKCellType( const std::string &shape_name )
{
  if (shape_name == "point") return SVTK_VERTEX;
  if (shape_name == "line")  return SVTK_LINE;
  if (shape_name == "tri")   return SVTK_TRIANGLE;
  if (shape_name == "quad")  return SVTK_QUAD;
  if (shape_name == "hex")   return SVTK_HEXAHEDRON;
  if (shape_name == "tet")   return SVTK_TETRA;

  SENSEI_ERROR("Warning: Unsupported Element Shape: " << shape_name);
  return( SVTK_EMPTY_CELL );
}

//-----------------------------------------------------------------------------
static inline int SVTKCellTypeSize( int cell_type )
{
  if (cell_type == SVTK_VERTEX)     return 1;
  if (cell_type == SVTK_LINE)       return 2;
  if (cell_type == SVTK_TRIANGLE)   return 3;
  if (cell_type == SVTK_QUAD)       return 4;
  if (cell_type == SVTK_HEXAHEDRON) return 8;
  if (cell_type == SVTK_TETRA)      return 4;

  return 0;
}

//-----------------------------------------------------------------------------
template<typename T> void Blueprint_MultiCompArray_To_SVTKDataArray( const conduit::Node &n, int ncomps, int ntuples, svtkDataArray *darray )
{
  // svtk reqs us to set number of comps before number of tuples
  if( ncomps == 2 ) // we need 3 comps for vectors
    darray->SetNumberOfComponents( 3 );
  else
    darray->SetNumberOfComponents( ncomps );
  // set number of tuples
  darray->SetNumberOfTuples( ntuples );
  if( n.number_of_children() > 0 )
  {
    // handle multi-component case
    for(svtkIdType c=0; c < ncomps ;++c)
    {
      conduit::DataArray<T> vals_array = n[c].value();

      for(svtkIdType i = 0; i < ntuples ;++i)
      {
        darray->SetComponent( i, c, (double)vals_array[i] );

        if( ncomps == 2 )
        {
          darray->SetComponent( i, 2, 0.0 );
        }
      }
    }
  }
  else
  {
    // single array case
    conduit::DataArray<T> vals_array = n.value();

    for(svtkIdType i = 0; i < ntuples ;++i)
    {
      darray->SetComponent( i, 0, (double)vals_array[i] );
    }
  }
}

//-----------------------------------------------------------------------------
svtkDataArray * ConduitArrayToSVTKDataArray( const conduit::Node &n )
{
  svtkDataArray *retval = NULL;
  
  int nchildren = n.number_of_children();
  int ntuples = 0;
  int ncomps  = 1;
  conduit::DataType vals_dtype;

  if( nchildren > 0 ) // n is a mcarray w/ children that hold the vals
  {
    conduit::Node v_info;
    if( !conduit::blueprint::mcarray::verify(n, v_info) )
    {
      SENSEI_ERROR( "Node is not a mcarray " << v_info.to_json() );
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
  if( vals_dtype.is_unsigned_char() )
  {
    retval = svtkUnsignedCharArray::New();
    Blueprint_MultiCompArray_To_SVTKDataArray<CONDUIT_NATIVE_UNSIGNED_CHAR>( n, ncomps, ntuples, retval );
  }
  else if( vals_dtype.is_unsigned_short() )
  {
    retval = svtkUnsignedShortArray::New();
    Blueprint_MultiCompArray_To_SVTKDataArray<CONDUIT_NATIVE_UNSIGNED_SHORT>( n, ncomps, ntuples, retval );
  }
  else if( vals_dtype.is_unsigned_int() )
  {
    retval = svtkUnsignedIntArray::New();
    Blueprint_MultiCompArray_To_SVTKDataArray<CONDUIT_NATIVE_UNSIGNED_INT>( n, ncomps, ntuples, retval );
  }
  else if( vals_dtype.is_char() )
  {
    retval = svtkCharArray::New();
    Blueprint_MultiCompArray_To_SVTKDataArray<CONDUIT_NATIVE_CHAR>( n, ncomps, ntuples, retval );
  }
  else if( vals_dtype.is_short() )
  {
    retval = svtkShortArray::New();
    Blueprint_MultiCompArray_To_SVTKDataArray<CONDUIT_NATIVE_SHORT>( n, ncomps, ntuples, retval );
  }
  else if( vals_dtype.is_int() )
  {
    retval = svtkIntArray::New();
    Blueprint_MultiCompArray_To_SVTKDataArray<CONDUIT_NATIVE_INT>( n, ncomps, ntuples, retval );
  }
  else if( vals_dtype.is_long() )
  {
    retval = svtkLongArray::New();
    Blueprint_MultiCompArray_To_SVTKDataArray<CONDUIT_NATIVE_LONG>( n, ncomps, ntuples, retval );
  }
  else if( vals_dtype.is_float() )
  {
    retval = svtkFloatArray::New();
    Blueprint_MultiCompArray_To_SVTKDataArray<CONDUIT_NATIVE_FLOAT>( n, ncomps, ntuples, retval );
  }
  else if( vals_dtype.is_double() )
  {
    retval = svtkDoubleArray::New();
    Blueprint_MultiCompArray_To_SVTKDataArray<CONDUIT_NATIVE_DOUBLE>( n, ncomps, ntuples, retval );
  }
  else
  {
    SENSEI_ERROR( "Conduit Array to SVTK Data Array:  unsupported data type: " << n.dtype().name() );
  }
  return( retval );
}

//-----------------------------------------------------------------------------
svtkCellArray * HomogeneousShapeTopologyToSVTKCellArray( const conduit::Node &n_topo, int /*npts*/ )
{
  svtkCellArray *ca = svtkCellArray::New();
  svtkIdTypeArray *ida = svtkIdTypeArray::New();

  int ctype = ElementShapeNameToSVTKCellType(n_topo["elements/shape"].as_string());
  int csize = SVTKCellTypeSize(ctype);
  int ncells = n_topo["elements/connectivity"].dtype().number_of_elements() / csize;

    conduit::int_array topo_conn;
    ida->SetNumberOfTuples(ncells * (csize + 1));
    for (int i=0; i < ncells ;++i)
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
      for (int j=0; j < csize ;++j)
      {
        ida->SetComponent((csize+1)*i+j+1, 0,topo_conn[i*csize+j]);
      }
  }
  ca->SetCells(ncells, ida);
  ida->Delete();
  return ca;
}

//-----------------------------------------------------------------------------
svtkPoints * ExplicitCoordsToSVTKPoints( const conduit::Node &coords )
{
  svtkPoints *points = svtkPoints::New();

  const conduit::Node &vals = coords["values"];

  // We always use doubles
  int npts = (int) vals["x"].dtype().number_of_elements();

  conduit::double_array x_vals;
  conduit::double_array y_vals;
  conduit::double_array z_vals;

  bool have_y = false;
  bool have_z = false;

  conduit::Node vals_double;

  if( !vals["x"].dtype().is_double() )
  {
    vals["x"].to_double_array( vals_double["x"] );
    x_vals = vals_double["x"].value();
  }
  else
  {
    x_vals = vals["x"].value();
  }

  if( vals.has_child("y") )
  {
    have_y = true;
    if( !vals["y"].dtype().is_double() )
    {
      vals["y"].to_double_array( vals_double["y"] );
      y_vals = vals_double["y"].value();
    }
    else
    {
      y_vals = vals["y"].value();
    }
  }

  if( vals.has_child("z") )
  {
    have_z = true;

    if( !vals["z"].dtype().is_double() )
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

  //TODO: we could describe the SVTK data array via 
  // and push the conversion directly into its memory. 

  for(svtkIdType i = 0; i < npts ;++i)
  {
    double x = x_vals[i];
    double y = have_y ? y_vals[i] : 0;
    double z = have_z ? z_vals[i] : 0;
    points->SetPoint(i, x, y, z);
  }

  return( points );
}
//-----------------------------------------------------------------------------
svtkDataSet* StructuredMesh( const conduit::Node* node )
{
  svtkStructuredGrid *sgrid = svtkStructuredGrid::New();
  const conduit::Node &coords = (*node)["coordsets"][0];
  const conduit::Node &topo   = (*node)["topologies"][0];

  int dims[3];
  dims[0] = topo.has_path("elements/dims/i") ? topo["elements/dims/i"].to_int()+1 : 1;
  dims[1] = topo.has_path("elements/dims/j") ? topo["elements/dims/j"].to_int()+1 : 1;
  dims[2] = topo.has_path("elements/dims/k") ? topo["elements/dims/k"].to_int()+1 : 1;
  sgrid->SetDimensions( dims );

  svtkPoints *points = ExplicitCoordsToSVTKPoints(coords);
  sgrid->SetPoints( points );
  points->Delete();

  return( sgrid );
}

//-----------------------------------------------------------------------------
svtkDataSet* UnstructuredMesh( const conduit::Node* node )
{
  const conduit::Node &coords = (*node)["coordsets"][0];
  const conduit::Node &topo   = (*node)["topologies"][0];

  svtkPoints *points = ExplicitCoordsToSVTKPoints( coords );

  svtkUnstructuredGrid *ugrid = svtkUnstructuredGrid::New();
  ugrid->SetPoints( points );
  points->Delete();

  //
  // Now, add explicit topology
  //
  svtkCellArray *ca = HomogeneousShapeTopologyToSVTKCellArray( topo, points->GetNumberOfPoints() );
  ugrid->SetCells( ElementShapeNameToSVTKCellType(topo["elements/shape"].as_string()), ca );
  ca->Delete();
  return( ugrid );
}

//-----------------------------------------------------------------------------
svtkDataSet* RectilinearMesh( const conduit::Node* node )
{
  svtkRectilinearGrid *rectgrid = svtkRectilinearGrid::New();

  const conduit::Node &coords         = (*node)["coordsets"][0];
  const conduit::Node &coords_values  = coords["values"];

  int dims[3] = {1, 1, 1};

  dims[0] = coords_values["x"].dtype().number_of_elements();
  if( coords_values.has_child("y") )
    dims[1] = coords_values["y"].dtype().number_of_elements();
  if( coords_values.has_child("z") )
    dims[2] = coords_values["z"].dtype().number_of_elements();
  rectgrid->SetDimensions( dims );

  svtkDataArray *svtk_coords[3] = {0, 0, 0};
  svtk_coords[0] = ConduitArrayToSVTKDataArray( coords_values["x"] );
  if( coords_values.has_child("y") )
    svtk_coords[1] = ConduitArrayToSVTKDataArray( coords_values["y"] );
  else
  {
    svtk_coords[1] = svtk_coords[0]->NewInstance();
    svtk_coords[1]->SetNumberOfTuples( 1 );
    svtk_coords[1]->SetComponent( 0, 0, 0 );
  }
  if( coords_values.has_child("z") )
    svtk_coords[2] = ConduitArrayToSVTKDataArray( coords_values["z"]) ;
  else
  {
    svtk_coords[2] = svtk_coords[0]->NewInstance();
    svtk_coords[2]->SetNumberOfTuples( 1 );
    svtk_coords[2]->SetComponent( 0, 0, 0 );
  }

  rectgrid->SetXCoordinates( svtk_coords[0] );
  rectgrid->SetYCoordinates( svtk_coords[1] );
  rectgrid->SetZCoordinates( svtk_coords[2] );

  svtk_coords[0]->Delete();
  svtk_coords[1]->Delete();
  svtk_coords[2]->Delete();

  return( rectgrid );
}

//-----------------------------------------------------------------------------

svtkDataSet* UniformMesh( const conduit::Node* node )
{
  svtkRectilinearGrid *rectgrid = svtkRectilinearGrid::New();
  const conduit::Node &coords = (*node)["coordsets"][0];
  int dims[3];

  dims[0]  = coords["dims"].has_child("i") ? coords["dims/i"].to_int() : 1;
  dims[1]  = coords["dims"].has_child("j") ? coords["dims/j"].to_int() : 1;
  dims[2]  = coords["dims"].has_child("k") ? coords["dims/k"].to_int() : 1;
  rectgrid->SetDimensions( dims );

  double spacing[3];
  spacing[0] = coords["spacing"].has_child("dx") ? coords["spacing/dx"].to_double(): 0;
  spacing[1] = coords["spacing"].has_child("dy") ? coords["spacing/dy"].to_double(): 0;
  spacing[2] = coords["spacing"].has_child("dz") ? coords["spacing/dz"].to_double(): 0;
  double origin[3];
  origin[0]  = coords["origin"].has_child("x") ? coords["origin/x"].to_double() : 0;
  origin[1]  = coords["origin"].has_child("y") ? coords["origin/y"].to_double() : 0;
  origin[2]  = coords["origin"].has_child("z") ? coords["origin/z"].to_double() : 0;

  for(int i = 0; i < 3 ;++i)
  {
    svtkDataArray *da = NULL;
    conduit::DataType dt = conduit::DataType::c_double();
    // we have we origin, we can infer type from it
    if( coords.has_path("origin/x") )
    {
      dt = coords["origin"]["x"].dtype();
    }

    // since svtk uses the c-native style types
    // only need to check for native types in conduit
    if( dt.is_unsigned_char() )
      da = svtkUnsignedCharArray::New();
    else if( dt.is_unsigned_short() )
      da = svtkUnsignedShortArray::New();
    else if( dt.is_unsigned_int() )
      da = svtkUnsignedIntArray::New();
    else if( dt.is_char() )
      da = svtkCharArray::New();
    else if( dt.is_short() )
      da = svtkShortArray::New();
    else if( dt.is_int() )
      da = svtkIntArray::New();
    else if( dt.is_long() )
      da = svtkLongArray::New();
    else if( dt.is_float() )
      da = svtkFloatArray::New();
    else if( dt.is_double() )
      da = svtkDoubleArray::New();
    else
    {
      SENSEI_ERROR( "Conduit Blueprint to Rectilinear Grid coordinates, unsupported data type: " << dt.name() );
    }

    da->SetNumberOfTuples( dims[i] );
    double x = origin[i];
    for (int j = 0; j < dims[i] ;++j, x += spacing[i])
      da->SetComponent( j, 0, x );

    if( i == 0 ) rectgrid->SetXCoordinates( da );
    if( i == 1 ) rectgrid->SetYCoordinates( da );
    if( i == 2 ) rectgrid->SetZCoordinates( da );
    da->Delete();
  }
  return( rectgrid );
}

//-----------------------------------------------------------------------------
/* ******* TODO ???
int ConduitDataAdaptor::GetNumberOfArrays( const std::string &meshName, int association, unsigned int &numberOfArrays )
{
  auto search = this->FieldNames.find( meshName );
  if( search == this->FieldNames.end() )
  {
    SENSEI_ERROR( "GetNumberOfArrays: Mesh " << meshName << " Cannot Be Found" );
    return( 1 );
  }
  // TODO look at removing the copy and just get the value from search->second.
  std::vector<std::string> vec = search->second;

  numberOfArrays = vec.size();
  return( 0 );
}
********* */

//-----------------------------------------------------------------------------
/* ******* TODO ???
int ConduitDataAdaptor::GetArrayName( const std::string &meshName, int association, unsigned int index, std::string &arrayName )
{
  auto search = this->FieldNames.find( meshName );
  if( search == this->FieldNames.end() )
  {
    SENSEI_ERROR( "GetArrayName: Mesh " << meshName << " Cannot Be Found" );
    return( -1 );
  }

  std::vector<std::string> vec = search->second;

  if( index > (unsigned int)vec.size() )
    return( -1 );

  arrayName = vec.at( index );

  return( 0 );
}
********* */

//-----------------------------------------------------------------------------
void ConduitDataAdaptor::SetNode( conduit::Node* node )
{
  this->Node = node;
  ConduitDataAdaptor::UpdateFields();
}

//-----------------------------------------------------------------------------
void ConduitDataAdaptor::UpdateFields()
{
  this->FieldNames.clear();
  if( conduit::blueprint::mesh::is_multi_domain(*this->Node) )
  {
    conduit::NodeConstIterator doms_itr = (this->Node)->children();
    while( doms_itr.has_next() )
    {
      const conduit::Node& d_node = doms_itr.next();
      conduit::NodeConstIterator field_itr = d_node["fields"].children();
      while( field_itr.has_next() )
      {
        //const conduit::Node& field = field_itr.next();
        field_itr.next();
        std::string field_name = field_itr.name();
        //TODO: There is no formal protocol for naming meshes within the node
        //This is a placeholder until there is a path in a node that
        //distinguishes what mesh the data belongs to.
        //For now, the meshName will be hardcoded as "mesh".
        //std::string meshName = field["mesh"].as_string();
        //std::string meshName = field["topology"].as_string();

        std::string meshName = "mesh";
        auto searchMesh = this->FieldNames.find( meshName );
        if( searchMesh != this->FieldNames.end() )
        {
          // TODO try and not make a copy of the vec (just use the searchMesh->second).
          std::vector<std::string> vec = searchMesh->second;
          std::vector<std::string>::iterator itr;
          itr = find( vec.begin(), vec.end(), field_name );
          if( itr == vec.end() )
          {
             // TODO insert the new field_name into searchMesh->second.
             vec.push_back( field_name );
             this->FieldNames.erase( meshName );
             this->FieldNames.insert( std::pair<std::string, std::vector<std::string>>(meshName, vec) );
          }
        }
        else
        {
           std::vector<std::string> vec;
           vec.push_back( field_name );
           this->FieldNames.insert( std::pair<std::string, std::vector<std::string>>(meshName, vec) );
        }
      }
    }
  }
  else
  {
    const conduit::Node& fields = (*this->Node)["fields"];
    conduit::NodeConstIterator fields_itr = fields.children();

    // TODO remove duplicate code above.
    while( fields_itr.has_next() )
    {
      const conduit::Node& field = fields_itr.next();
      std::string field_name = fields_itr.name();
      std::string meshName = field["topology"].as_string();

      auto searchMesh = this->FieldNames.find( meshName );
      if( searchMesh != this->FieldNames.end() )
      {
        std::vector<std::string> vec = searchMesh->second;
        std::vector<std::string>::iterator itr;
        itr = find( vec.begin(), vec.end(), field_name );
        if( itr == vec.end() )
        {
           vec.push_back( field_name );
           this->FieldNames.erase( meshName );
           this->FieldNames.insert( std::pair<std::string, std::vector<std::string>>(meshName, vec) );
        }
      }
      else
      {
        std::vector<std::string> vec;
        vec.push_back( field_name );
        this->FieldNames.insert( std::pair<std::string, std::vector<std::string>>(meshName, vec) );
      }
    }
  }
}

//-----------------------------------------------------------------------------
int ConduitDataAdaptor::GetMesh( const std::string &meshName, bool /*structureOnly*/, svtkDataObject *&mesh )
{   
  auto search = this->FieldNames.find( meshName );
  if( search == this->FieldNames.end() )
  {
    SENSEI_ERROR("GetMesh: Mesh " << meshName << " Cannot Be Found");
    return( -1 );
  }

  svtkMultiBlockDataSet *mb_mesh = svtkMultiBlockDataSet::New();

  int start = 0, total_blocks = 0;
  int size, rank;
  MPI_Comm_size( this->GetCommunicator(), &size );
  MPI_Comm_rank( this->GetCommunicator(), &rank );

  int global_blocks[size] = {};

  if( conduit::blueprint::mesh::is_multi_domain(*this->Node) )
  {
    int local_blocks[size] = {};
    local_blocks[rank] = this->Node->number_of_children();
    MPI_Allreduce( &local_blocks, &global_blocks, size, MPI_INT, MPI_SUM, this->GetCommunicator() );
    this->GlobalBlockDistribution = (int *)malloc( (sizeof(int) * size) );
    memcpy( this->GlobalBlockDistribution, global_blocks, (sizeof(int) * size) );
    for(int i = 0; i < size ;++i)
    {
      total_blocks += global_blocks[i];
      if( i < rank )
        start += global_blocks[i];
    }
    mb_mesh->SetNumberOfBlocks( total_blocks );
    int domain = 0;
    conduit::NodeConstIterator domain_itr = this->Node->children();
    while( domain_itr.has_next() )
    {
      int block = domain + start;
      const conduit::Node &d_node = domain_itr.next();
      const conduit::Node &coords = d_node["coordsets"][0];
      const conduit::Node &topo   = d_node["topologies"][0];
      if( coords["type"].as_string() == "uniform" )
      {
        mb_mesh->SetBlock( block, UniformMesh(&d_node) );
      }
      else if( coords["type"].as_string() == "rectilinear" )
      {
        mb_mesh->SetBlock( block, RectilinearMesh(&d_node) );
      }
      else if( coords["type"].as_string() == "explicit" )
      {
        if( topo["type"].as_string() == "structured" )
        {
          mb_mesh->SetBlock( block, StructuredMesh(&d_node) );
        }
        else
        {
          mb_mesh->SetBlock( block, UnstructuredMesh(&d_node) );
        }
      }
      ++domain;
    }
  }
  else
  {
    int local_blocks[size] = {};
    local_blocks[rank] = 1;
    MPI_Allreduce( &local_blocks, &global_blocks, size, MPI_INT, MPI_SUM, this->GetCommunicator() );
    this->GlobalBlockDistribution = (int*)malloc( (sizeof(int) * size) );
    memcpy( this->GlobalBlockDistribution, global_blocks, (sizeof(int) * size) );
    for(int i = 0; i < size ;++i)
    {
      total_blocks += global_blocks[i];
      if( i < rank )
        start += global_blocks[i];
    }

    mb_mesh->SetNumberOfBlocks( total_blocks );

    int block = start;
    const conduit::Node &coords = (*this->Node)["coordsets"][0];
    const conduit::Node &topo   = (*this->Node)["topologies"][0];
    if( coords["type"].as_string() == "uniform" )
    {
       mb_mesh->SetBlock(block, UniformMesh(this->Node));
    }
    else if( coords["type"].as_string() == "rectilinear" )
    {
      mb_mesh->SetBlock( block, RectilinearMesh(this->Node) );
    }
    else if( coords["type"].as_string() == "explicit" )
    {
      if( topo["type"].as_string() == "structured" )
      {
        mb_mesh->SetBlock( block, StructuredMesh(this->Node) );
      }
      else
      {
        mb_mesh->SetBlock( block, UnstructuredMesh(this->Node) );
      }
    }
  }
  mesh = mb_mesh;
  return( 0 );
}
//-----------------------------------------------------------------------------
int ConduitDataAdaptor::GetNumberOfMeshes( unsigned int &numberOfMeshes )
{
  //Once multiple meshes are present in the same node; you can use
  //this->FieldNames.size(). The size of the map corresponds to how many
  //different meshes.
  numberOfMeshes = 1;
  return( 0 );
}
//-----------------------------------------------------------------------------
int ConduitDataAdaptor::GetMeshMetadata(unsigned int /*id*/, sensei::MeshMetadataPtr &metadata)
{
  //this->FieldNames can be used to find the mesh name, once that becomes
  //supported in the conduit node.
  metadata->MeshName = "mesh";
  return( 0 );
}

//-----------------------------------------------------------------------------
int ConduitDataAdaptor::AddArray( svtkDataObject* mesh, const std::string &meshName, int /*association*/, const std::string &arrayname )
{
  auto search = this->FieldNames.find( meshName );
  if( search == this->FieldNames.end() )
  {
    SENSEI_ERROR( "AddArray: Mesh " << meshName << " Cannot Be Found" );
  }

  std::vector<std::string> vec = search->second;
  int flag = 1;
  for(size_t i = 0; i < vec.size() ;++i)
  {
    if( arrayname.compare(vec.at(i)) == 0 )
    {
      flag = 0;
      break;
    }
  }
  if( flag )
  {
    SENSEI_ERROR( "ERROR: field " << arrayname << " does not reside on Mesh ");
    return( -1 );
  }

  int rank, start = 0;
  MPI_Comm_rank( this->GetCommunicator(), &rank );
  for(int i = 0; i < rank ;++i)
    start += this->GlobalBlockDistribution[i];

  svtkMultiBlockDataSet *mb = dynamic_cast<svtkMultiBlockDataSet*>( mesh );

  if( !mb )
  {
    SENSEI_ERROR( "MultiBlockDataSet is NULL" );
    return( -1 );
  }
  if( conduit::blueprint::mesh::is_multi_domain(*this->Node) )
  {
    conduit::NodeConstIterator dom_itr = this->Node->children();
    int domain = 0;
    while( dom_itr.has_next() )
    {
      const conduit::Node& d_node = dom_itr.next();
      const conduit::Node& fields = d_node["fields"];
      const conduit::Node& field  = fields[arrayname];
      const conduit::Node& values = field["values"];
            
      svtkSmartPointer<svtkDataArray> array = ConduitArrayToSVTKDataArray( values );
      array->SetName( arrayname.c_str() );
       
      svtkDataObject *block = mb->GetBlock( start + domain );

      std::stringstream ss;
      ss << "fields/" << arrayname << "/association";
      std::string association_path  = ss.str();
      std::string field_association = d_node[association_path].as_string();

      std::stringstream rr;
      rr << "fields/" << arrayname << "/values";
      std::string values_path = rr.str();
      int nchildren = values.number_of_children();
      if( field_association == "vertex" )
      {
        if( nchildren > 0 )
        {
          svtkDataSetAttributes *attr = block->GetAttributes( svtkDataObject::POINT );
          attr->AddArray( array );
        }
        else
        {
          svtkDataSetAttributes *attr = block->GetAttributes( svtkDataObject::POINT );
          attr->AddArray( array );
        }
      }
      else if( field_association == "element" )
      {
        if( nchildren > 0 )
        {
          svtkDataSetAttributes *attr = block->GetAttributes( svtkDataObject::CELL );
          attr->AddArray( array );
        }
        else
        {
          svtkDataSetAttributes *attr = block->GetAttributes( svtkDataObject::CELL );
          attr->AddArray( array );
        }
      }
      else
      {
        SENSEI_ERROR( "ERROR: association of type " << field_association << " incompatible" );
        return( -1 );
      }
      ++domain;
    }
  }
  else
  {
    const conduit::Node& fields  = (*this->Node)["fields"];
    const conduit::Node& field   = fields[arrayname];
    const conduit::Node& values  = field["values"];
    svtkSmartPointer<svtkDataArray> array = ConduitArrayToSVTKDataArray( values );
    array->SetName( arrayname.c_str() );

    svtkDataObject *block = mb->GetBlock( start );

    std::stringstream ss;
    ss << "fields/" << arrayname << "/association";
    std::string association_path  = ss.str();
    std::string field_association = (*this->Node)[association_path].as_string();

    std::stringstream rr;
    rr << "fields/" << arrayname << "/values";
    std::string values_path = rr.str();
    int nchildren = (*this->Node)[values_path].number_of_children();

    if( field_association == "vertex" )
    {
      if( nchildren > 0 )
      {
        svtkDataSetAttributes *attr = block->GetAttributes( svtkDataObject::POINT );
        attr->AddArray( array );
      }
      else
      {
        svtkDataSetAttributes *attr = block->GetAttributes( svtkDataObject::POINT );
        attr->AddArray( array );
      }
    }
    else if( field_association == "element" )
    {
      if( nchildren > 0 )
      {
        svtkDataSetAttributes *attr = block->GetAttributes( svtkDataObject::CELL );
        attr->AddArray( array );
      }
      else
      {
        svtkDataSetAttributes *attr = block->GetAttributes( svtkDataObject::CELL );
        attr->AddArray( array );
      }
    }
    else
    {
      SENSEI_ERROR( "ERROR: association of type " << field_association << " incompatible" );
      return( -1 );
    }
  }
  return( 0 );
}

//-----------------------------------------------------------------------------
int ConduitDataAdaptor::ReleaseData()
{
  this->Node = NULL;
  this->FieldNames.clear();
  free( this->GlobalBlockDistribution );

  return( 0 );
}

}
