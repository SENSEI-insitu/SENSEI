#include "AscentAnalysisAdaptor.h"
#include "DataAdaptor.h"
#include "Error.h"
#include "MeshMetadataMap.h"
#include "SVTKUtils.h"

#include <mpi.h>

#include <conduit.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp> 

#include <svtkObjectFactory.h>
#include <svtkCellArray.h>
#include <svtkDataObject.h>
#include <svtkDataArray.h>
#include <svtkFieldData.h>
#include <svtkDataSet.h>
#include <svtkDataSetAttributes.h>
#include <svtkCompositeDataIterator.h>
#include <svtkCompositeDataSet.h>
#include <svtkImageData.h>
#include <svtkRectilinearGrid.h>
#include <svtkStructuredGrid.h>
#include <svtkUnstructuredGrid.h>
#include <svtkCellData.h>
#include <svtkPointData.h>
#include <svtkAOSDataArrayTemplate.h>
#include <svtkSOADataArrayTemplate.h>
#include <svtkDataArrayTemplate.h>
#include <svtkUnsignedCharArray.h>

#define DEBUG_TRACE 0   //  for debug work, wes 8/2021

namespace
{
// --------------------------------------------------------------------------
template<typename n_t> struct conduit_tt {};

#define declare_conduit_tt(cpp_t, conduit_t) \
template<> struct conduit_tt<cpp_t>          \
{                                            \
  using conduit_type = conduit_t;            \
};

declare_conduit_tt(char, conduit::int8);
declare_conduit_tt(signed char, conduit::int8);
declare_conduit_tt(unsigned char, conduit::uint8);
declare_conduit_tt(short, conduit::int16);
declare_conduit_tt(unsigned short, conduit::uint16);
declare_conduit_tt(int, conduit::int32);
declare_conduit_tt(unsigned int, conduit::uint32);
declare_conduit_tt(long, conduit::int32);
declare_conduit_tt(unsigned long, conduit::uint32);
declare_conduit_tt(long long, conduit::int64);
declare_conduit_tt(unsigned long long, conduit::uint64);
declare_conduit_tt(float, conduit::float32);
declare_conduit_tt(double, conduit::float64);
declare_conduit_tt(long double, conduit::float64);

//------------------------------------------------------------------------------
void GetShape(std::string &shape, int type)
{
  if(type == 1) shape = "point";
  else if(type == 2) shape = "line";
  else if(type == 3) shape = "tri";
  else if(type == 4) shape = "quad";
  else if(type == 8) shape = "hex";
  else SENSEI_ERROR("Error: Unsupported element shape");
}
//#define DEBUG_SAVE_DATA
#ifdef DEBUG_SAVE_DATA
void DebugSaveAscentData( conduit::Node &data, conduit::Node &_optionsNode )
{
    ascent::Ascent a;

    std::cout << "DebugSaveAscentData" << std::endl;

    // Open ascent
    a.open( _optionsNode );

    // Publish data to ascent
    a.publish( data );

    conduit::Node extracts;
    extracts["e1/type"] = "relay";
    extracts["e1/params/path"] = "debugAscentDataSave";
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    // Setup actions
    conduit::Node actions;
    conduit::Node &add_act = actions.append();
    add_act["action"] = "add_extracts";
    add_act["extracts"] = extracts;

    actions.append()["action"] = "execute";

    // execute
    a.execute(actions);
    std::cout << "DebugSaveAscentData: execute" << std::endl;

    // close ascent
    a.close();
}
#endif  // DEBUG_SAVE_DATA

//------------------------------------------------------------------------------
int PassGhostsZones(svtkDataSet* ds, conduit::Node& node)
{
  // Check if the mesh has ghost zone data.
  if( ds->HasAnyGhostCells() || ds->HasAnyGhostPoints() )
  {
    // If so, add the data for Acsent.
    node["fields/ascent_ghosts/association"] = "element";
    node["fields/ascent_ghosts/topology"] = "mesh";
    node["fields/ascent_ghosts/type"] = "scalar";

    svtkUnsignedCharArray *gc = svtkUnsignedCharArray::SafeDownCast(ds->GetCellData()->GetArray("svtkGhostType"));
    unsigned char *gcp = (unsigned char *)gc->GetVoidPointer( 0 );
    auto size = gc->GetSize();

    // In Acsent, 0 means real data, 1 means ghost data, and 2 or greater means garbage data.
    std::vector<conduit::int32> ghost_flags(size);

    // Ascent needs int32 not unsigned char. I don't know why, that is the way.
    for(svtkIdType i=0; i < size ;++i)
    {
        ghost_flags[i] = gcp[i];
    }
    node["fields/ascent_ghosts/values"].set(ghost_flags);
  }

  return 0;
}

/* TODO look at
int PassFields(svtkDataSetAttributes *dsa, int centering, conduit::Node &node)
{
  int nArrays = dsa->GetNumberOfArrays();
  for (int i = 0; i < nArrays; ++i)
  {
    svtkDataArray *da = dsa->GetArray(i);
    const char *name = da->GetName();
    long nElem = da->GetNumberOfTuples();
    int nComps = da->GetNumberOfComponents();

    std::string arrayName(name);

    //nComp = 1 -> type = scalar; nComp > 1 -> type = vector
    std::stringstream ss;
    ss << "fields/" << arrayName << "/type";
    std::string typePath = ss.str();
    ss.str(std::string());

    //nComp = 1
    ss << "fields/" << arrayName << "/values";
    std::string valPath = ss.str();
    ss.str(std::string());

    //nComp > 1
    ss << "fields/" << arrayName << "/values/u";
    std::string uValPath = ss.str();
    ss.str(std::string());

    ss << "fields/" << arrayName << "/values/v";
    std::string vValPath = ss.str();
    ss.str(std::string());

    ss << "fields/" << arrayName << "/values/w";
    std::string wValPath = ss.str();
    ss.str(std::string());


    switch (da->GetDataType())
    {
      svtkTemplateMacro(
        svtkAOSDataArrayTemplate<SVTK_TT> *aosda =
          dynamic_cast<svtkAOSDataArrayTemplate<SVTK_TT>*>(da);

        svtkSOADataArrayTemplate<SVTK_TT> *soada =
          dynamic_cast<svtkSOADataArrayTemplate<SVTK_TT>*>(da);

        if (aosda)
        {
          // AOS
          //SVTK_TT *ptr = aosda->GetPointer(0);

          // TODO -- Have to set the u v w separately -- like below
          // Not sure how to do that with zero copy
          // Original code copied the data into three separate vectors
          //node[valPath].set_external(ptr, nElem);
          //node[uValPath].set_external(ptr, nElem);
          //node[vValPath].set_external(ptr, nElem);
          //node[wValPath].set_external(ptr, nElem);
        }
        else if (soada)
        {
          // SOA
          for (int j = 0; j < nComps; ++j)
          {
            SVTK_TT *ptr = soada->GetComponentArrayPointer(j);
            if(nComps == 1)
            {
              node[valPath].set_external((conduit_tt<SVTK_TT>::conduit_type*)ptr, nElem, 0,
                sizeof(SVTK_TT), sizeof(SVTK_TT), conduit::Endianness::DEFAULT_ID);
              node[typePath] = "scalar";
            }
            else
            {
              switch(j)
              {
                 case 0: node[uValPath].set_external((conduit_tt<SVTK_TT>::conduit_type*)ptr, nElem, 0,
                           sizeof(SVTK_TT), sizeof(SVTK_TT), conduit::Endianness::DEFAULT_ID);
                         node[typePath] = "vector";
                         break;
                 case 1: node[vValPath].set_external((conduit_tt<SVTK_TT>::conduit_type*)ptr, nElem, 0,
                           sizeof(SVTK_TT), sizeof(SVTK_TT), conduit::Endianness::DEFAULT_ID);
                         break;
                 case 2: node[wValPath].set_external((conduit_tt<SVTK_TT>::conduit_type*)ptr, nElem, 0,
                           sizeof(SVTK_TT), sizeof(SVTK_TT), conduit::Endianness::DEFAULT_ID);
                         break;

              }
            }
          }
        }
        else
        {
          // this should never happen
          SENSEI_ERROR("Invalid array type \"" << da->GetClassName() << "\" at " << i);
        }
        );
      default:
        SENSEI_ERROR("Invalid type from " << SVTKUtils::GetAttributesName(centering)
          << " data array " << i << " named \"" << (name ? name : "") << "\"");
    }
  }

  return( 0 );
}

int PassFields(int bid, svtkDataSet *ds, conduit::Node &node)
{
  // handle the arrays
  if (PassFields(ds->GetPointData(), svtkDataObject::POINT, node))
  {
    SENSEI_ERROR("Failed to transfer point data from block " << bid);
    return( -1 );
  }

  if (PassFields(ds->GetCellData(), svtkDataObject::CELL, node))
  {
    SENSEI_ERROR("Failed to transfer cell data from block " << bid);
    return( -1 );
  }

  return( 0 );
}
*/

// **************************************************************************
int PassFields(svtkDataArray *da, conduit::Node& node,
  const std::string &arrayName, int arrayCen)
{
  std::stringstream ss;
  ss << "fields/" << arrayName << "/association";
  std::string assocPath = ss.str();
  ss.str(std::string());

  //ss << "fields/" << arrayName << "/volume_dependent";
  //std::string volPath = ss.str();
  //ss.str(std::string());

  ss << "fields/" << arrayName << "/topology";
  std::string topoPath = ss.str();
  ss.str(std::string());

  ss << "fields/" << arrayName << "/type";
  std::string typePath = ss.str();
  ss.str(std::string());

  ss << "fields/" << arrayName << "/values";
  std::string valPath = ss.str();
  ss.str(std::string());

  ss << "fields/" << arrayName << "/values/u";
  std::string uValPath = ss.str();
  ss.str(std::string());

  ss << "fields/" << arrayName << "/values/v";
  std::string vValPath = ss.str();
  ss.str(std::string());

  ss << "fields/" << arrayName << "/values/w";
  std::string wValPath = ss.str();
  ss.str(std::string());

  //ss << "fields/" << arrayName << "/grid_function";
  //std::string gridPath = ss.str();
  //ss.str(std::string());

  //ss << "fields/" << arrayName << "/matset";
  //std::string matPath = ss.str();
  //ss.str(std::string());

  //ss << "fields/" << arrayName << "/matset_values";
  //std::string matValPath = ss.str();
  //ss.str(std::string());


  // tell ascent the centering
  std::string cenType;
  if (arrayCen == svtkDataObject::POINT)
  {
    cenType = "vertex";
  }
  else if (arrayCen == svtkDataObject::CELL)
  {
    cenType = "element";
  }
  else
  {
    SENSEI_ERROR("Invlaid centering " << arrayCen)
    return -1;
  }
  node[assocPath] = cenType;

  // FIXME -- zero coopy transfer the data
  int components = da->GetNumberOfComponents();
  int tuples = da->GetNumberOfTuples();

  if(components == 1)
  {
    node[typePath] = "scalar";

    // FIXME -- get the type from SVTK
    std::vector<conduit::float64> vals(tuples, 0.0);

    for(int i = 0; i < tuples; ++i)
    {
      vals[i] = da->GetComponent(i, 0);
    }
    node[valPath].set(vals);
  }
  else if(components == 2)
  {
    node[typePath] = "vector";

    std::vector<conduit::float64> uVals(tuples, 0.0);
    std::vector<conduit::float64> vVals(tuples, 0.0);

    for(int i = 0; i < tuples; ++i)
    {
      uVals[i] = da->GetComponent(i, 0);
      vVals[i] = da->GetComponent(i, 1);
    }
    node[uValPath].set(uVals);
    node[vValPath].set(vVals);
  }
  else if(components == 3)
  {
    node[typePath] = "vector";

    std::vector<conduit::float64> uVals(tuples, 0.0);
    std::vector<conduit::float64> vVals(tuples, 0.0);
    std::vector<conduit::float64> wVals(tuples, 0.0);

    for(int i = 0; i < tuples; ++i)
    {
      uVals[i] = da->GetComponent(i, 0);
      vVals[i] = da->GetComponent(i, 1);
      wVals[i] = da->GetComponent(i, 2);
    }
    node[uValPath].set(uVals);
    node[vValPath].set(vVals);
    node[wValPath].set(wVals);
  }
  else
  {
    SENSEI_ERROR("Too many components (" << components << ") associated with " << arrayName);
    return -1;
  }

  // tell ascent which topology the array belongs to
  node[topoPath] = "mesh";

  return 0;
}

//------------------------------------------------------------------------------
int PassTopology(svtkDataSet* ds, conduit::Node& node)
{
  svtkImageData *uniform             = svtkImageData::SafeDownCast(ds);
  svtkRectilinearGrid *rectilinear   = svtkRectilinearGrid::SafeDownCast(ds);
  svtkStructuredGrid *structured     = svtkStructuredGrid::SafeDownCast(ds);
  svtkUnstructuredGrid *unstructured = svtkUnstructuredGrid::SafeDownCast(ds);

  if(uniform != nullptr)
  {
    node["topologies/mesh/type"]     = "uniform";
    node["topologies/mesh/coordset"] = "coords";

    int dims[3] = {0, 0, 0};
    uniform->GetDimensions(dims);

    double origin[3] = {0.0, 0.0, 0.0};
    uniform->GetOrigin(origin);

    int extents[6] = {0, 0, 0, 0, 0, 0};
    uniform->GetExtent(extents);

    double spacing[3] = {0.0, 0.0, 0.0};
    uniform->GetSpacing(spacing);

    node["topologies/mesh/elements/origin/i0"] = origin[0] + (extents[0] * spacing[0]);
    node["topologies/mesh/elements/origin/j0"] = origin[1] + (extents[2] * spacing[1]);
// wes    if(dims[2] != 0 && dims[2] != 1)
    if(dims[2] != 0)
      node["topologies/mesh/elements/origin/k0"] = origin[2] + (extents[4] * spacing[2]);
  }
  else if(rectilinear != nullptr)
  {
    node["topologies/mesh/type"]     = "rectilinear";
    node["topologies/mesh/coordset"] = "coords";

  }
  else if(structured != nullptr)
  {
    node["topologies/mesh/type"]     = "structured";
    node["topologies/mesh/coordset"] = "coords";

    int dims[3] = {0, 0, 0};
    structured->GetDimensions(dims);

    node["topologies/mesh/elements/dims/i"] = dims[0] - 1;
    node["topologies/mesh/elements/dims/j"] = dims[1] - 1;
    if(dims[2] != 0 && dims[2] != 1)
      node["topologies/mesh/elements/dims/k"] = dims[2] - 1;
  }
  else if(unstructured != nullptr)
  {
    // fixme: this code makes a deep copy, look for ways
    // to do a more shallow copy
    if(!unstructured->IsHomogeneous())
    {
      SENSEI_ERROR("Unstructured cells must be homogenous");
      return( -1 );
    }
    node["topologies/mesh/type"]     = "unstructured";
    node["topologies/mesh/coordset"] = "coords";

    svtkCellArray* cellarray = unstructured->GetCells();

    svtkIdTypeArray *idarray = svtkIdTypeArray::New();
    cellarray->ExportLegacyFormat(idarray);

    // svtkIdType *ptr = cellarray->GetPointer();
    svtkIdType *ptr = idarray->GetPointer(0);

    std::string shape;
    GetShape(shape, ptr[0]);
    node["topologies/mesh/elements/shape"] = shape;

    int ncells = unstructured->GetNumberOfCells();
    int connections = ncells*(ptr[0] + 1);
    std::vector<int> data(ncells*ptr[0], 0);

    int count = 0;
    for(int i = 0; i < connections; ++i)
    {
      int offset = ptr[0] + 1;
      if(i%offset == 0)
        continue;
      else
      {
        data[count] = ptr[i];
        count++;
      }
    }
    node["topologies/mesh/elements/connectivity"].set(data);
    idarray->Delete();
  }
  else
  {
    SENSEI_ERROR("Mesh structure not supported");
    return( -1 );
  }

  return 0;
}

//------------------------------------------------------------------------------
int PassState(svtkDataSet *, conduit::Node& node, sensei::DataAdaptor *dataAdaptor)
{
    node["state/time"] = dataAdaptor->GetDataTime();
    node["state/cycle"] = (int)dataAdaptor->GetDataTimeStep();
    return  0;
}

//------------------------------------------------------------------------------
int PassCoordsets(svtkDataSet* ds, conduit::Node& node)
{
  svtkImageData *uniform             = svtkImageData::SafeDownCast(ds);
  svtkRectilinearGrid *rectilinear   = svtkRectilinearGrid::SafeDownCast(ds);
  svtkStructuredGrid *structured     = svtkStructuredGrid::SafeDownCast(ds);
  svtkUnstructuredGrid *unstructured = svtkUnstructuredGrid::SafeDownCast(ds);

  if(uniform != nullptr)
  {
    node["coordsets/coords/type"] = "uniform";

    //Local Dimensions
    int dims[3] = {0,0,0};
    uniform->GetDimensions(dims);
    node["coordsets/coords/dims/i"] = dims[0];
    node["coordsets/coords/dims/j"] = dims[1];
    node["coordsets/coords/dims/k"] = dims[2];

    //Global Origin
    double origin[3] = {0.0, 0.0, 0.0};
    uniform->GetOrigin(origin);

    int extents[6] = {0, 0, 0, 0, 0, 0};
    uniform->GetExtent(extents);

    double spacing[3] = {0.0, 0.0, 0.0};
    uniform->GetSpacing(spacing);

    node["coordsets/coords/origin/x"] = origin[0] + (extents[0] * spacing[0]);
    node["coordsets/coords/origin/y"] = origin[1] + (extents[2] * spacing[1]);
    node["coordsets/coords/origin/z"] = origin[2] + (extents[4] * spacing[2]);

    //Global Spacing == Local Spacing
    node["coordsets/coords/spacing/dx"] = spacing[0];
    node["coordsets/coords/spacing/dy"] = spacing[1];
    node["coordsets/coords/spacing/dz"] = spacing[2];
  }
  else if(rectilinear != nullptr)
  {
    node["coordsets/coords/type"] = "rectilinear";

    int dims[3] = {0, 0, 0};
    rectilinear->GetDimensions(dims);

    svtkDataArray *x = rectilinear->GetXCoordinates();
    svtkDataArray *y = rectilinear->GetYCoordinates();
    svtkDataArray *z = rectilinear->GetZCoordinates();

    if (!x || !y || !z)
    {
      SENSEI_ERROR("Invalid coordinate arrays in rectilinear");
      return( -1 );
    }

    switch (x->GetDataType())
    {
      svtkTemplateMacro(
        // x
        svtkAOSDataArrayTemplate<SVTK_TT> *tx =
           dynamic_cast<svtkAOSDataArrayTemplate<SVTK_TT>*>(x);
        SVTK_TT *ptr = tx->GetPointer(0);
        long nElem = tx->GetNumberOfTuples();
        node["coordsets/coords/values/x"].set_external((conduit_tt<SVTK_TT>::conduit_type*)ptr, nElem, 0,
          sizeof(SVTK_TT), sizeof(SVTK_TT), conduit::Endianness::DEFAULT_ID);
        // y
        svtkAOSDataArrayTemplate<SVTK_TT> *ty =
           dynamic_cast<svtkAOSDataArrayTemplate<SVTK_TT>*>(y);
        ptr = ty->GetPointer(0);
        nElem = ty->GetNumberOfTuples();
        node["coordsets/coords/values/y"].set_external((conduit_tt<SVTK_TT>::conduit_type*)ptr, nElem, 0,
          sizeof(SVTK_TT), sizeof(SVTK_TT), conduit::Endianness::DEFAULT_ID);
        // z
        svtkAOSDataArrayTemplate<SVTK_TT> *tz =
           dynamic_cast<svtkAOSDataArrayTemplate<SVTK_TT>*>(z);
        ptr = tz->GetPointer(0);
        nElem = tz->GetNumberOfTuples();
        if( nElem > 1 )
            node["coordsets/coords/values/z"].set_external((conduit_tt<SVTK_TT>::conduit_type*)ptr, nElem, 0,
                sizeof(SVTK_TT), sizeof(SVTK_TT), conduit::Endianness::DEFAULT_ID);
        );
      default:
        SENSEI_ERROR("Invlaid data type for recilinear grid coordinates");
        return( -1 );
    }
  }
  else if(structured != nullptr)
  {
    node["coordsets/coords/type"] = "explicit";

    int dims[3] = {0, 0, 0};
    structured->GetDimensions(dims);

    int numPoints = structured->GetPoints()->GetNumberOfPoints();
    double point[3] = {0, 0, 0};
    std::vector<conduit::float64> x(numPoints, 0.0);
    std::vector<conduit::float64> y(numPoints, 0.0);
    std::vector<conduit::float64> z(numPoints, 0.0);

    for(int i = 0; i < numPoints; ++i)
    {
      structured->GetPoints()->GetPoint(i, point);
      x[i] = point[0];
      y[i] = point[1];
      z[i] = point[2];
    }
    node["coordsets/coords/values/x"].set(x);
    node["coordsets/coords/values/y"].set(y);
    if(dims[2] != 0 && dims[2] != 1)
      node["coordsets/coords/values/z"].set(z);
  }
  else if(unstructured != nullptr)
  {
    node["coordsets/coords/type"] = "explicit";

    int numPoints = unstructured->GetPoints()->GetNumberOfPoints();
    double point[3] = {0, 0, 0};
    std::vector<conduit::float64> x(numPoints, 0.0);
    std::vector<conduit::float64> y(numPoints, 0.0);
    std::vector<conduit::float64> z(numPoints, 0.0);

    for(int i = 0; i < numPoints; ++i)
    {
      unstructured->GetPoints()->GetPoint(i, point);
      x[i] = point[0];
      y[i] = point[1];
      z[i] = point[2];
    }
    node["coordsets/coords/values/x"].set(x);
    node["coordsets/coords/values/y"].set(y);
    node["coordsets/coords/values/z"].set(z);
  }
  else
  {
    SENSEI_ERROR("Mesh type not supported");
    return -1;
  }

  return 0;
}

// **************************************************************************
void NodeIter(const conduit::Node& node, std::set<std::string>& fields)
{
  if(node.number_of_children() > 0)
  {
    if(node.has_child("field"))
    {
      std::string field = node["field"].as_string();
      fields.insert(field);
    }
    else
    {
      conduit::NodeConstIterator itr = node.children();
      while(itr.has_next())
      {
        const conduit::Node& child = itr.next();
        NodeIter(child, fields);
      }
    }
  }
}

// **************************************************************************
int PassData(svtkDataSet* ds, conduit::Node& node,
  sensei::DataAdaptor *dataAdaptor)
{
    // FIXME -- do error checking on all these and report any errors
    // FIXME -- instead of passing data adaptor pass time and time step

    PassState(ds, node, dataAdaptor);

    PassCoordsets(ds, node);
    PassTopology(ds, node);

    int arrayCens[] = {svtkDataObject::POINT, svtkDataObject::CELL};
    for (int j = 0; j < 2; ++j)
    {
        int arrayCen = arrayCens[j];

        svtkDataSetAttributes *atts = ds->GetAttributes(arrayCen);
        int numArrays = atts->GetNumberOfArrays();
        for (int i = 0; i < numArrays; ++i)
        {
          svtkDataArray *da = atts->GetArray(i);

          const char *arrayName = da->GetName();

          PassFields(da, node, arrayName, arrayCen);
        }
    }

    PassGhostsZones(ds, node);
    return 0;
}


//------------------------------------------------------------------------------
int LoadConfig(const std::string &file_name, conduit::Node& node)
{
  if (!conduit::utils::is_file(file_name))
  {
    SENSEI_ERROR("Failed to load ascent config. \"" << file_name
      << "\" is not a valid ascent config.")
    return -1;
  }

  conduit::Node file_node;
  file_node.load(file_name, "json");
  node.update(file_node);

  return 0;
}

}



namespace sensei
{
//------------------------------------------------------------------------------
senseiNewMacro(AscentAnalysisAdaptor);

//------------------------------------------------------------------------------
AscentAnalysisAdaptor::AscentAnalysisAdaptor()
{
}

//------------------------------------------------------------------------------
AscentAnalysisAdaptor::~AscentAnalysisAdaptor()
{
}

//-----------------------------------------------------------------------------
int AscentAnalysisAdaptor::SetDataRequirements(const DataRequirements &reqs)
{
  this->Requirements = reqs;
  return 0;
}

//-----------------------------------------------------------------------------
int AscentAnalysisAdaptor::AddDataRequirement(const std::string &meshName,
  int association, const std::vector<std::string> &arrays)
{
  this->Requirements.AddRequirement(meshName, association, arrays);
  return 0;
}

//------------------------------------------------------------------------------
void AscentAnalysisAdaptor::GetFieldsFromActions()
{
  const conduit::Node& temp = this->actionsNode;
  NodeIter(temp, this->Fields);
}

//------------------------------------------------------------------------------
int AscentAnalysisAdaptor::Initialize(const std::string &json_file_path,
  const std::string &options_file_path)
{
  if (!options_file_path.empty() &&
    ::LoadConfig(options_file_path, this->optionsNode))
  {
    SENSEI_ERROR("Failed to load options from \""
      << options_file_path << "\"")
    return -1;
  }

  optionsNode["mpi_comm"] = MPI_Comm_c2f(this->GetCommunicator());
  //optionsNode["runtime/type"] = "ascent";

  this->_ascent.open(this->optionsNode);

  if (::LoadConfig(json_file_path, this->actionsNode))
  {
    SENSEI_ERROR("Failed to load actions from \""
      << json_file_path << "\"")
    return -1;
  }

#ifdef DEBUG_SAVE_DATA
  std::cout << "------ "  << options_file_path << " ------" << std::endl;
  this->optionsNode.print();
  std::cout << "------ "  << json_file_path << " ------" << std::endl;
  this->actionsNode.print();
  std::cout << "----------------------------" << std::endl;
#endif

  return 0;
}

//------------------------------------------------------------------------------
bool AscentAnalysisAdaptor::Execute_new(DataAdaptor* dataAdaptor)
{
#if DEBUG_TRACE
  std::cout << "AscentAnalysisAdaptor::Execute_new() - begin " << std::endl;
#endif

  conduit::Node localRoot;

  MeshMetadataFlags flags;
  flags.SetBlockDecomp();
  flags.SetBlockSize();
  flags.SetBlockBounds();
  flags.SetBlockExtents();
  flags.SetBlockArrayRange();

  MeshMetadataMap mdm;
  if (mdm.Initialize(dataAdaptor, flags))
  {
     SENSEI_ERROR("Failed to get metadata")
     return false;
  }

  // if no dataAdaptor requirements are given, push all the data
  // fill in the requirements with every thing
  if (this->Requirements.Empty())
  {
     if (this->Requirements.Initialize(dataAdaptor, false))
     {
       SENSEI_ERROR("Failed to initialze dataAdaptor description")
       return false;
     }
     SENSEI_WARNING("No subset specified. Writing all available data")
  }

  // collect the specified data objects and metadata
  std::vector<svtkCompositeDataSet*> objects;
  std::vector<MeshMetadataPtr> metadata;

  MeshRequirementsIterator mit =
      this->Requirements.GetMeshRequirementsIterator();

  while (mit)
  {
    // get metadata
    MeshMetadataPtr md;

    if (mdm.GetMeshMetadata(mit.MeshName(), md))
    {
      SENSEI_ERROR("Failed to get mesh metadata for mesh \""
              << mit.MeshName() << "\"")
      return false;
    }

#if DEBUG_TRACE
    std::cout << " Fetching mesh named: [" << mit.MeshName() << "] " << std::endl;
#endif

    // get the mesh
    svtkCompositeDataSet *dobj = nullptr;

    if (dataAdaptor->GetMesh(mit.MeshName(), mit.StructureOnly(), dobj))
    {
      SENSEI_ERROR("Failed to get mesh \"" << mit.MeshName() << "\"")
      return false;
    }

    // add the ghost cell arrays to the mesh
    if ((md->NumGhostCells || SVTKUtils::AMR(md)) &&
            dataAdaptor->AddGhostCellsArray(dobj, mit.MeshName()))
    {
      SENSEI_ERROR("Failed to get ghost cells for mesh \"" << mit.MeshName() << "\"")
      return false;
    }

    // add the ghost node arrays to the mesh
    if (md->NumGhostNodes && dataAdaptor->AddGhostNodesArray(dobj, mit.MeshName()))
    {
      SENSEI_ERROR("Failed to get ghost nodes for mesh \"" << mit.MeshName() << "\"")
      return false;
    }

    // add the required arrays
    ArrayRequirementsIterator ait =
        this->Requirements.GetArrayRequirementsIterator(mit.MeshName());

    while (ait)
    {
      if (dataAdaptor->AddArray(dobj, mit.MeshName(),
                  ait.Association(), ait.Array()))
      {
        SENSEI_ERROR("Failed to add "
                 << SVTKUtils::GetAttributesName(ait.Association())
                 << " data array \"" << ait.Array() << "\" to mesh \""
                 << mit.MeshName() << "\"")
        return false;
      }

#if DEBUG_TRACE
      std::cout << " Fetching array named [" << ait.Array() << "] with centering [" << SVTKUtils::GetAttributesName(ait.Association()) << "] " << std::endl;
#endif

      ++ait;
    } // while (ait))

    // generate a global view of the metadata. everything we do from here
    // on out depends on having the global view.
    md->GlobalizeView(this->GetCommunicator());

    // add to the collection
    objects.push_back(dobj);
    metadata.push_back(md);

    ++mit;

  } // while(mit)

  // at this point, we pulled/fetched all data from the simulation requested
  // by the config file. this collection is located in:
  // objects: vector of svtkCompositeDataSet
  // metadata: vector of MeshMetaDataPtr

  // now, iterate over the collection of data objects and convert them
  // to conduit meshes
  for (unsigned int i=0; i < objects.size(); i++)
  {
    // process each of the objects[i]
    // see this URL for info about the processing motif:
    // https://www.paraview.org/Wiki/SVTK/Tutorials/Composite_Datasets

    svtkCompositeDataSet* input = objects[i];
    svtkCompositeDataIterator* iter = input->NewIterator();

    int nds=0;
    for ( iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem(), nds++)
    {
      svtkDataSet* inputDS = svtkDataSet::SafeDownCast(iter->GetCurrentDataObject());
//#if DEBUG_TRACE
#if 0
      std::cout << "Processing cds# " << i << " and working on the " << nds << "'th dataset " << std::endl;
#endif
      
      ::PassData(inputDS, localRoot, dataAdaptor);
    }

    iter->Delete();
  }

  // invoke ascent's execute method
  this->_ascent.publish(localRoot);
  this->_ascent.execute(this->actionsNode);

  localRoot.reset();

  unsigned int n_objects = objects.size();
  for (unsigned int i = 0; i < n_objects; ++i)
      objects[i]->Delete();

   return true;

}

//------------------------------------------------------------------------------
bool AscentAnalysisAdaptor::Execute(DataAdaptor* dataAdaptor, DataAdaptor*&)
{
    // return AscentAnalysisAdaptor::Execute_original(dataAdaptor);
    return AscentAnalysisAdaptor::Execute_new(dataAdaptor);
}

//------------------------------------------------------------------------------
bool AscentAnalysisAdaptor::Execute_original(DataAdaptor* dataAdaptor)
{
  // FIXME -- data requirements needs to be determined from ascent
  // configs.
  // get the mesh name
  std::cout << "AscentAnalysisAdaptor::Execute_original() - begin " << std::endl;

  if (this->Requirements.Empty())
    {
    SENSEI_ERROR("Data requirements have not been provided")
    return false;
    }

  MeshRequirementsIterator mit =
    this->Requirements.GetMeshRequirementsIterator();

  if (!mit)
    {
    SENSEI_ERROR("Invalid data requirements")
    return false;
    }

  const std::string &meshName = mit.MeshName();

  // get the array name
  ArrayRequirementsIterator ait =
    this->Requirements.GetArrayRequirementsIterator(meshName);

  if (!ait)
    {
    SENSEI_ERROR("Invalid data requirements. No array for mesh \""
      << meshName << "\"")
    return false;
    }

  std::string arrayName = ait.Array();
  int arrayCen = ait.Association();

  conduit::Node root;
  svtkDataObject* obj = nullptr;

  // see what the simulation is providing
  MeshMetadataMap mdMap;
  if (mdMap.Initialize(dataAdaptor))
  {
    SENSEI_ERROR("Failed to get metadata")
    return( false );
  }

  if (dataAdaptor->GetMesh(meshName, false, obj))
  {
    SENSEI_ERROR("Failed to get mesh");
    return( false );
  }

  if (dataAdaptor->AddArray(obj, meshName, arrayCen, arrayName))
  {
    SENSEI_ERROR("Failed to add "
      << " data array \"" << arrayName << "\" to mesh \""
      << meshName << "\"");
    return( -1 );
  }

  // get metadata for the requested mesh
  MeshMetadataPtr metadata;
  if (mdMap.GetMeshMetadata(meshName, metadata))
  {
    SENSEI_ERROR("Failed to get metadata for mesh \"" << meshName << "\"");
    return( false );
  }

  // Add the ghost cell arrays to the mesh.
  if ((metadata->NumGhostCells || SVTKUtils::AMR(metadata)) &&
    dataAdaptor->AddGhostCellsArray(obj, meshName))
  {
      SENSEI_ERROR("Failed to get ghost cells for mesh \"" << meshName << "\"");
      return( false );
  }

  // Add the ghost node arrays to the mesh.
  if (metadata->NumGhostNodes &&
    dataAdaptor->AddGhostNodesArray(obj, meshName))
  {
      SENSEI_ERROR("Failed to get ghost nodes for mesh \"" << meshName << "\"");
      return( false );
  }

  int domainNum = 0;
  if (svtkCompositeDataSet *cds = dynamic_cast<svtkCompositeDataSet*>(obj))
  {
    size_t numBlocks = 0;
    svtkCompositeDataIterator *itr = cds->NewIterator();

    // FIXME - use metadata to get the number of blocks
    svtkCompositeDataIterator *iter = cds->NewIterator();
    iter->SkipEmptyNodesOn();
    for(iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
      ++numBlocks;

    itr->SkipEmptyNodesOn();
    itr->InitTraversal();
    while(!itr->IsDoneWithTraversal())
    {
      if (svtkDataSet *ds = dynamic_cast<svtkDataSet*>(cds->GetDataSet(itr)))
      {
        char domain[20] = "";
        if( numBlocks > 1 )
        {
          snprintf( domain, sizeof(domain), "domain_%.6d", domainNum );
          ++domainNum;
          conduit::Node &temp_node = root[domain];

          // FIXME -- check retuirn for error
          ::PassData(ds, temp_node, dataAdaptor);
        }
        else
        {
          conduit::Node &temp_node = root;

	  // tmp, check for validity
	  std::cout << "printing out contents of the conduit node/tree " << std::endl;

	  temp_node.print();

	  // following example from here: https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#uniform
	  conduit::Node verify_info;
	  if (!conduit::blueprint::mesh::verify(temp_node, verify_info))
	  {
	     std::cout << "Verify failed! " << std::endl;
	     verify_info.print();
	  }


          // FIXME -- check retuirn for error
          ::PassData(ds, temp_node, dataAdaptor);
        }
      }
      itr->GoToNextItem();
    }
  }
  else if (svtkDataSet *ds = svtkDataSet::SafeDownCast(obj))
  {
    conduit::Node &temp_node = root;

    // FIXME -- check retuirn for error
    ::PassData(ds, temp_node, dataAdaptor);
  }
  else
  {
    SENSEI_ERROR("Data object " << obj->GetClassName()
      << " is not supported.");
    return( -1 );
  }

#ifdef DEBUG_SAVE_DATA
  std::cout << "----------------------------" << std::endl;
  std::cout << "i: " << i << " size: " << size << std::endl;
  std::cout << "ACTIONS" << std::endl;
  this->actionsNode.print();
  std::cout << "NODE" << std::endl;
  root.print();
  std::cout << "----------------------------" << std::endl;

  DebugSaveAscentData( root, this->optionsNode );
#endif

#if 1
  // wes 4/20/2021
  // invoke blueprint method to verify that we have created something valid
  // 
  conduit::Node verify_info;
  if (!conduit::blueprint::mesh::verify(root, verify_info))
  {
     std::cout << "Mesh verify failed!" << std::endl;
//     std::cout << verify_info.to_yaml() << std::endl;
  }
  else
  {
     std::cout << "Mesh verify success!" << std::endl;
     std::cout << root.to_yaml() << std::endl;
//     std::cout << veri
  }

  std::cout << "Contents of this->actionsNode(): " << std::endl;
  std::cout << this->actionsNode.to_yaml() << std::endl;
//  std::cout << ascent::about() << std::endl;

  // temp wes 4/20/2021: write a file and send to Matt because
  // it causes a crash in vtk-m, something they thought was fixed
  conduit::relay::io::save(root, "sensei-ascent-demo.json");

#endif

#if 0
  // wes 7/1/2021
  std::cout << ascent::about() << std::endl;
#endif


  this->_ascent.publish(root);
  this->_ascent.execute(this->actionsNode);


  root.reset();

  return( true );
}

//------------------------------------------------------------------------------
int AscentAnalysisAdaptor::Finalize()
{
  this->_ascent.close();
  this->Fields.clear();

  return( 0 );
}

}   // namespace sensei

