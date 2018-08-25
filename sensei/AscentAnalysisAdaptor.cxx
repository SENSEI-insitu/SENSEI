#include "AscentAnalysisAdaptor.h"
#include "DataAdaptor.h"
#include "Error.h"
#include <conduit.hpp>
#include <conduit_blueprint.hpp>

#include <vtkObjectFactory.h>
#include <vtkCellArray.h>
#include <vtkDataObject.h>
#include <vtkDataArray.h>
#include <vtkFieldData.h>
#include <vtkDataSet.h>
#include <vtkDataSetAttributes.h>
#include <vtkCompositeDataIterator.h>
#include <vtkCompositeDataSet.h>
#include <vtkImageData.h>
#include <vtkRectilinearGrid.h>
#include <vtkStructuredGrid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellData.h>
#include <vtkPointData.h>


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



//------------------------------------------------------------------------------
void
GetShape(std::string &shape, int type)
{
  if(type == 1) shape = "point";
  else if(type == 2) shape = "line";
  else if(type == 3) shape = "tri";
  else if(type == 4) shape = "quad";
  else if(type == 8) shape = "hex";
  else SENSEI_ERROR("Error: Unsupported element shape");
  return;
}

//------------------------------------------------------------------------------
int
VTK_To_Fields(vtkDataSet* ds, conduit::Node& node, std::string arrayName, vtkDataObject *obj)
{
  vtkImageData *uniform             = vtkImageData::SafeDownCast(ds);
  vtkRectilinearGrid *rectilinear   = vtkRectilinearGrid::SafeDownCast(ds);
  vtkStructuredGrid *structured     = vtkStructuredGrid::SafeDownCast(ds);
  vtkUnstructuredGrid *unstructured = vtkUnstructuredGrid::SafeDownCast(ds);

  std::stringstream ss;
  ss << "fields/" << arrayName << "/association";
  std::string assocPath = ss.str();
  ss.str(std::string());

  ss << "fields/" << arrayName << "/volume_dependent";
  std::string volPath = ss.str();
  ss.str(std::string());

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

  ss << "fields/" << arrayName << "/grid_function";
  std::string gridPath = ss.str();
  ss.str(std::string());

  ss << "fields/" << arrayName << "/matset";
  std::string matPath = ss.str();
  ss.str(std::string());

  ss << "fields/" << arrayName << "/matset_values";
  std::string matValPath = ss.str();
  ss.str(std::string());


  if(uniform != nullptr)
  {
    vtkDataSetAttributes *attrP = obj->GetAttributes(vtkDataObject::POINT);
    vtkDataSetAttributes *attrC = obj->GetAttributes(vtkDataObject::CELL);
    vtkDataArray *point = attrP->GetArray(0);
    vtkDataArray *cell = attrC->GetArray(0);

    if(point)
    {
      node[assocPath] = "vertex";

      int components = point->GetNumberOfComponents();
      int tuples     = point->GetNumberOfTuples();

      if(components == 1)
      {
        node[typePath] = "scalar";

        std::vector<conduit::float64> vals(tuples,0.0);
        for(int i = 0; i < tuples; i++)
        {
          double tuple[1] = {0.0};
          point->GetTuple(i, tuple);
          vals[i] = tuple[0];
        }
        node[valPath].set(vals);
      }
      else if(components == 2)
      {
        node[typePath] = "vector";

        int size = tuples;
        std::vector<conduit::float64> uVals(size, 0.0);
        std::vector<conduit::float64> vVals(size, 0.0);
        for(int i = 0; i < size; i++)
        {
          uVals[i] = point->GetComponent(i, 0);
          vVals[i] = point->GetComponent(i, 1);
        }
        node[uValPath].set(uVals);
        node[vValPath].set(vVals);
      }
      else if(components == 3)
      {
        node[typePath] = "vector";

        int size = tuples;
        std::vector<conduit::float64> uVals(size, 0.0);
        std::vector<conduit::float64> vVals(size, 0.0);
        std::vector<conduit::float64> wVals(size, 0.0);
        for(int i = 0; i < size; i++)
        {
          uVals[i] = point->GetComponent(i, 0);
          vVals[i] = point->GetComponent(i, 1);
          wVals[i] = point->GetComponent(i, 2);
        }
        node[uValPath].set(uVals);
        node[vValPath].set(vVals);
        node[wValPath].set(wVals);

      }
      else
      {
        SENSEI_ERROR("Too many components associated with " << arrayName)
        return -1;
      }
    }
    else if(cell)
    {
      node[assocPath] = "element";

      int components = cell->GetNumberOfComponents();
      int tuples     = cell->GetNumberOfTuples();

      if(components == 1)
      {
        node[typePath] = "scalar";

        std::vector<conduit::float64> vals(tuples,0.0);
        for(int i = 0; i < tuples; i++)
          vals[i] = *cell->GetTuple(i);
        node[valPath].set(vals);
      }
      else if(components == 2)
      {
        node[typePath] = "vector";

        int size = tuples;
        std::vector<conduit::float64> uVals(size, 0.0);
        std::vector<conduit::float64> vVals(size, 0.0);
        for(int i = 0; i < size; i++)
        {
          uVals[i] = cell->GetComponent(i, 0);
          vVals[i] = cell->GetComponent(i, 1);
        }
        node[uValPath].set(uVals);
        node[vValPath].set(vVals);
      }
      else if(components == 3)
      {
        node[typePath] = "vector";

        int size = tuples;
        std::vector<conduit::float64> uVals(size, 0.0);
        std::vector<conduit::float64> vVals(size, 0.0);
        std::vector<conduit::float64> wVals(size, 0.0);
        for(int i = 0; i < size; i++)
        {
          uVals[i] = cell->GetComponent(i, 0);
          vVals[i] = cell->GetComponent(i, 1);
          wVals[i] = cell->GetComponent(i, 2);
        }
        node[uValPath].set(uVals);
        node[vValPath].set(vVals);
        node[wValPath].set(wVals);

      }
      else
      {
        SENSEI_ERROR("Too many components associated with " << arrayName)
        return -1;
      }
    }
    else
    {
      SENSEI_ERROR("Field Orientation unsupported :: Must be vertext or element")
      return -1;
    }
    node[topoPath] = "mesh";
  }
  else if(rectilinear != nullptr)
  {
    vtkDataSetAttributes *attrP = obj->GetAttributes(vtkDataObject::POINT);
    vtkDataSetAttributes *attrC = obj->GetAttributes(vtkDataObject::CELL);
    vtkDataArray *point = attrP->GetArray(0);
    vtkDataArray *cell = attrC->GetArray(0);

    if(point)
    {
      node[assocPath] = "vertex";

      int components = point->GetNumberOfComponents();
      int tuples     = point->GetNumberOfTuples();

      if(components == 1)
      {
        node[typePath] = "scalar";

        std::vector<conduit::float64> vals(tuples,0.0);
        for(int i = 0; i < tuples; i++)
        {
          double tuple[1] = {0.0};
          point->GetTuple(i, tuple);
          vals[i] = tuple[0];
        }
        node[valPath].set(vals);
      }
      else if(components == 2)
      {
        node[typePath] = "vector";

        int size = tuples;
        std::vector<conduit::float64> uVals(size, 0.0);
        std::vector<conduit::float64> vVals(size, 0.0);
        for(int i = 0; i < size; i++)
        {
          uVals[i] = point->GetComponent(i, 0);
          vVals[i] = point->GetComponent(i, 1);
        }
        node[uValPath].set(uVals);
        node[vValPath].set(vVals);
      }
      else if(components == 3)
      {
        node[typePath] = "vector";

        int size = tuples;
        std::vector<conduit::float64> uVals(size, 0.0);
        std::vector<conduit::float64> vVals(size, 0.0);
        std::vector<conduit::float64> wVals(size, 0.0);
        for(int i = 0; i < size; i++)
        {
          uVals[i] = point->GetComponent(i, 0);
          vVals[i] = point->GetComponent(i, 1);
          wVals[i] = point->GetComponent(i, 2);
        }
        node[uValPath].set(uVals);
        node[vValPath].set(vVals);
        node[wValPath].set(wVals);

      }
      else
      {
        SENSEI_ERROR("Too many components associated with " << arrayName)
        return -1;
      }
    }
    else if(cell)
    {
      node[assocPath] = "element";

      int components = cell->GetNumberOfComponents();
      int tuples     = cell->GetNumberOfTuples();

      if(components == 1)
      {
        node[typePath] = "scalar";

        std::vector<conduit::float64> vals(tuples,0.0);
        for(int i = 0; i < tuples; i++)
          vals[i] = *cell->GetTuple(i);
        node[valPath].set(vals);
      }
      else if(components == 2)
      {
        node[typePath] = "vector";

        int size = tuples;
        std::vector<conduit::float64> uVals(size, 0.0);
        std::vector<conduit::float64> vVals(size, 0.0);
        for(int i = 0; i < size; i++)
        {
          uVals[i] = cell->GetComponent(i, 0);
          vVals[i] = cell->GetComponent(i, 1);
        }
        node[uValPath].set(uVals);
        node[vValPath].set(vVals);
      }
      else if(components == 3)
      {
        node[typePath] = "vector";

        int size = tuples;
        std::vector<conduit::float64> uVals(size, 0.0);
        std::vector<conduit::float64> vVals(size, 0.0);
        std::vector<conduit::float64> wVals(size, 0.0);
        for(int i = 0; i < size; i++)
        {
          uVals[i] = cell->GetComponent(i, 0);
          vVals[i] = cell->GetComponent(i, 1);
          wVals[i] = cell->GetComponent(i, 2);
        }
        node[uValPath].set(uVals);
        node[vValPath].set(vVals);
        node[wValPath].set(wVals);

      }
      else
      {
        SENSEI_ERROR("Too many components associated with " << arrayName)
        return -1;
      }
    }
    else
    {
      SENSEI_ERROR("Field Orientation unsupported :: Must be vertext or element")
      return -1;
    }
    node[topoPath] = "mesh";

  }
  else if(structured != nullptr)
  {
    vtkDataSetAttributes *attrP = obj->GetAttributes(vtkDataObject::POINT);
    vtkDataSetAttributes *attrC = obj->GetAttributes(vtkDataObject::CELL);
    vtkDataArray *point = attrP->GetArray(0);
    vtkDataArray *cell = attrC->GetArray(0);

    if(point)
    {
      node[assocPath] = "vertex";

      int components = point->GetNumberOfComponents();
      int tuples     = point->GetNumberOfTuples();

      if(components == 1)
      {
        node[typePath] = "scalar";

        std::vector<conduit::float64> vals(tuples,0.0);
        for(int i = 0; i < tuples; i++)
        {
          double tuple[1] = {0.0};
          point->GetTuple(i, tuple);
          vals[i] = tuple[0];
        }
        node[valPath].set(vals);
      }
      else if(components == 2)
      {
        node[typePath] = "vector";

        int size = tuples;
        std::vector<conduit::float64> uVals(size, 0.0);
        std::vector<conduit::float64> vVals(size, 0.0);
        for(int i = 0; i < size; i++)
        {
          uVals[i] = point->GetComponent(i, 0);
          vVals[i] = point->GetComponent(i, 1);
        }
        node[uValPath].set(uVals);
        node[vValPath].set(vVals);
      }
      else if(components == 3)
      {
        node[typePath] = "vector";

        int size = tuples;
        std::vector<conduit::float64> uVals(size, 0.0);
        std::vector<conduit::float64> vVals(size, 0.0);
        std::vector<conduit::float64> wVals(size, 0.0);
        for(int i = 0; i < size; i++)
        {
          uVals[i] = point->GetComponent(i, 0);
          vVals[i] = point->GetComponent(i, 1);
          wVals[i] = point->GetComponent(i, 2);
        }
        node[uValPath].set(uVals);
        node[vValPath].set(vVals);
        node[wValPath].set(wVals);

      }
      else
      {
        SENSEI_ERROR("Too many components associated with " << arrayName)
        return -1;
      }
    }
    else if(cell)
    {
      node[assocPath] = "element";

      int components = cell->GetNumberOfComponents();
      int tuples     = cell->GetNumberOfTuples();

      if(components == 1)
      {
        node[typePath] = "scalar";

        std::vector<conduit::float64> vals(tuples,0.0);
        for(int i = 0; i < tuples; i++)
          vals[i] = *cell->GetTuple(i);
        node[valPath].set(vals);
      }
      else if(components == 2)
      {
        node[typePath] = "vector";

        int size = tuples;
        std::vector<conduit::float64> uVals(size, 0.0);
        std::vector<conduit::float64> vVals(size, 0.0);
        for(int i = 0; i < size; i++)
        {
          uVals[i] = cell->GetComponent(i, 0);
          vVals[i] = cell->GetComponent(i, 1);
        }
        node[uValPath].set(uVals);
        node[vValPath].set(vVals);
      }
      else if(components == 3)
      {
        node[typePath] = "vector";

        int size = tuples;
        std::vector<conduit::float64> uVals(size, 0.0);
        std::vector<conduit::float64> vVals(size, 0.0);
        std::vector<conduit::float64> wVals(size, 0.0);
        for(int i = 0; i < size; i++)
        {
          uVals[i] = cell->GetComponent(i, 0);
          vVals[i] = cell->GetComponent(i, 1);
          wVals[i] = cell->GetComponent(i, 2);
        }
        node[uValPath].set(uVals);
        node[vValPath].set(vVals);
        node[wValPath].set(wVals);

      }
      else
      {
        SENSEI_ERROR("Too many components associated with " << arrayName)
        return -1;
      }
    }
    else
    {
      SENSEI_ERROR("Field Orientation unsupported :: Must be vertext or element")
      return -1;
    }
    node[topoPath] = "mesh";

  }
  else if(unstructured != nullptr)
  {
    vtkDataSetAttributes *attrP = obj->GetAttributes(vtkDataObject::POINT);
    vtkDataSetAttributes *attrC = obj->GetAttributes(vtkDataObject::CELL);
    vtkDataArray *point = attrP->GetArray(0);
    vtkDataArray *cell = attrC->GetArray(0);

    if(point)
    {
      node[assocPath] = "vertex";

      int components = point->GetNumberOfComponents();
      int tuples     = point->GetNumberOfTuples();

      if(components == 1)
      {
        node[typePath] = "scalar";

        std::vector<conduit::float64> vals(tuples,0.0);
        for(int i = 0; i < tuples; i++)
        {
          double tuple[1] = {0.0};
          point->GetTuple(i, tuple);
          vals[i] = tuple[0];
        }
        node[valPath].set(vals);
      }
      else if(components == 2)
      {
        node[typePath] = "vector";

        int size = tuples;
        std::vector<conduit::float64> uVals(size, 0.0);
        std::vector<conduit::float64> vVals(size, 0.0);
        for(int i = 0; i < size; i++)
        {
          uVals[i] = point->GetComponent(i, 0);
          vVals[i] = point->GetComponent(i, 1);
        }
        node[uValPath].set(uVals);
        node[vValPath].set(vVals);
      }
      else if(components == 3)
      {
        node[typePath] = "vector";

        int size = tuples;
        std::vector<conduit::float64> uVals(size, 0.0);
        std::vector<conduit::float64> vVals(size, 0.0);
        std::vector<conduit::float64> wVals(size, 0.0);
        for(int i = 0; i < size; i++)
        {
          uVals[i] = point->GetComponent(i, 0);
          vVals[i] = point->GetComponent(i, 1);
          wVals[i] = point->GetComponent(i, 2);
        }
        node[uValPath].set(uVals);
        node[vValPath].set(vVals);
        node[wValPath].set(wVals);

      }
      else
      {
        SENSEI_ERROR("Too many components associated with " << arrayName)
        return -1;
      }
    }
//TODO: How vectors are arranged in vtk?? interwoven or each vector sequentially? 
    else if(cell)
    {
      node[assocPath] = "element";

      int components = cell->GetNumberOfComponents();
      int tuples     = cell->GetNumberOfTuples();

      if(components == 1)
      {
        node[typePath] = "scalar";

        std::vector<conduit::float64> vals(tuples,0.0);
        for(int i = 0; i < tuples; i++)
          vals[i] = *cell->GetTuple(i);
        node[valPath].set(vals);
      }
      else if(components == 2)
      {
        node[typePath] = "vector";

        int size = tuples;
        std::vector<conduit::float64> uVals(size, 0.0);
        std::vector<conduit::float64> vVals(size, 0.0);
        for(int i = 0; i < size; i++)
        {
          uVals[i] = cell->GetComponent(i, 0);
          vVals[i] = cell->GetComponent(i, 1);
        }
        node[uValPath].set(uVals);
        node[vValPath].set(vVals);
      }
      else if(components == 3)
      {
        node[typePath] = "vector";

        int size = tuples;
        std::vector<conduit::float64> uVals(size, 0.0);
        std::vector<conduit::float64> vVals(size, 0.0);
        std::vector<conduit::float64> wVals(size, 0.0);
        for(int i = 0; i < size; i++)
        {
          uVals[i] = cell->GetComponent(i, 0);
          vVals[i] = cell->GetComponent(i, 1);
          wVals[i] = cell->GetComponent(i, 2);
        }
        node[uValPath].set(uVals);
        node[vValPath].set(vVals);
        node[wValPath].set(wVals);

      }
      else
      {
        SENSEI_ERROR("Too many components associated with " << arrayName)
        return -1;
      }
    }
    else
    {
      SENSEI_ERROR("Field Orientation unsupported :: Must be vertext or element")
      return -1;
    }
    node[topoPath] = "mesh";
  }
  else
  {
    SENSEI_ERROR("Mesh structure not supported")
    return -1;
  }



  return 1;
}

//------------------------------------------------------------------------------

int
VTK_To_Topology(vtkDataSet* ds, conduit::Node& node)
{
  vtkImageData *uniform             = vtkImageData::SafeDownCast(ds);
  vtkRectilinearGrid *rectilinear   = vtkRectilinearGrid::SafeDownCast(ds);
  vtkStructuredGrid *structured     = vtkStructuredGrid::SafeDownCast(ds);
  vtkUnstructuredGrid *unstructured = vtkUnstructuredGrid::SafeDownCast(ds);

  if(uniform != nullptr)
  {
    node["topologies/mesh/type"]     = "uniform";
    node["topologies/mesh/coordset"] = "coords";

    int dims[3] = {0,0,0};
    uniform->GetDimensions(dims);
    double origin[3] = {0.0,0.0,0.0};
    uniform->GetOrigin(origin);

    node["topologies/mesh/elements/origin/i0"] = origin[0];
    node["topologies/mesh/elements/origin/j0"] = origin[1];
    if(dims[2] != 0 && dims[2] != 1)
      node["topologies/mesh/elements/origin/k0"] = origin[2];
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

    int dims[3] = {0,0,0};
    structured->GetDimensions(dims);

    node["topologies/mesh/elements/dims/i"] = dims[0] - 1;
    node["topologies/mesh/elements/dims/j"] = dims[1] - 1;
    if(dims[2] != 0 && dims[2] != 1)
      node["topologies/mesh/elements/dims/k"] = dims[2] - 1;
  }
  else if(unstructured != nullptr)
  {
    if(!unstructured->IsHomogeneous())
    {
      SENSEI_ERROR("Unstructured cells must be homogenous")
      return -1;
    }
    node["topologies/mesh/type"]     = "unstructured";
    node["topologies/mesh/coordset"] = "coords";

    vtkCellArray* cellarray = unstructured->GetCells();
    vtkIdType *ptr = cellarray->GetPointer();

    std::string shape;
    GetShape(shape, ptr[0]);
    node["topologies/mesh/elements/shape"] = shape;

    int ncells = unstructured->GetNumberOfCells();
    int connections = ncells*(ptr[0] + 1);
    std::vector<int> data(ncells*ptr[0], 0);

    int count = 0;
    for(int i = 0; i < connections; i++)
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
  }
  else
  {
    SENSEI_ERROR("Mesh structure not supported")
    return -1;
  }

return 1;
}


//------------------------------------------------------------------------------

int
VTK_To_Coordsets(vtkDataSet* ds, conduit::Node& node)
{
  vtkImageData *uniform             = vtkImageData::SafeDownCast(ds);
  vtkRectilinearGrid *rectilinear   = vtkRectilinearGrid::SafeDownCast(ds);
  vtkStructuredGrid *structured     = vtkStructuredGrid::SafeDownCast(ds);
  vtkUnstructuredGrid *unstructured = vtkUnstructuredGrid::SafeDownCast(ds);

  if(uniform != nullptr)
  {
    node["coordsets/coords/type"] = "uniform";

    int dims[3] = {0,0,0};
    uniform->GetDimensions(dims);
    node["coordsets/coords/dims/i"] = dims[0];
    node["coordsets/coords/dims/j"] = dims[1];
    node["coordsets/coords/dims/k"] = dims[2];

    double origin[3] = {0.0,0.0,0.0};
    uniform->GetOrigin(origin);
    node["coordsets/coords/origin/x"] = origin[0];
    node["coordsets/coords/origin/y"] = origin[1];
    node["coordsets/coords/origin/z"] = origin[2];

    double spacing[3] = {0.0,0.0,0.0};
    uniform->GetSpacing(spacing);
    node["coordsets/coords/spacing/dx"] = spacing[0];
    node["coordsets/coords/spacing/dy"] = spacing[1];
    node["coordsets/coords/spacing/dz"] = spacing[2];
  }
  else if(rectilinear != nullptr)
  {
    node["coordsets/coords/type"] = "rectilinear";

    int dims[3] = {0,0,0};
    rectilinear->GetDimensions(dims);
    //std::cout << "dims " << dims[0] << " " << dims[1] << " " << dims[2] << std::endl;

    vtkDataArray *x = rectilinear->GetXCoordinates();
    vtkDataArray *y = rectilinear->GetYCoordinates();
    vtkDataArray *z = rectilinear->GetZCoordinates();
    if(x != nullptr)
    {
      int size = x->GetNumberOfTuples();
      double *ptr  = (double *)x->GetVoidPointer(0);
      std::vector<conduit::float64> vals(size, 0.0);
      for(int i = 0; i < size; i++)
        vals[i] = ptr[i];
      node["coordsets/coords/values/x"].set(vals);
    }
    if(y != nullptr)
    {
      int size = y->GetNumberOfTuples();
      double *ptr  = (double *)y->GetVoidPointer(0);
      std::vector<conduit::float64> vals(size, 0.0);
      for(int i = 0; i < size; i++)
        vals[i] = ptr[i];
      node["coordsets/coords/values/y"].set(vals);
    }
    if(z != nullptr && dims[2] != 0 && dims[2] != 1)
    {
      int size = z->GetNumberOfTuples();
      double *ptr  = (double *)z->GetVoidPointer(0);
      std::vector<conduit::float64> vals(size, 0.0);
      for(int i = 0; i < size; i++)
        vals[i] = ptr[i];
      node["coordsets/coords/values/z"].set(vals);
    }
  }
  else if(structured != nullptr)
  {
    node["coordsets/coords/type"] = "explicit";

    int dims[3] = {0,0,0};
    structured->GetDimensions(dims);

    int numPoints = structured->GetPoints()->GetNumberOfPoints();
    double point[3] = {0,0,0};
    std::vector<conduit::float64> x(numPoints,0.0); 
    std::vector<conduit::float64> y(numPoints,0.0); 
    std::vector<conduit::float64> z(numPoints,0.0); 
    for(int i = 0; i < numPoints; i++)
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
    double point[3] = {0,0,0};
    std::vector<conduit::float64> x(numPoints,0.0); 
    std::vector<conduit::float64> y(numPoints,0.0); 
    std::vector<conduit::float64> z(numPoints,0.0); 
    for(int i = 0; i < numPoints; i++)
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
    SENSEI_ERROR("Mesh type not supported")
    return -1;
  }
  return 1;
}

//------------------------------------------------------------------------------
void
AscentAnalysisAdaptor::Initialize(conduit::Node actions)
{
 this->a.open();
 this->actionNode = actions; 

return;
}


//------------------------------------------------------------------------------
bool
AscentAnalysisAdaptor::Execute(DataAdaptor* dataAdaptor)
{
  std::cout << "IN EzXECUTE" << std::endl;
  conduit::Node node;
  vtkDataObject* obj = nullptr;
  if(dataAdaptor->GetMesh("mesh", false, obj))
  {
    SENSEI_ERROR("Failed to get mesh")
    return -1;
  }

  std::string arrayName;

  conduit::Node actions;
  conduit::Node &add_actions = actions.append();

  if(this->actionNode["action"].as_string() ==  "add_scenes")
  {
    arrayName = this->actionNode["scenes/scene1/plots/plt1/params/field"].as_string();
    add_actions["action"] = "add_scenes";
    add_actions["scenes"] = this->actionNode["scenes"];
  }
  else if(this->actionNode["action"].as_string() == "add_pipelines")
  {
      std::cout << "HERE" << std::endl;
    arrayName = this->actionNode["pipelines/pl1/f1/params/field"].as_string();
    add_actions["action"] = "add_pipelines";
    add_actions["pipelines"]    = this->actionNode["pipelines"];

    conduit::Node scenes;
    scenes["action"] = "add_scenes";
    scenes["scenes/scene1/plots/plt1/params/field"] = arrayName;
    scenes["scenes/scene1/plots/plt1/type"] = "pseudocolor";
    scenes["scenes/scene1/plots/plt1/pipeline"] = "pl1";

    actions.append() = scenes;
  }
  actions.append()["action"] = "execute";
  actions.print();


  std::cout << "ArrayName " << arrayName <<std::endl;
  dataAdaptor->AddArray(obj, "mesh", 1, arrayName);
  obj->Print(std::cout);
  conduit::Node temp_node;

  vtkCompositeDataSet *cds = vtkCompositeDataSet::SafeDownCast(obj);
  if(cds != nullptr)
  {
    vtkCompositeDataIterator *itr = cds->NewIterator();
    itr->SkipEmptyNodesOn();
    itr->InitTraversal();
    while(!itr->IsDoneWithTraversal())
    {
      vtkDataObject *obj2 = cds->GetDataSet(itr);
      if(obj2 != nullptr && vtkDataSet::SafeDownCast(obj2) != nullptr)
      {
        vtkDataSet *ds = vtkDataSet::SafeDownCast(obj2);

        temp_node.reset();
        VTK_To_Coordsets(ds, temp_node);
        VTK_To_Topology(ds, temp_node);
        VTK_To_Fields(ds, temp_node, arrayName, obj2);

        conduit::Node& build = node.append();
        build.set(temp_node);

      }
      itr->GoToNextItem();
    }
  }
  else if(vtkDataSet::SafeDownCast(obj) != nullptr)
  {
    vtkDataSet *ds = vtkDataSet::SafeDownCast(obj);

    temp_node.reset();
    VTK_To_Coordsets(ds, temp_node);
    VTK_To_Topology(ds, temp_node);
    VTK_To_Fields(ds, temp_node, arrayName, obj);

    conduit::Node& build = node.append();
    build.set(temp_node);
  }
  else
  {
    SENSEI_ERROR("Data object is not supported.")
  }
  std::cout << "NODE PRINT" << std::endl;
  node.print();

  this->a.publish(node);
  this->a.execute(actions);

  return true;
}


//------------------------------------------------------------------------------
int
AscentAnalysisAdaptor::Finalize()
{
this->a.close();
return 0;
}

}
