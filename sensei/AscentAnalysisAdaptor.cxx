#include "AscentAnalysisAdaptor.h"
#include "DataAdaptor.h"
#include "Error.h"
#include <conduit.hpp>
#include <conduit_blueprint.hpp>

#include <vtkObjectFactory.h>
#include <vtkDataObject.h>
#include <vtkDataArray.h>
#include <vtkFieldData.h>
#include <vtkDataSet.h>
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
void VTK_To_Fields(DataAdaptor* dataAdaptor, conduit::Node& node)
{

return;
}

//------------------------------------------------------------------------------

void VTK_To_Topology(DataAdaptor* dataAdaptor, conduit::Node& node)
{

return;
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

    return 1;
  }
  else if(rectilinear != nullptr)
  {
    node["coordsets/coords/type"] = "rectilinear";

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
    if(z != nullptr)
    {
      int size = z->GetNumberOfTuples();
      double *ptr  = (double *)z->GetVoidPointer(0);
      std::vector<conduit::float64> vals(size, 0.0);
      for(int i = 0; i < size; i++)
        vals[i] = ptr[i];
      node["coordsets/coords/values/z"].set(vals);
    }
    return 1;
  }
  else if(structured != nullptr)
  {

    return 1;
  }
  else if(unstructured != nullptr)
  {

    return 1;
  }
  else
  {
    SENSEI_ERROR("Mesh type not supported")
    return -1;
  }
}

//------------------------------------------------------------------------------
void
AscentAnalysisAdaptor::Initialize()
{

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
//  obj->Print(std::cout); 
  unsigned int nArrays = 0;
  dataAdaptor->GetNumberOfArrays("mesh", 1, nArrays);
  std::cout << "Number of arrays " << nArrays << std::endl;
  std::string arrayName;
  dataAdaptor->GetArrayName("mesh", 1, 0, arrayName);
  std::cout << "ArrayName " << arrayName <<std::endl;
  dataAdaptor->AddArray(obj, "mesh", 1, "dist");
  obj->Print(std::cout);
  conduit::Node temp_node;
  int count = 0;

  vtkCompositeDataSet *cds = vtkCompositeDataSet::SafeDownCast(obj);
  if(cds != nullptr)
  {
//    conduit::blueprint::mesh::to_multi_domain(node, node);
    vtkCompositeDataIterator *itr = cds->NewIterator();
    itr->SkipEmptyNodesOn();
    itr->InitTraversal();
    while(!itr->IsDoneWithTraversal())
    {
      vtkDataObject *obj2 = cds->GetDataSet(itr);
      if(obj2 != nullptr && vtkDataSet::SafeDownCast(obj2) != nullptr)
      {
        temp_node.reset();
        std::stringstream ss;
        ss << "domain_00000" << count;
        std::string domain = ss.str();
        count++;
        vtkDataSet *ds = vtkDataSet::SafeDownCast(obj2);
        VTK_To_Coordsets(ds, temp_node);
        temp_node.print();
        conduit::Node& build = node[domain].append();
        build.set(temp_node);

      }
      itr->GoToNextItem();
    }
  }
  else if(vtkDataSet::SafeDownCast(obj) != nullptr)
  {
    vtkDataSet::SafeDownCast(obj)->Print(std::cout);
  }
  else
  {
    SENSEI_ERROR("Data object is not supported.")
  }
  node.print();
return true;
}


//------------------------------------------------------------------------------
int
AscentAnalysisAdaptor::Finalize()
{

return 0;
}

}
