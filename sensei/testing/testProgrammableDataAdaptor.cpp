#include "ProgrammableDataAdaptor.h"
#include "Histogram.h"

#include <vtkDataObject.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>

#include <vector>
#include <iostream>

#include <mpi.h>

using std::cerr;
using std::endl;

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  //
  //                                              *
  //                                              * *
  //                                            * * *
  //                                            * * * *
  //                                          * * * * *
  //                                        * * * * * * *
  std::vector<unsigned int> baselineHist = {1,2,4,6,5,3,1};
  //                                        0,1,2,3,4,5,6

  std::vector<double> data = {0, 1,1, 2,2,2,2,
    3,3,3,3,3,3, 4,4,4,4,4, 5,5,5, 6};

  // get mesh callback
  auto getMesh = [&data](bool) -> vtkDataObject*
    {
    cerr << "===getMesh" << endl;
    vtkImageData *im = vtkImageData::New();
    im->SetDimensions(data.size(), 1, 1);
    return im;
    };

  // add array callback
  auto addArray = [&data](vtkDataObject *mesh, int assoc,
    const std::string &name) -> bool
    {
    cerr << "===addArray" << endl;
    if ((assoc == vtkDataObject::POINT) && (name == "data"))
      {
      vtkDoubleArray *da = vtkDoubleArray::New();
      da->SetName("data");
      da->SetArray(data.data(), data.size(), 1);

      static_cast<vtkImageData*>(mesh)->GetPointData()->AddArray(da);
      da->Delete();
      return true;
      }
    return false;
    };

  // number of arrays callback
  auto getNumArrays = [](int assoc) -> unsigned int
    {
    cerr << "===getNumArrays" << endl;
    if (assoc == vtkDataObject::POINT)
      return 1;
    return 0;
    };

  // array name callback
  auto getArrayName = [](int assoc, unsigned int id) -> std::string
    {
    cerr << "===getArrayName" << endl;
    if ((assoc == vtkDataObject::POINT) && (id == 0))
      return "data";
    return "";
    };

  // release data callback
  auto releaseData = []()
    {
    cerr << "===releaseData" << endl;
    };

  sensei::ProgrammableDataAdaptor *pda = sensei::ProgrammableDataAdaptor::New();
  pda->SetGetMeshCallback(getMesh);
  pda->SetAddArrayCallback(addArray);
  pda->SetGetNumberOfArraysCallback(getNumArrays);
  pda->SetGetArrayNameCallback(getArrayName);
  pda->SetReleaseDataCallback(releaseData);

  int result = -1;
  if (pda->GetNumberOfArrays(vtkDataObject::POINT) > 0)
    {
    sensei::Histogram *ha = sensei::Histogram::New();

    ha->Initialize(MPI_COMM_WORLD, 7, vtkDataObject::POINT,
      pda->GetArrayName(vtkDataObject::POINT, 0));

    ha->Execute(pda);

    pda->ReleaseData();

    double min = 0.0;
    double max = 0.0;
    std::vector<unsigned int> hist;
    ha->GetHistogram(min, max, hist);

    if (hist == baselineHist)
      result = 0;

    pda->Delete();
    ha->Delete();
    }

  MPI_Finalize();

  return result;
}
