#include "ADIOS1AnalysisAdaptor.h"
#include "ADIOS1DataAdaptor.h"
#include "HDF5AnalysisAdaptor.h"
#include "HDF5DataAdaptor.h"
#include "VTKDataAdaptor.h"
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCharArray.h>
#include <vtkCompositeDataIterator.h>
#include <vtkDataArray.h>
#include <vtkDataArrayAccessor.h>
#include <vtkDataSetAttributes.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIdTypeArray.h>
#include <vtkImageData.h>
#include <vtkIndent.h>
#include <vtkIntArray.h>
#include <vtkLongArray.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnsignedIntArray.h>
#include <vtkUnsignedLongArray.h>

#include <stdlib.h>

using A1DataAdaptorPtr = vtkSmartPointer<sensei::ADIOS1DataAdaptor>;
using H5DataAdaptorPtr = vtkSmartPointer<sensei::HDF5DataAdaptor>;
using A1AnalysisAdaptorPtr = vtkSmartPointer<sensei::ADIOS1AnalysisAdaptor>;
using H5AnalysisAdaptorPtr = vtkSmartPointer<sensei::HDF5AnalysisAdaptor>;

void get_data_arrays(unsigned long size, vtkDataSetAttributes* dsa);

class AAWrap
{
public:
  AAWrap(A1AnalysisAdaptorPtr& adios1) { _adios1 = adios1; }
  AAWrap(H5AnalysisAdaptorPtr& h5) { _h5 = h5; }

  H5AnalysisAdaptorPtr _h5 = nullptr;
  A1AnalysisAdaptorPtr _adios1 = nullptr;

  sensei::AnalysisAdaptor* GetAA()
  {
    if (_h5 != NULL)
      return _h5;
    return _adios1;
  }
};
class TimedAdaptorWrap
{
public:
  TimedAdaptorWrap(A1DataAdaptorPtr& adios1) { _adios1 = adios1; }
  TimedAdaptorWrap(H5DataAdaptorPtr& h5) { _h5 = h5; }

  H5DataAdaptorPtr _h5 = nullptr;
  A1DataAdaptorPtr _adios1 = nullptr;
  /*
  TimedAdaptorWrap(sensei::HDF5DataAdaptor* h5) {_h5 = h5;}
  TimedAdaptorWrap(sensei::ADIOS1DataAdaptor* adios1) {_adios1 = adios1;}

  sensei::HDF5DataAdaptor* _h5 = NULL;  // temporary so can compile
  sensei::ADIOS1DataAdaptor* _adios1 = NULL;
  */

  ~TimedAdaptorWrap() {}

  sensei::DataAdaptor* GetDA()
  {
    if (_h5 != NULL)
      return _h5;
    return _adios1;
  }

  int Close()
  {
    if (_h5 != NULL)
      return _h5->CloseStream();
    return _adios1->CloseStream();
  }

  bool Advance()
  {
    if (_h5 != NULL)
      return _h5->AdvanceStream();
    return _adios1->AdvanceStream();
  }
};

//
// reading
//
int check_array(vtkDataArray* array)
{
  int n_comp = array->GetNumberOfComponents();
  int n_tuple = array->GetNumberOfTuples();
  int typeSize = array->GetElementComponentSize();

  // int total = n_comp * n_tuple;

  if (n_comp == 1)
    {
      for (int i = 0; i < n_tuple; i++)
        {
          // this is how array values are setup in this example
          if (array->GetTuple1(i) != i * typeSize)
            {
              std::cout << "............. oh no at " << i << " th.. "
                        << array->GetTuple1(i) << " != " << i * typeSize
                        << std::endl
                        << std::endl;
              return -1;
            }
        }
    }
  return 0;
}

int readMe(TimedAdaptorWrap* daWrap, MPI_Comm& comm)
{
  int rank, n_ranks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_ranks);

  sensei::DataAdaptor* da = daWrap->GetDA();

#ifdef DBCK
  // initialize the analysis adaptor
  sensei::ADIOS1AnalysisAdaptor* aw = sensei::ADIOS1AnalysisAdaptor::New();

  aw->SetFileName("dbck");
  aw->SetMethod("MPI");
#endif

  int n_steps = 0;
  int retval = 0;
  while (true)
    {
      double t = da->GetDataTime();
      int it = da->GetDataTimeStep();
      if (rank == 0)
        std::cout << "\n===> received step: " << it << " time: " << t
                  << std::endl;
      ;

      unsigned int nMeshes;
      da->GetNumberOfMeshes(nMeshes);

      unsigned int i = 0;
      while (i < nMeshes)
        {
          sensei::MeshMetadataPtr mmd = sensei::MeshMetadata::New();

          if (da->GetMeshMetadata(i, mmd))
            {
              std::cerr << "Unable to get metadata" << std::endl;
              break;
            }

          std::string meshName = mmd->MeshName;

          vtkDataObject* mesh = nullptr;
          da->GetMesh(meshName, false, mesh);
// request each array
#ifdef BEFORE
          int assoc_ids[2] = { vtkDataObject::POINT, vtkDataObject::CELL };
          std::string assoc_names[2] = { "point", "cell" };
          for (int k = 0; k < 2; k++)
            {
              unsigned int n_arrays;
              da->GetNumberOfArrays(meshName, assoc_ids[k], n_arrays);

              for (unsigned int j = 0; j < n_arrays; j++)
                {
                  std::string array_name;
                  da->GetArrayName(meshName, assoc_ids[k], j, array_name);
                  da->AddArray(mesh, meshName, assoc_ids[k], array_name);
                }
            }
#else
          unsigned int n_arrays = mmd->NumArrays;

          for (unsigned int j = 0; j < n_arrays; j++)
            {
              std::string arrayName = mmd->ArrayName[j];
              da->AddArray(mesh,
                           meshName,
                           /*assoc_ids[k],*/ mmd->ArrayCentering[j],
                           arrayName);
            }
#endif

          /*
          # this often will cause segv's if the dataset has been
          # improperly constructed, thus serves as a good check

          str_rep = str(ds)
          */
          // check the arrays have the expected data

          if (mesh == NULL)
            break;

          vtkMultiBlockDataSet* ds = vtkMultiBlockDataSet::SafeDownCast(mesh);

          if (ds == NULL)
            {
              std::cerr
                << "Unexpected mesh!! Not multiblock, but got this class: "
                << mesh->GetClassName() << std::endl;
              retval = -1;
              break;
            }
          vtkCompositeDataIterator* it = ds->NewIterator();

          while (!it->IsDoneWithTraversal())
            {
              vtkDataObject* curr = it->GetCurrentDataObject();

              vtkDataSet* bds = vtkDataSet::SafeDownCast(curr);
              if (bds == NULL)
                {
                  std::cerr << " surprise! " << std::endl;
                  break;
                }

              // int idx = it->GetCurrentFlatIndex();
              // std::cout<<"  .. info:   block:"<<idx<<"
              // classname="<<curr->GetClassName()<<"  numArrays =
              // "<<n_arrays<<" totalBlocks="<<mmd->NumBlocks<<std::endl;
              // std::cout<<"  .. info:
              // numBLocksLocal="<<mmd->NumBlocksLocal[0]<<"
              // "<<mmd->NumBlocksLocal.size()<<std::endl;
              n_arrays = mmd->NumArrays;
#ifdef BEFORE
#else
              for (unsigned int j = 0; j < n_arrays; j++)
                {
                  vtkDataArray* array;
                  if (mmd->ArrayCentering[j] == vtkDataObject::POINT)
                    {
                      array = bds->GetPointData()->GetArray(
                        mmd->ArrayName[j].c_str());
                    }
                  else
                    {
                      array =
                        bds->GetCellData()->GetArray(mmd->ArrayName[j].c_str());
                    }

		  if (mmd->ArrayName[j].find("BlockOwner") == std::string::npos) 
		    {
		      if (check_array(array))
			{
			  std::cerr << "Test failed on array " << j
				    << " name=" << array->GetName() << std::endl;
			  retval = -1;
			  break;
			}
		    }
                }
#endif
              it->GoToNextItem();
            } // while it
          it->Delete();
          i += 1;
          mesh->Delete();
        } // while mesh

      n_steps += 1;

#ifdef DBCK
      aw->Execute(da);
#endif
      da->ReleaseData();
      if (daWrap->Advance())
        {
          break;
        }

    } // while true;

  // close down the stream
  daWrap->Close();
  if (rank == 0)
    std::cout << "closed stream after receiving " << n_steps << " steps."
              << std::endl;

#ifdef DBCK
  aw->Finalize();
#endif
  return retval;
}

/////////////// writing out /////////////////

vtkPolyData* points_to_polydata(std::vector<float>& x,
                                std::vector<float>& y,
                                std::vector<float>& z)
{
  vtkPolyData* pd = vtkPolyData::New();
  unsigned long nx = x.size();

  vtkFloatArray* vxyz = vtkFloatArray::New();
  vxyz->SetNumberOfComponents(3);
  vxyz->SetNumberOfTuples(nx);
  for (unsigned long i = 0; i < nx; i++)
    {
      vxyz->SetTuple3(i, x[i], y[i], z[i]);
    }

  vtkPoints* pts = vtkPoints::New();
  pts->SetData(vxyz);

  /*
  vtkIntArray* cids = vtkIntArray::New();
  cids->SetNumberOfTuples(2*nx);
  cids->SetNumberOfComponents(1);
  for (unsigned long i=0; i<nx; i++) {
    cids->SetTuple1(i*2, 1);
    cids->SetTuple1(i*2+ 1, i);
    }*/

  vtkIdTypeArray* cellVals = vtkIdTypeArray::New();
  cellVals->SetNumberOfComponents(1);
  cellVals->SetNumberOfTuples(2 * nx);
  for (unsigned long i = 0; i < nx; i++)
    {
      cellVals->SetTuple1(i * 2, 1);
      cellVals->SetTuple1(i * 2 + 1, i);
    }

  vtkCellArray* cells = vtkCellArray::New();
  cells->SetCells(nx, cellVals);

  pd->SetPoints(pts);
  pd->SetVerts(cells);

  get_data_arrays(nx, pd->GetPointData());
  get_data_arrays(nx, pd->GetCellData());

  return pd;
  /*
  nx = len(x)
  # points
  xyz = np.zeros(3*nx, dtype=np.float32)
  xyz[::3] = x[:]
  xyz[1::3] = y[:]
  xyz[2::3] = z[:]
  vxyz = vtknp.numpy_to_vtk(xyz, deep=1)
  vxyz.SetNumberOfComponents(3)
  vxyz.SetNumberOfTuples(nx)
  pts = vtk.vtkPoints()
  pts.SetData(vxyz)
  # cells
  cids = np.empty(2*nx, dtype=np.int32)
  cids[::2] = 1
  cids[1::2] = np.arange(0,nx,dtype=np.int32)
  cells = vtk.vtkCellArray()
  cells.SetCells(nx, vtknp.numpy_to_vtk(cids, \
      deep=1, array_type=vtk.VTK_ID_TYPE))
  # package it all up in a poly data set
  pd = vtk.vtkPolyData()
  pd.SetPoints(pts)
  pd.SetVerts(cells)
  # add some scalar data
  get_data_arrays(nx, pd.GetPointData())
  get_data_arrays(nx, pd.GetCellData())
  return pd
  */
}

vtkPolyData* get_polydata(unsigned long nx)
{
  srand(2);

  std::vector<float> x;
  x.reserve(nx);
  std::vector<float> y;
  y.reserve(nx);
  std::vector<float> z;
  z.reserve(nx);
  for (unsigned long i = 0; i < nx; i++)
    {
      x.push_back(((float)rand() / (RAND_MAX)));
      y.push_back(((float)rand() / (RAND_MAX)));
      z.push_back(((float)rand() / (RAND_MAX)));
    }

  vtkPolyData* pd = points_to_polydata(x, y, z);
  // std::cout<<"......... poly:: points_to_poly  above ... "<<nx<<std::endl;

  // std::cout<<"......... poly:: point array now ... check
  // "<<pd->GetPointData()->GetNumberOfArrays()<<std::endl;
  get_data_arrays(nx, pd->GetPointData());
  // std::cout<<"......... poly:: point array above ...
  // "<<pd->GetPointData()->GetNumberOfArrays()<<std::endl;
  get_data_arrays(nx, pd->GetCellData());
  // std::cout<<"......... poly:: cell array above ... "<<std::endl;

  return pd;
}

template<class T>
vtkDataArray* generate_array(const char* name,
                             unsigned long size,
                             vtkDataArray* result)
{
  std::vector<T> values(size);

  unsigned long i = 0;
  for (i = 0; i < size; i++)
    {
      values[i] = i * sizeof(T);
    }

  // std::cout<<" generate_array: name="<<name<<"
  // (tuple)size="<<size<<std::endl;
  result->SetNumberOfComponents(1);
  result->SetNumberOfTuples(size);
  result->SetName(name);

  for (i = 0; i < size; i++)
    {
      // result->InsertNextValue(values[i]);
      result->SetTuple1(i, values[i]);
    }

  return result;
}

void get_data_arrays(unsigned long size, vtkDataSetAttributes* dsa)
{
  // dsa->AddArray(generate_array<char>("char_array", size,
  // vtkCharArray::New())); dsa->AddArray(generate_array<double>("double_array",
  // size, vtkDoubleArray::New()));
  // dsa->AddArray(generate_array<int>("int_array", size, vtkIntArray::New()));

  // dsa->AddArray(generate_array<float>("float_array", size,
  // vtkFloatArray::New()));  dsa->AddArray(generate_array<long>("long_array",
  // size, vtkLongArray::New()));

  // dsa->AddArray(generate_array<unsigned char>("unsigned_char_array", size,
  // vtkUnsignedCharArray::New()));  dsa->AddArray(generate_array<unsigned
  // int>("unsigned_int_array", size, vtkUnsignedIntArray::New()));
  // dsa->AddArray(generate_array<unsigned long>("unsigned_long_array", size,
  // vtkUnsignedLongArray::New()));

  vtkDataArray* temp =
    generate_array<char>("char_array", size, vtkCharArray::New());
  dsa->AddArray(temp);
  temp->Delete();

  temp = generate_array<double>("double_array", size, vtkDoubleArray::New());
  dsa->AddArray(temp);
  temp->Delete();

  temp = generate_array<int>("int_array", size, vtkIntArray::New());
  dsa->AddArray(temp);
  temp->Delete();

  temp = generate_array<float>("float_array", size, vtkFloatArray::New());
  dsa->AddArray(temp);
  temp->Delete();

  temp = generate_array<long>("long_array", size, vtkLongArray::New());
  dsa->AddArray(temp);
  temp->Delete();

  temp = generate_array<unsigned char>(
    "unsigned_char_array", size, vtkUnsignedCharArray::New());
  dsa->AddArray(temp);
  temp->Delete();

  temp = generate_array<unsigned int>(
    "unsigned_int_array", size, vtkUnsignedIntArray::New());
  dsa->AddArray(temp);
  temp->Delete();

  temp = generate_array<unsigned long>(
    "unsigned_long_array", size, vtkUnsignedLongArray::New());
  dsa->AddArray(temp);
  temp->Delete();
}

vtkImageData* get_image(unsigned long i0,
                        unsigned long i1,
                        unsigned long j0,
                        unsigned long j1,
                        unsigned long k0,
                        unsigned long k1)
{
  vtkImageData* im = vtkImageData::New();
  im->SetExtent(i0, i1, j0, j1, k0, k1);
  unsigned long nx = i1 - i0 + 1;
  unsigned long ny = j1 - j0 + 1;
  unsigned long nz = k1 - k0 + 1;
  unsigned long npts = (nx + 1) * (ny + 1) * (nz + 1);
  unsigned long ncells = nx * ny * nz;
  get_data_arrays(npts, im->GetPointData());

  get_data_arrays(ncells, im->GetCellData());

  return im;
}

void writeMe(sensei::AnalysisAdaptor* aw, int n_its, MPI_Comm& comm)
{

  int rank, n_ranks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_ranks);

  // the first mesh is an image

  vtkSmartPointer<vtkMultiBlockDataSet> im =
    vtkSmartPointer<vtkMultiBlockDataSet>::New();

  im->SetNumberOfBlocks(n_ranks);
  im->SetBlock(rank, get_image(rank, rank, 0, 16, 0, 1));

  // the second mesh is unstructured
  vtkSmartPointer<vtkMultiBlockDataSet> ug =
    vtkSmartPointer<vtkMultiBlockDataSet>::New();

  ug->SetNumberOfBlocks(n_ranks);
  ug->SetBlock(rank, get_polydata(16));

  //# associate a name with each mesh

  // meshes = {'image':im, 'unstructured':ug}
  std::string meshNames[2] = { "image", "unstructured" };
  vtkDataObject* meshObj[2] = { im, ug };

  // loop over time steps
  for (int i = 0; i < n_its; i++)
    {
      float t = i * 1.0;
      int it = i;

      if (rank == 0)
        std::cout << "initializing the VTKDataAdaptor step:" << it << " t=" << t
                  << std::endl;

      sensei::VTKDataAdaptor* da = sensei::VTKDataAdaptor::New();
      da->SetDataTime(t);
      da->SetDataTimeStep(it);

      da->SetDataObject(meshNames[0], meshObj[0]);
      da->SetDataObject(meshNames[1], meshObj[1]);

      aw->Execute(da);
      da->ReleaseData();
      da->Delete();
      da = NULL;
    }
  aw->Finalize();
  aw->Delete();

  if (rank == 0)
    std::cout << "finished writing " << n_its << " steps \n" << std::endl;
}

AAWrap* GetWriteAdaptor(const std::string& file_name,
                        const std::string& method,
                        int rank)
{
  std::size_t found = file_name.find("h5");
  if (found != std::string::npos)
    {
      bool doStreaming = false;
      bool doCollective = false;

      if ('s' == method[0])
        doStreaming = true;
      if ((method.size() > 1) && ('c' == method[1]))
        doCollective = true;

      if (rank == 0)
        std::cout << " ======>>>> [HDF5] Analysis  Adaptor <<<<<======"
                  << "Streamin? " << doStreaming << " collective? "
                  << doCollective << std::endl;

      // sensei::HDF5AnalysisAdaptor* aw = sensei::HDF5AnalysisAdaptor::New();
      H5AnalysisAdaptorPtr aw = H5AnalysisAdaptorPtr::New();
      aw->SetStreamName(file_name);
      aw->SetStreaming(doStreaming);
      aw->SetCollective(doCollective);

      AAWrap* result = new AAWrap(aw);
      return result;
    }
  else
    {
      if (rank == 0)
        std::cout << " ======>>>> [ADIOS] Analysis  Adaptor <<<<<======"
	          << " with config file:"<<file_name
                  << std::endl;

      // initialize the analysis adaptor
      // sensei::ADIOS1AnalysisAdaptor* aw =
      // sensei::ADIOS1AnalysisAdaptor::New();
      A1AnalysisAdaptorPtr aw = A1AnalysisAdaptorPtr::New();

      aw->SetFileName(file_name);
      aw->SetMethod(method);

      AAWrap* result = new AAWrap(aw);
      return result;
    }
}

TimedAdaptorWrap* GetReadAdaptor(const std::string& file_name,
                                 const std::string& method,
                                 MPI_Comm& comm)
{
  // if  (file_name.find(".h5") != std::string::npos) {
  // std::cout<<"initializing ADIOSDataAdaptor "<<file_name<<std::endl;
  int rank;
  MPI_Comm_rank(comm, &rank);
  std::size_t found = file_name.find("h5");
  if (found != std::string::npos)
    {
      bool doStreaming = false;
      bool doCollective = false;

      if ('s' == method[0])
        doStreaming = true;
      if ((method.size() > 1) && ('c' == method[1]))
        doCollective = true;

      if (rank == 0)
        std::cout << " HDF5  DATA  Adaptor  streaming? " << doStreaming
                  << " collective? " << doCollective << std::endl;

      // sensei::HDF5DataAdaptor* da = sensei::HDF5DataAdaptor::New();
      H5DataAdaptorPtr da = H5DataAdaptorPtr::New();
      da->SetCommunicator(comm);
      da->SetStreaming(doStreaming);
      da->SetCollective(doCollective);
      da->SetStreamName(file_name);
      da->OpenStream();
      TimedAdaptorWrap* result = new TimedAdaptorWrap(da);
      return result;
    }
  else
    {
      if (rank == 0)
        std::cout << " ADIOS DATA   Adaptor " << std::endl;

      // initialize the analysis adaptor
      // sensei::ADIOS1DataAdaptor* da = sensei::ADIOS1DataAdaptor::New();
      A1DataAdaptorPtr da = A1DataAdaptorPtr::New();
      da->SetCommunicator(comm);
      da->SetFileName(file_name);
      da->SetReadMethod(method);
      da->OpenStream();
      TimedAdaptorWrap* result = new TimedAdaptorWrap(da);
      return result;
    }

  return NULL;
}

// this show HDF5 is fine!!

//
// usage: ./ctest n_its method(default MPI)
// e.g. ./ctest 10 POSIX
//
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;
  int n_ranks, rank;
  MPI_Comm_size(comm, &n_ranks);
  MPI_Comm_rank(comm, &rank);

  if (argc == 1)
    {
      std::cout << " please use the following options: " << std::endl;
      std::cout << argv[0] << "  w iter mode file-name " << std::endl;
      std::cout << argv[0] << "  r file-name mode" << std::endl;
      return 0;
    }

  std::string method = "MPI"; // or "POSIX"
  if (argv[1][0] == 'w')
    {
      std::string base_file_name = "hello";

      int n_its = 2;
      if (argc > 2)
        {
          n_its = atoi(argv[2]);
        }

      if (argc > 3)
        {
          method = argv[3];
        }

      if (argc > 4)
        {
          base_file_name = argv[4];
        }

      char file_name[base_file_name.size()];
      sprintf(file_name, "%s.n%d", base_file_name.c_str(), n_ranks);

      if (rank == 0)
        std::cout << " ==> WRITING : " << file_name << std::endl;

      AAWrap* aw = GetWriteAdaptor(file_name, method, rank);
      writeMe(aw->GetAA(), n_its, comm);

    }
  else
    {
      char* file_name = argv[2];

      if (rank == 0)
        std::cout << " ==> READING : " << file_name << std::endl;

      if (argc > 3)
        {
          method = argv[3];
        }

      TimedAdaptorWrap* result = GetReadAdaptor(file_name, method, comm);

      readMe(result, comm);

      delete result;
    }

  MPI_Finalize();
  return 0;
}
