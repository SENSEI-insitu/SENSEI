#include "libISSchema.h"
#include "MeshMetadataMap.h"
#include "BinaryStream.h"
#include "Partitioner.h"
#include "VTKUtils.h"
#include "MPIUtils.h"
#include "Error.h"

#include <vtkCellTypes.h>
#include <vtkCellData.h>
#include <vtkCompositeDataIterator.h>
#include <vtkCompositeDataSet.h>
#include <vtkDataSetAttributes.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkUnsignedIntArray.h>
#include <vtkLongArray.h>
#include <vtkUnsignedLongArray.h>
#include <vtkLongLongArray.h>
#include <vtkUnsignedLongLongArray.h>
#include <vtkCharArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkIdTypeArray.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkStructuredPoints.h>
#include <vtkStructuredGrid.h>
#include <vtkRectilinearGrid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkImageData.h>
#include <vtkUniformGrid.h>
#include <vtkTable.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkHierarchicalBoxDataSet.h>
#include <vtkMultiPieceDataSet.h>
#include <vtkHyperTreeGrid.h>
#include <vtkOverlappingAMR.h>
#include <vtkNonOverlappingAMR.h>
#include <vtkUniformGridAMR.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>

#include <mpi.h>

//#include <adios.h>
//#include <adios_read.h>

#include <vector>
#include <map>
#include <set>
#include <string>
#include <functional>
#include <sstream>

namespace senseilibIS
{


struct DataObjectCollectionSchema::InternalsType
{
  sensei::MeshMetadataMap SenderMdMap;
  sensei::MeshMetadataMap ReceiverMdMap;
};

// --------------------------------------------------------------------------
DataObjectCollectionSchema::DataObjectCollectionSchema()
{
  this->Internals = new DataObjectCollectionSchema::InternalsType;
}

// --------------------------------------------------------------------------
DataObjectCollectionSchema::~DataObjectCollectionSchema()
{
  delete this->Internals;
}


// --------------------------------------------------------------------------
int DataObjectCollectionSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  const std::vector<sensei::MeshMetadataPtr> &metadata)
{
  return 0;
}


}
