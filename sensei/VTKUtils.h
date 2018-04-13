#ifndef VTKUtils_h
#define VTKUtils_h

class vtkDataSet;
class vtkDataObject;
class vtkFieldData;

#include <functional>

namespace sensei
{

/// A collection of generally useful funcitons implementing
/// common access patterns or operations on VTK data structures
namespace VTKUtils
{

/// returns the name of the association, point, cell or field
const char *GetAttributesName(int association);

/// returns the container for the associations: vtkPointData,
/// vtkCellData, or vtkFieldData
vtkFieldData *GetAttributes(vtkDataSet *dobj, int association);

/// callback that processes input and output datasets
/// return 0 for success, > zero to stop without error, < zero to stop with error
using BinaryDatasetFunction = std::function<int(vtkDataSet*, vtkDataSet*)>;

/// Applies the function to leaves of the structurally equivalent
/// input and output data objects.
int Apply(vtkDataObject *input, vtkDataObject *output,
  BinaryDatasetFunction &func);

/// callback that processes input and output datasets
/// return 0 for success, > zero to stop without error, < zero to stop with error
using DatasetFunction = std::function<int(vtkDataSet*)>;

/// Applies the function to the data object
/// The function is called once for each leaf dataset
int Apply(vtkDataObject *dobj, DatasetFunction &func);

}
}

#endif
