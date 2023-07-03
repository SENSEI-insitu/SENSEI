/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataObjectToConduit.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkDataObjectToConduit.h"

#include "svtkAOSDataArrayTemplate.h"
#include "svtkCellData.h"
#include "svtkCellTypes.h"
#include "svtkDataArray.h"
#include "svtkDataObject.h"
#include "svtkDataSet.h"
#include "svtkFieldData.h"
#include "svtkImageData.h"
#include "svtkLogger.h"
#include "svtkPointData.h"
#include "svtkPointSet.h"
#include "svtkPoints.h"
#include "svtkPolyData.h"
#include "svtkRectilinearGrid.h"
#include "svtkSOADataArrayTemplate.h"
#include "svtkStringArray.h"
#include "svtkStructuredGrid.h"
#include "svtkTypeFloat32Array.h"
#include "svtkTypeFloat64Array.h"
#include "svtkTypeInt16Array.h"
#include "svtkTypeInt32Array.h"
#include "svtkTypeInt64Array.h"
#include "svtkTypeInt8Array.h"
#include "svtkTypeUInt16Array.h"
#include "svtkTypeUInt32Array.h"
#include "svtkTypeUInt64Array.h"
#include "svtkTypeUInt8Array.h"
#include "svtkUnstructuredGrid.h"

#include <catalyst_conduit.hpp>

namespace
{

//----------------------------------------------------------------------------
bool IsMixedShape(svtkDataSet* grid)
{
  // WARNING: This is inefficient
  svtkNew<svtkCellTypes> cell_types;
  grid->GetCellTypes(cell_types);
  return cell_types->GetNumberOfTypes() > 1;
}

//----------------------------------------------------------------------------
svtkCellArray* GetCells(svtkUnstructuredGrid* ugrid, int)
{
  return ugrid->GetCells();
}

//----------------------------------------------------------------------------
svtkCellArray* GetCells(svtkPolyData* polydata, int cellType)
{
  switch (cellType)
  {
    case SVTK_POLYGON:
    case SVTK_QUAD:
    case SVTK_TRIANGLE:
      return polydata->GetPolys();
    case SVTK_LINE:
      return polydata->GetLines();
    case SVTK_VERTEX:
      return polydata->GetVerts();
    default:
      svtkLog(ERROR, << "Unsupported cell type in polydata. Cell type: "
                    << svtkCellTypes::GetClassNameFromTypeId(cellType));
      return nullptr;
  }
}
//----------------------------------------------------------------------------
bool IsSignedIntegralType(int data_type)
{
  constexpr bool is_char_type_signed = (CHAR_MIN == SCHAR_MIN) && (CHAR_MAX == SCHAR_MAX);

  return (is_char_type_signed && (data_type == SVTK_CHAR)) || (data_type == SVTK_SIGNED_CHAR) ||
    (data_type == SVTK_SHORT) || (data_type == SVTK_INT) || (data_type == SVTK_LONG) ||
    (data_type == SVTK_ID_TYPE) || (data_type == SVTK_LONG_LONG) || (data_type == SVTK_TYPE_INT64);
}

//----------------------------------------------------------------------------
bool IsUnsignedIntegralType(int data_type)
{
  constexpr bool is_char_type_signed = (CHAR_MIN == SCHAR_MIN) && (CHAR_MAX == SCHAR_MAX);

  return (!is_char_type_signed && (data_type == SVTK_CHAR)) || (data_type == SVTK_UNSIGNED_CHAR) ||
    (data_type == SVTK_UNSIGNED_SHORT) || (data_type == SVTK_UNSIGNED_INT) ||
    (data_type == SVTK_UNSIGNED_LONG) || (data_type == SVTK_ID_TYPE) ||
    (data_type == SVTK_UNSIGNED_LONG_LONG);
}

//----------------------------------------------------------------------------
bool IsFloatType(int data_type)
{
  return ((data_type == SVTK_FLOAT) || (data_type == SVTK_DOUBLE));
}

//----------------------------------------------------------------------------
bool ConvertDataArrayToMCArray(
  svtkDataArray* data_array, int offset, int stride, conduit_cpp::Node& conduit_node)
{
  stride = std::max(stride, 1);
  conduit_index_t number_of_elements = data_array->GetNumberOfValues() / stride;

  int data_type = data_array->GetDataType();
  int data_type_size = data_array->GetDataTypeSize();
  int array_type = data_array->GetArrayType();

  if (array_type != svtkAbstractArray::AoSDataArrayTemplate)
  {
    svtkLog(ERROR, "Unsupported data array type: " << data_array->GetDataTypeAsString());
    return false;
  }

  // The code below uses the legacy GetVoidPointer on purpose to get zero copy.
  bool is_supported = true;
  if (IsSignedIntegralType(data_type))
  {
    switch (data_type_size)
    {
      case 1:
        conduit_node.set_external_int8_ptr((conduit_int8*)data_array->GetVoidPointer(0),
          number_of_elements, offset * sizeof(conduit_int8), stride * sizeof(conduit_int8));
        break;

      case 2:
        conduit_node.set_external_int16_ptr((conduit_int16*)data_array->GetVoidPointer(0),
          number_of_elements, offset * sizeof(conduit_int16), stride * sizeof(conduit_int16));
        break;

      case 4:
        conduit_node.set_external_int32_ptr((conduit_int32*)data_array->GetVoidPointer(0),
          number_of_elements, offset * sizeof(conduit_int32), stride * sizeof(conduit_int32));
        break;

      case 8:
        conduit_node.set_external_int64_ptr((conduit_int64*)data_array->GetVoidPointer(0),
          number_of_elements, offset * sizeof(conduit_int64), stride * sizeof(conduit_int64));
        break;

      default:
        is_supported = false;
    }
  }
  else if (IsUnsignedIntegralType(data_type))
  {
    switch (data_type_size)
    {
      case 1:
        conduit_node.set_external_uint8_ptr((conduit_uint8*)data_array->GetVoidPointer(0),
          number_of_elements, offset * sizeof(conduit_uint8), stride * sizeof(conduit_uint8));
        break;

      case 2:
        conduit_node.set_external_uint16_ptr((conduit_uint16*)data_array->GetVoidPointer(0),
          number_of_elements, offset * sizeof(conduit_uint16), stride * sizeof(conduit_uint16));
        break;

      case 4:
        conduit_node.set_external_uint32_ptr((conduit_uint32*)data_array->GetVoidPointer(0),
          number_of_elements, offset * sizeof(conduit_uint32), stride * sizeof(conduit_uint32));
        break;

      case 8:
        conduit_node.set_external_uint64_ptr((conduit_uint64*)data_array->GetVoidPointer(0),
          number_of_elements, offset * sizeof(conduit_uint64), stride * sizeof(conduit_uint64));
        break;

      default:
        is_supported = false;
    }
  }
  else if (IsFloatType(data_type))
  {
    switch (data_type_size)
    {
      case 4:
        conduit_node.set_external_float32_ptr((conduit_float32*)data_array->GetVoidPointer(0),
          number_of_elements, offset * sizeof(conduit_float32), stride * sizeof(conduit_float32));
        break;

      case 8:
        conduit_node.set_external_float64_ptr((conduit_float64*)data_array->GetVoidPointer(0),
          number_of_elements, offset * sizeof(conduit_float64), stride * sizeof(conduit_float64));
        break;

      default:
        is_supported = false;
    }
  }

  if (!is_supported)
  {
    svtkLog(ERROR,
      "Unsupported data array type: " << data_array->GetDataTypeAsString()
                                      << " size: " << data_type_size << " type: " << array_type);
  }

  return is_supported;
}

//----------------------------------------------------------------------------
bool ConvertDataArrayToMCArray(svtkDataArray* data_array, conduit_cpp::Node& conduit_node,
  const std::vector<std::string> names = std::vector<std::string>())
{
  size_t nComponents = data_array->GetNumberOfComponents();
  if (nComponents > 1)
  {
    bool success = true;
    for (size_t i = 0; i < nComponents; ++i)
    {
      conduit_cpp::Node component_node;
      if (i < names.size())
      {
        component_node = conduit_node[names[i]];
      }
      else
      {
        component_node = conduit_node[std::to_string(i)];
      }
      success = success && ConvertDataArrayToMCArray(data_array, i, nComponents, component_node);
    }
    return success;
  }
  else
  {
    return ConvertDataArrayToMCArray(data_array, 0, 0, conduit_node);
  }
}

//----------------------------------------------------------------------------
template <class T>
bool FillTopology(T* dataset, conduit_cpp::Node& conduit_node)
{
  const char* datasetType = dataset->GetClassName();
  if (IsMixedShape(dataset))
  {
    svtkLogF(ERROR, "%s with mixed shape type unsupported.", datasetType);
    return false;
  }

  auto coords_node = conduit_node["coordsets/coords"];

  coords_node["type"] = "explicit";

  auto values_node = coords_node["values"];
  auto* points = dataset->GetPoints();

  if (points)
  {
    if (!ConvertDataArrayToMCArray(points->GetData(), values_node, { "x", "y", "z" }))
    {
      svtkLogF(ERROR, "ConvertPoints failed for %s.", datasetType);
      return false;
    }
  }
  else
  {
    values_node["x"] = std::vector<float>();
    values_node["y"] = std::vector<float>();
    values_node["z"] = std::vector<float>();
  }

  auto topologies_node = conduit_node["topologies/mesh"];
  topologies_node["type"] = "unstructured";
  topologies_node["coordset"] = "coords";

  int cell_type = SVTK_VERTEX;
  const auto number_of_cells = dataset->GetNumberOfCells();
  if (number_of_cells > 0)
  {
    cell_type = dataset->GetCellType(0);
  }

  switch (cell_type)
  {
    case SVTK_HEXAHEDRON:
      topologies_node["elements/shape"] = "hex";
      break;
    case SVTK_TETRA:
      topologies_node["elements/shape"] = "tet";
      break;
    case SVTK_POLYGON:
      topologies_node["elements/shape"] = "polygonal";
      break;
    case SVTK_QUAD:
      topologies_node["elements/shape"] = "quad";
      break;
    case SVTK_TRIANGLE:
      topologies_node["elements/shape"] = "tri";
      break;
    case SVTK_LINE:
      topologies_node["elements/shape"] = "line";
      break;
    case SVTK_VERTEX:
      topologies_node["elements/shape"] = "point";
      break;
    default:
      svtkLogF(ERROR, "Unsupported cell type in %s. Cell type: %s", datasetType,
        svtkCellTypes::GetClassNameFromTypeId(cell_type));
      return false;
  }

  auto cell_connectivity = GetCells(dataset, cell_type);
  auto connectivity_node = topologies_node["elements/connectivity"];

  if (!ConvertDataArrayToMCArray(cell_connectivity->GetConnectivityArray(), connectivity_node))
  {
    svtkLogF(ERROR, "ConvertDataArrayToMCArray failed for %s.", datasetType);
    return false;
  }
  return true;
}

//----------------------------------------------------------------------------
bool FillTopology(svtkDataSet* data_set, conduit_cpp::Node& conduit_node)
{
  if (auto imageData = svtkImageData::SafeDownCast(data_set))
  {
    auto coords_node = conduit_node["coordsets/coords"];

    coords_node["type"] = "uniform";

    int* dimensions = imageData->GetDimensions();
    coords_node["dims/i"] = dimensions[0];
    coords_node["dims/j"] = dimensions[1];
    coords_node["dims/k"] = dimensions[2];

    double* origin = imageData->GetOrigin();
    coords_node["origin/x"] = origin[0];
    coords_node["origin/y"] = origin[1];
    coords_node["origin/z"] = origin[2];

    double* spacing = imageData->GetSpacing();
    coords_node["spacing/dx"] = spacing[0];
    coords_node["spacing/dy"] = spacing[1];
    coords_node["spacing/dz"] = spacing[2];

    auto topologies_node = conduit_node["topologies/mesh"];
    topologies_node["type"] = "uniform";
    topologies_node["coordset"] = "coords";
  }
  else if (auto rectilinear_grid = svtkRectilinearGrid::SafeDownCast(data_set))
  {
    auto coords_node = conduit_node["coordsets/coords"];

    coords_node["type"] = "rectilinear";

    auto x_values_node = coords_node["values/x"];
    if (!ConvertDataArrayToMCArray(rectilinear_grid->GetXCoordinates(), x_values_node))
    {
      svtkLog(ERROR, "Failed ConvertDataArrayToMCArray for values/x");
      return false;
    }

    auto y_values_node = coords_node["values/y"];
    if (!ConvertDataArrayToMCArray(rectilinear_grid->GetYCoordinates(), y_values_node))
    {
      svtkLog(ERROR, "Failed ConvertDataArrayToMCArray for values/y");
      return false;
    }

    auto z_values_node = coords_node["values/z"];
    if (!ConvertDataArrayToMCArray(rectilinear_grid->GetZCoordinates(), z_values_node))
    {
      svtkLog(ERROR, "Failed ConvertDataArrayToMCArray for values/z");
      return false;
    }

    auto topologies_node = conduit_node["topologies/mesh"];
    topologies_node["type"] = "rectilinear";
    topologies_node["coordset"] = "coords";
  }
  else if (auto structured_grid = svtkStructuredGrid::SafeDownCast(data_set))
  {
    auto coords_node = conduit_node["coordsets/coords"];

    coords_node["type"] = "explicit";

    auto values_node = coords_node["values"];
    if (!ConvertDataArrayToMCArray(
          structured_grid->GetPoints()->GetData(), values_node, { "x", "y", "z" }))
    {
      svtkLog(ERROR, "Failed ConvertPoints for structured grid");
      return false;
    }

    auto topologies_node = conduit_node["topologies/mesh"];
    topologies_node["type"] = "structured";
    topologies_node["coordset"] = "coords";
    int dimensions[3];
    structured_grid->GetDimensions(dimensions);
    topologies_node["elements/dims/i"] = dimensions[0];
    topologies_node["elements/dims/j"] = dimensions[1];
    topologies_node["elements/dims/k"] = dimensions[2];
  }
  else if (auto unstructured_grid = svtkUnstructuredGrid::SafeDownCast(data_set))
  {
    return FillTopology(unstructured_grid, conduit_node);
  }
  else if (auto polydata = svtkPolyData::SafeDownCast(data_set))
  {
    return FillTopology(polydata, conduit_node);
  }
  else
  {
    svtkLog(ERROR, "Unsupported data set type: " << data_set->GetClassName());
    return false;
  }

  return true;
}

//----------------------------------------------------------------------------
bool FillFields(
  svtkFieldData* field_data, const std::string& association, conduit_cpp::Node& conduit_node)
{
  bool is_success = true;

  int array_count = field_data->GetNumberOfArrays();
  for (int array_index = 0; is_success && array_index < array_count; ++array_index)
  {
    auto array = field_data->GetAbstractArray(array_index);
    auto name = array->GetName();
    if (!name)
    {
      svtkLogF(WARNING, "Unnamed array, it will be ignored.");
      continue;
    }

    if (association.empty())
    {
      // SVTK Field Data are translated to state/fields childs.
      auto field_node = conduit_node["state/fields"][name];

      if (auto string_array = svtkStringArray::SafeDownCast(array))
      {
        if (string_array->GetNumberOfValues() > 0)
        {
          field_node.set_string(string_array->GetValue(0));
          if (string_array->GetNumberOfValues() > 1)
          {
            svtkLog(WARNING,
              "The string array '" << string_array->GetName()
                                   << "' contains more than one element. Only the first one will "
                                      "be converted to conduit node.");
          }
        }
      }
      else if (auto data_array = svtkDataArray::SafeDownCast(array))
      {
        is_success = ConvertDataArrayToMCArray(data_array, field_node);
      }
      else
      {
        svtkLogF(ERROR, "Unknown array type '%s' in Field Data.", name);
        is_success = false;
      }
    }
    else if (auto data_array = svtkDataArray::SafeDownCast(array))
    {
      auto field_node = conduit_node["fields"][name];
      field_node["association"] = association;
      field_node["topology"] = "mesh";
      field_node["volume_dependent"] = "false";

      auto values_node = field_node["values"];
      is_success = ConvertDataArrayToMCArray(data_array, values_node);
    }
    else
    {
      svtkLogF(ERROR, "Unknown array type '%s' associated to: %s", name, association.c_str());
      is_success = false;
    }
  }

  return is_success;
}

//----------------------------------------------------------------------------
bool FillFields(svtkDataSet* data_set, conduit_cpp::Node& conduit_node)
{
  if (auto cell_data = data_set->GetCellData())
  {
    if (!FillFields(cell_data, "element", conduit_node))
    {
      svtkVLog(svtkLogger::VERBOSITY_ERROR, "FillFields with element failed.");
      return false;
    }
  }

  if (auto point_data = data_set->GetPointData())
  {
    if (!FillFields(point_data, "vertex", conduit_node))
    {
      svtkVLog(svtkLogger::VERBOSITY_ERROR, "FillFields with vertex failed.");
      return false;
    }
  }

  if (auto field_data = data_set->GetFieldData())
  {
    if (!FillFields(field_data, "", conduit_node))
    {
      svtkVLog(svtkLogger::VERBOSITY_ERROR, "FillFields with field data failed.");
      return false;
    }
  }

  return true;
}

//----------------------------------------------------------------------------
bool FillConduitNodeFromDataSet(svtkDataSet* data_set, conduit_cpp::Node& conduit_node)
{
  return FillTopology(data_set, conduit_node) && FillFields(data_set, conduit_node);
}

} // anonymous namespace

namespace svtkDataObjectToConduit
{

//----------------------------------------------------------------------------
bool FillConduitNode(svtkDataObject* data_object, conduit_cpp::Node& conduit_node)
{
  auto data_set = svtkDataSet::SafeDownCast(data_object);
  if (!data_set)
  {
    svtkLogF(ERROR, "Only Data Set objects are supported in svtkDataObjectToConduit.");
    return false;
  }

  return FillConduitNodeFromDataSet(data_set, conduit_node);
}

} // svtkDataObjectToConduit namespace
