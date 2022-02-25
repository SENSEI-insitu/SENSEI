/*=========================================================================

Program:   Visualization Toolkit
Module:    svtkDataObject.cxx

Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkDataObject.h"

#include "svtkDataSetAttributes.h"
#include "svtkFieldData.h"
#include "svtkGarbageCollector.h"
#include "svtkInformation.h"
#include "svtkInformationDataObjectKey.h"
#include "svtkInformationDoubleKey.h"
#include "svtkInformationDoubleVectorKey.h"
#include "svtkInformationInformationVectorKey.h"
#include "svtkInformationIntegerKey.h"
#include "svtkInformationIntegerPointerKey.h"
#include "svtkInformationIntegerVectorKey.h"
#include "svtkInformationStringKey.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkDataObject);

svtkCxxSetObjectMacro(svtkDataObject, Information, svtkInformation);
svtkCxxSetObjectMacro(svtkDataObject, FieldData, svtkFieldData);

svtkInformationKeyMacro(svtkDataObject, DATA_TYPE_NAME, String);
svtkInformationKeyMacro(svtkDataObject, DATA_OBJECT, DataObject);
svtkInformationKeyMacro(svtkDataObject, DATA_EXTENT_TYPE, Integer);
svtkInformationKeyMacro(svtkDataObject, DATA_PIECE_NUMBER, Integer);
svtkInformationKeyMacro(svtkDataObject, DATA_NUMBER_OF_PIECES, Integer);
svtkInformationKeyMacro(svtkDataObject, DATA_NUMBER_OF_GHOST_LEVELS, Integer);
svtkInformationKeyMacro(svtkDataObject, DATA_TIME_STEP, Double);
svtkInformationKeyMacro(svtkDataObject, POINT_DATA_VECTOR, InformationVector);
svtkInformationKeyMacro(svtkDataObject, CELL_DATA_VECTOR, InformationVector);
svtkInformationKeyMacro(svtkDataObject, VERTEX_DATA_VECTOR, InformationVector);
svtkInformationKeyMacro(svtkDataObject, EDGE_DATA_VECTOR, InformationVector);
svtkInformationKeyMacro(svtkDataObject, FIELD_ARRAY_TYPE, Integer);
svtkInformationKeyMacro(svtkDataObject, FIELD_ASSOCIATION, Integer);
svtkInformationKeyMacro(svtkDataObject, FIELD_ATTRIBUTE_TYPE, Integer);
svtkInformationKeyMacro(svtkDataObject, FIELD_ACTIVE_ATTRIBUTE, Integer);
svtkInformationKeyMacro(svtkDataObject, FIELD_NAME, String);
svtkInformationKeyMacro(svtkDataObject, FIELD_NUMBER_OF_COMPONENTS, Integer);
svtkInformationKeyMacro(svtkDataObject, FIELD_NUMBER_OF_TUPLES, Integer);
svtkInformationKeyRestrictedMacro(svtkDataObject, FIELD_RANGE, DoubleVector, 2);
svtkInformationKeyRestrictedMacro(svtkDataObject, PIECE_EXTENT, IntegerVector, 6);
svtkInformationKeyMacro(svtkDataObject, FIELD_OPERATION, Integer);
svtkInformationKeyRestrictedMacro(svtkDataObject, ALL_PIECES_EXTENT, IntegerVector, 6);
svtkInformationKeyRestrictedMacro(svtkDataObject, DATA_EXTENT, IntegerPointer, 6);
svtkInformationKeyRestrictedMacro(svtkDataObject, ORIGIN, DoubleVector, 3);
svtkInformationKeyRestrictedMacro(svtkDataObject, SPACING, DoubleVector, 3);
svtkInformationKeyRestrictedMacro(svtkDataObject, DIRECTION, DoubleVector, 9);
svtkInformationKeyMacro(svtkDataObject, SIL, DataObject);
svtkInformationKeyRestrictedMacro(svtkDataObject, BOUNDING_BOX, DoubleVector, 6);

// Initialize static member that controls global data release
// after use by filter
static int svtkDataObjectGlobalReleaseDataFlag = 0;

// this list must be kept in-sync with the FieldAssociations enum
static const char* FieldAssociationsNames[] = { "svtkDataObject::FIELD_ASSOCIATION_POINTS",
  "svtkDataObject::FIELD_ASSOCIATION_CELLS", "svtkDataObject::FIELD_ASSOCIATION_NONE",
  "svtkDataObject::FIELD_ASSOCIATION_POINTS_THEN_CELLS", "svtkDataObject::FIELD_ASSOCIATION_VERTICES",
  "svtkDataObject::FIELD_ASSOCIATION_EDGES", "svtkDataObject::FIELD_ASSOCIATION_ROWS" };

// this list must be kept in-sync with the AttributeTypes enum
static const char* AttributeTypesNames[] = { "svtkDataObject::POINT", "svtkDataObject::CELL",
  "svtkDataObject::FIELD", "svtkDataObject::POINT_THEN_CELL", "svtkDataObject::VERTEX",
  "svtkDataObject::EDGE", "svtkDataObject::ROW" };

//----------------------------------------------------------------------------
svtkDataObject::svtkDataObject()
{
  this->Information = svtkInformation::New();

  // We have to assume that if a user is creating the data on their own,
  // then they will fill it with valid data.
  this->DataReleased = 0;

  this->FieldData = nullptr;
  svtkFieldData* fd = svtkFieldData::New();
  this->SetFieldData(fd);
  fd->FastDelete();
}

//----------------------------------------------------------------------------
svtkDataObject::~svtkDataObject()
{
  this->SetInformation(nullptr);
  this->SetFieldData(nullptr);
}

//----------------------------------------------------------------------------
void svtkDataObject::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  if (this->Information)
  {
    os << indent << "Information: " << this->Information << "\n";
  }
  else
  {
    os << indent << "Information: (none)\n";
  }

  os << indent << "Data Released: " << (this->DataReleased ? "True\n" : "False\n");
  os << indent
     << "Global Release Data: " << (svtkDataObjectGlobalReleaseDataFlag ? "On\n" : "Off\n");

  os << indent << "UpdateTime: " << this->UpdateTime << endl;

  os << indent << "Field Data:\n";
  this->FieldData->PrintSelf(os, indent.GetNextIndent());
}

//----------------------------------------------------------------------------
// Determine the modified time of this object
svtkMTimeType svtkDataObject::GetMTime()
{
  svtkMTimeType result;

  result = svtkObject::GetMTime();
  if (this->FieldData)
  {
    svtkMTimeType mtime = this->FieldData->GetMTime();
    result = (mtime > result ? mtime : result);
  }

  return result;
}

//----------------------------------------------------------------------------
void svtkDataObject::Initialize()
{
  if (this->FieldData)
  {
    this->FieldData->Initialize();
  }

  if (this->Information)
  {
    // Make sure the information is cleared.
    this->Information->Remove(ALL_PIECES_EXTENT());
    this->Information->Remove(DATA_PIECE_NUMBER());
    this->Information->Remove(DATA_NUMBER_OF_PIECES());
    this->Information->Remove(DATA_NUMBER_OF_GHOST_LEVELS());
    this->Information->Remove(DATA_TIME_STEP());
  }

  this->Modified();
}

//----------------------------------------------------------------------------
void svtkDataObject::SetGlobalReleaseDataFlag(int val)
{
  if (val == svtkDataObjectGlobalReleaseDataFlag)
  {
    return;
  }
  svtkDataObjectGlobalReleaseDataFlag = val;
}

//----------------------------------------------------------------------------
svtkInformation* svtkDataObject::GetActiveFieldInformation(
  svtkInformation* info, int fieldAssociation, int attributeType)
{
  int i;
  svtkInformation* fieldDataInfo;
  svtkInformationVector* fieldDataInfoVector;

  if (fieldAssociation == FIELD_ASSOCIATION_POINTS)
  {
    fieldDataInfoVector = info->Get(POINT_DATA_VECTOR());
  }
  else if (fieldAssociation == FIELD_ASSOCIATION_CELLS)
  {
    fieldDataInfoVector = info->Get(CELL_DATA_VECTOR());
  }
  else if (fieldAssociation == FIELD_ASSOCIATION_VERTICES)
  {
    fieldDataInfoVector = info->Get(VERTEX_DATA_VECTOR());
  }
  else if (fieldAssociation == FIELD_ASSOCIATION_EDGES)
  {
    fieldDataInfoVector = info->Get(EDGE_DATA_VECTOR());
  }
  else
  {
    svtkGenericWarningMacro("Unrecognized field association!");
    return nullptr;
  }

  if (!fieldDataInfoVector)
  {
    return nullptr;
  }

  for (i = 0; i < fieldDataInfoVector->GetNumberOfInformationObjects(); i++)
  {
    fieldDataInfo = fieldDataInfoVector->GetInformationObject(i);
    if (fieldDataInfo->Has(FIELD_ACTIVE_ATTRIBUTE()) &&
      (fieldDataInfo->Get(FIELD_ACTIVE_ATTRIBUTE()) & (1 << attributeType)))
    {
      return fieldDataInfo;
    }
  }
  return nullptr;
}

//----------------------------------------------------------------------------
svtkInformation* svtkDataObject::GetNamedFieldInformation(
  svtkInformation* info, int fieldAssociation, const char* name)
{
  int i;
  svtkInformation* fieldDataInfo;
  svtkInformationVector* fieldDataInfoVector;

  if (fieldAssociation == FIELD_ASSOCIATION_POINTS)
  {
    fieldDataInfoVector = info->Get(POINT_DATA_VECTOR());
  }
  else if (fieldAssociation == FIELD_ASSOCIATION_CELLS)
  {
    fieldDataInfoVector = info->Get(CELL_DATA_VECTOR());
  }
  else if (fieldAssociation == FIELD_ASSOCIATION_VERTICES)
  {
    fieldDataInfoVector = info->Get(VERTEX_DATA_VECTOR());
  }
  else if (fieldAssociation == FIELD_ASSOCIATION_EDGES)
  {
    fieldDataInfoVector = info->Get(EDGE_DATA_VECTOR());
  }
  else
  {
    svtkGenericWarningMacro("Unrecognized field association!");
    return nullptr;
  }

  if (!fieldDataInfoVector)
  {
    return nullptr;
  }

  for (i = 0; i < fieldDataInfoVector->GetNumberOfInformationObjects(); i++)
  {
    fieldDataInfo = fieldDataInfoVector->GetInformationObject(i);
    if (fieldDataInfo->Has(FIELD_NAME()) && !strcmp(fieldDataInfo->Get(FIELD_NAME()), name))
    {
      return fieldDataInfo;
    }
  }
  return nullptr;
}

//----------------------------------------------------------------------------
void svtkDataObject::RemoveNamedFieldInformation(
  svtkInformation* info, int fieldAssociation, const char* name)
{
  int i;
  svtkInformation* fieldDataInfo;
  svtkInformationVector* fieldDataInfoVector;

  if (fieldAssociation == FIELD_ASSOCIATION_POINTS)
  {
    fieldDataInfoVector = info->Get(POINT_DATA_VECTOR());
  }
  else if (fieldAssociation == FIELD_ASSOCIATION_CELLS)
  {
    fieldDataInfoVector = info->Get(CELL_DATA_VECTOR());
  }
  else if (fieldAssociation == FIELD_ASSOCIATION_VERTICES)
  {
    fieldDataInfoVector = info->Get(VERTEX_DATA_VECTOR());
  }
  else if (fieldAssociation == FIELD_ASSOCIATION_EDGES)
  {
    fieldDataInfoVector = info->Get(EDGE_DATA_VECTOR());
  }
  else
  {
    svtkGenericWarningMacro("Unrecognized field association!");
    return;
  }

  if (!fieldDataInfoVector)
  {
    return;
  }

  for (i = 0; i < fieldDataInfoVector->GetNumberOfInformationObjects(); i++)
  {
    fieldDataInfo = fieldDataInfoVector->GetInformationObject(i);
    if (fieldDataInfo->Has(FIELD_NAME()) && !strcmp(fieldDataInfo->Get(FIELD_NAME()), name))
    {
      fieldDataInfoVector->Remove(fieldDataInfo);
      return;
    }
  }
}

//----------------------------------------------------------------------------
svtkInformation* svtkDataObject::SetActiveAttribute(
  svtkInformation* info, int fieldAssociation, const char* attributeName, int attributeType)
{
  int i;
  svtkInformation* fieldDataInfo;
  svtkInformationVector* fieldDataInfoVector;

  if (fieldAssociation == FIELD_ASSOCIATION_POINTS)
  {
    fieldDataInfoVector = info->Get(POINT_DATA_VECTOR());
  }
  else if (fieldAssociation == FIELD_ASSOCIATION_CELLS)
  {
    fieldDataInfoVector = info->Get(CELL_DATA_VECTOR());
  }
  else if (fieldAssociation == FIELD_ASSOCIATION_VERTICES)
  {
    fieldDataInfoVector = info->Get(VERTEX_DATA_VECTOR());
  }
  else if (fieldAssociation == FIELD_ASSOCIATION_EDGES)
  {
    fieldDataInfoVector = info->Get(EDGE_DATA_VECTOR());
  }
  else
  {
    svtkGenericWarningMacro("Unrecognized field association!");
    return nullptr;
  }
  if (!fieldDataInfoVector)
  {
    fieldDataInfoVector = svtkInformationVector::New();
    if (fieldAssociation == FIELD_ASSOCIATION_POINTS)
    {
      info->Set(POINT_DATA_VECTOR(), fieldDataInfoVector);
    }
    else if (fieldAssociation == FIELD_ASSOCIATION_CELLS)
    {
      info->Set(CELL_DATA_VECTOR(), fieldDataInfoVector);
    }
    else if (fieldAssociation == FIELD_ASSOCIATION_VERTICES)
    {
      info->Set(VERTEX_DATA_VECTOR(), fieldDataInfoVector);
    }
    else // if (fieldAssociation == FIELD_ASSOCIATION_EDGES)
    {
      info->Set(EDGE_DATA_VECTOR(), fieldDataInfoVector);
    }
    fieldDataInfoVector->FastDelete();
  }

  // if we find a matching field, turn it on (active);  if another field of same
  // attribute type was active, turn it off (not active)
  svtkInformation* activeField = nullptr;
  int activeAttribute;
  const char* fieldName;
  for (i = 0; i < fieldDataInfoVector->GetNumberOfInformationObjects(); i++)
  {
    fieldDataInfo = fieldDataInfoVector->GetInformationObject(i);
    activeAttribute = fieldDataInfo->Get(FIELD_ACTIVE_ATTRIBUTE());
    fieldName = fieldDataInfo->Get(FIELD_NAME());
    // if names match (or both empty... no field name), then set active
    if ((attributeName && fieldName && !strcmp(attributeName, fieldName)) ||
      (!attributeName && !fieldName))
    {
      activeAttribute |= 1 << attributeType;
      fieldDataInfo->Set(FIELD_ACTIVE_ATTRIBUTE(), activeAttribute);
      activeField = fieldDataInfo;
    }
    else if (activeAttribute & (1 << attributeType))
    {
      activeAttribute &= ~(1 << attributeType);
      fieldDataInfo->Set(FIELD_ACTIVE_ATTRIBUTE(), activeAttribute);
    }
  }

  // if we didn't find a matching field, create one
  if (!activeField)
  {
    activeField = svtkInformation::New();
    activeField->Set(FIELD_ACTIVE_ATTRIBUTE(), 1 << attributeType);
    activeField->Set(FIELD_ASSOCIATION(), fieldAssociation);
    if (attributeName)
    {
      activeField->Set(FIELD_NAME(), attributeName);
    }
    fieldDataInfoVector->Append(activeField);
    activeField->FastDelete();
  }

  return activeField;
}

//----------------------------------------------------------------------------
void svtkDataObject::SetActiveAttributeInfo(svtkInformation* info, int fieldAssociation,
  int attributeType, const char* name, int arrayType, int numComponents, int numTuples)
{
  svtkInformation* attrInfo =
    svtkDataObject::GetActiveFieldInformation(info, fieldAssociation, attributeType);
  if (!attrInfo)
  {
    // create an entry and set it as active
    attrInfo = SetActiveAttribute(info, fieldAssociation, name, attributeType);
  }

  if (name)
  {
    attrInfo->Set(FIELD_NAME(), name);
  }

  // Set the scalar type if it was given.  If it was not given and
  // there is no current scalar type set the default to SVTK_DOUBLE.
  if (arrayType != -1)
  {
    attrInfo->Set(FIELD_ARRAY_TYPE(), arrayType);
  }
  else if (!attrInfo->Has(FIELD_ARRAY_TYPE()))
  {
    attrInfo->Set(FIELD_ARRAY_TYPE(), SVTK_DOUBLE);
  }

  // Set the number of components if it was given.  If it was not
  // given and there is no current number of components set the
  // default to 1.
  if (numComponents != -1)
  {
    attrInfo->Set(FIELD_NUMBER_OF_COMPONENTS(), numComponents);
  }
  else if (!attrInfo->Has(FIELD_NUMBER_OF_COMPONENTS()))
  {
    attrInfo->Set(FIELD_NUMBER_OF_COMPONENTS(), 1);
  }

  if (numTuples != -1)
  {
    attrInfo->Set(FIELD_NUMBER_OF_TUPLES(), numTuples);
  }
}

//----------------------------------------------------------------------------
void svtkDataObject::SetPointDataActiveScalarInfo(
  svtkInformation* info, int arrayType, int numComponents)
{
  svtkDataObject::SetActiveAttributeInfo(info, FIELD_ASSOCIATION_POINTS,
    svtkDataSetAttributes::SCALARS, nullptr, arrayType, numComponents, -1);
}

//----------------------------------------------------------------------------
void svtkDataObject::DataHasBeenGenerated()
{
  this->DataReleased = 0;
  this->UpdateTime.Modified();
}

//----------------------------------------------------------------------------
int svtkDataObject::GetGlobalReleaseDataFlag()
{
  return svtkDataObjectGlobalReleaseDataFlag;
}

//----------------------------------------------------------------------------
void svtkDataObject::ReleaseData()
{
  this->Initialize();
  this->DataReleased = 1;
}

//----------------------------------------------------------------------------
svtkMTimeType svtkDataObject::GetUpdateTime()
{
  return this->UpdateTime.GetMTime();
}

//----------------------------------------------------------------------------

unsigned long svtkDataObject::GetActualMemorySize()
{
  return this->FieldData->GetActualMemorySize();
}

//----------------------------------------------------------------------------
void svtkDataObject::ShallowCopy(svtkDataObject* src)
{
  if (!src)
  {
    svtkWarningMacro("Attempted to ShallowCopy from null.");
    return;
  }

  this->InternalDataObjectCopy(src);

  if (!src->FieldData)
  {
    this->SetFieldData(nullptr);
  }
  else
  {
    if (this->FieldData)
    {
      this->FieldData->ShallowCopy(src->FieldData);
    }
    else
    {
      svtkFieldData* fd = svtkFieldData::New();
      fd->ShallowCopy(src->FieldData);
      this->SetFieldData(fd);
      fd->FastDelete();
    }
  }
}

//----------------------------------------------------------------------------
void svtkDataObject::DeepCopy(svtkDataObject* src)
{
  svtkFieldData* srcFieldData = src->GetFieldData();

  this->InternalDataObjectCopy(src);

  if (srcFieldData)
  {
    svtkFieldData* newFieldData = svtkFieldData::New();
    newFieldData->DeepCopy(srcFieldData);
    this->SetFieldData(newFieldData);
    newFieldData->FastDelete();
  }
  else
  {
    this->SetFieldData(nullptr);
  }
}

//----------------------------------------------------------------------------
void svtkDataObject::InternalDataObjectCopy(svtkDataObject* src)
{
  this->DataReleased = src->DataReleased;

  // Do not copy pipeline specific information from data object to
  // data object. This meta-data is specific to the algorithm and the
  // what was requested of it when it executed. What looks like a single
  // piece to an internal algorithm may be a piece to an external
  // algorithm.
  /*
  if(src->Information->Has(DATA_PIECE_NUMBER()))
    {
    this->Information->Set(DATA_PIECE_NUMBER(),
                           src->Information->Get(DATA_PIECE_NUMBER()));
    }
  if(src->Information->Has(DATA_NUMBER_OF_PIECES()))
    {
    this->Information->Set(DATA_NUMBER_OF_PIECES(),
                           src->Information->Get(DATA_NUMBER_OF_PIECES()));
    }
  if(src->Information->Has(DATA_NUMBER_OF_GHOST_LEVELS()))
    {
    this->Information->Set(DATA_NUMBER_OF_GHOST_LEVELS(),
                           src->Information->Get(DATA_NUMBER_OF_GHOST_LEVELS()));
    }
  */
  if (src->Information->Has(DATA_TIME_STEP()))
  {
    this->Information->CopyEntry(src->Information, DATA_TIME_STEP(), 1);
  }

  // This also caused a pipeline problem.
  // An input pipelineMTime was copied to output.  Pipeline did not execute...
  // We do not copy MTime of object, so why should we copy these.
  // this->PipelineMTime = src->PipelineMTime;
  // this->UpdateTime = src->UpdateTime;
  // this->Locality = src->Locality;
}

//----------------------------------------------------------------------------
// This should be a pure virtual method.
void svtkDataObject::Crop(const int*) {}

//----------------------------------------------------------------------------
svtkDataObject* svtkDataObject::GetData(svtkInformation* info)
{
  return info ? info->Get(DATA_OBJECT()) : nullptr;
}

//----------------------------------------------------------------------------
svtkDataObject* svtkDataObject::GetData(svtkInformationVector* v, int i)
{
  return svtkDataObject::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
const char* svtkDataObject::GetAssociationTypeAsString(int associationType)
{
  if (associationType < 0 || associationType >= NUMBER_OF_ASSOCIATIONS)
  {
    svtkGenericWarningMacro("Bad association type.");
    return nullptr;
  }
  return FieldAssociationsNames[associationType];
}

//----------------------------------------------------------------------------
int svtkDataObject::GetAssociationTypeFromString(const char* associationName)
{
  if (!associationName)
  {
    svtkGenericWarningMacro("nullptr association name.");
    return -1;
  }

  // check for the name in the FieldAssociations enum
  for (int i = 0; i < NUMBER_OF_ASSOCIATIONS; i++)
  {
    if (!strcmp(associationName, FieldAssociationsNames[i]))
    {
      return i;
    }
  }

  // check for the name in the AttributeTypes enum
  for (int i = 0; i < NUMBER_OF_ATTRIBUTE_TYPES; i++)
  {
    if (!strcmp(associationName, AttributeTypesNames[i]))
    {
      return i;
    }
  }

  svtkGenericWarningMacro("Bad association name \"" << associationName << "\".");
  return -1;
}

//----------------------------------------------------------------------------
svtkDataSetAttributes* svtkDataObject::GetAttributes(int type)
{
  return svtkDataSetAttributes::SafeDownCast(this->GetAttributesAsFieldData(type));
}

//----------------------------------------------------------------------------
svtkDataArray* svtkDataObject::GetGhostArray(int type)
{
  svtkFieldData* fieldData = this->GetAttributesAsFieldData(type);
  return fieldData ? fieldData->GetArray(svtkDataSetAttributes::GhostArrayName()) : nullptr;
}

//----------------------------------------------------------------------------
svtkFieldData* svtkDataObject::GetAttributesAsFieldData(int type)
{
  switch (type)
  {
    case FIELD:
      return this->FieldData;
  }
  return nullptr;
}

//----------------------------------------------------------------------------
int svtkDataObject::GetAttributeTypeForArray(svtkAbstractArray* arr)
{
  for (int i = 0; i < NUMBER_OF_ATTRIBUTE_TYPES; ++i)
  {
    svtkFieldData* data = this->GetAttributesAsFieldData(i);
    if (data)
    {
      for (int j = 0; j < data->GetNumberOfArrays(); ++j)
      {
        if (data->GetAbstractArray(j) == arr)
        {
          return i;
        }
      }
    }
  }
  return -1;
}

//----------------------------------------------------------------------------
svtkIdType svtkDataObject::GetNumberOfElements(int type)
{
  switch (type)
  {
    case FIELD:
      return this->FieldData->GetNumberOfTuples();
  }
  return 0;
}
