/*=========================================================================

Program:   Visualization Toolkit
Module:    svtkDataObjectTypes.cxx

Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkDataObjectTypes.h"

#include "svtkObjectFactory.h"

#include "svtkAnnotation.h"
#include "svtkAnnotationLayers.h"
#include "svtkCompositeDataSet.h"
#include "svtkDataObject.h"
#include "svtkDataSet.h"
#include "svtkDirectedAcyclicGraph.h"
#include "svtkDirectedGraph.h"
#include "svtkExplicitStructuredGrid.h"
#include "svtkGenericDataSet.h"
#include "svtkGraph.h"
#include "svtkHierarchicalBoxDataSet.h"
#include "svtkHyperTreeGrid.h"
#include "svtkImageData.h"
#include "svtkMultiBlockDataSet.h"
#include "svtkMultiPieceDataSet.h"
#include "svtkNonOverlappingAMR.h"
#include "svtkOverlappingAMR.h"
#include "svtkPartitionedDataSet.h"
#include "svtkPartitionedDataSetCollection.h"
#include "svtkPath.h"
#include "svtkPiecewiseFunction.h"
#include "svtkPointSet.h"
#include "svtkPolyData.h"
#include "svtkRectilinearGrid.h"
#include "svtkReebGraph.h"
#include "svtkSelection.h"
#include "svtkStructuredGrid.h"
#include "svtkStructuredPoints.h"
#include "svtkTable.h"
#include "svtkTree.h"
#include "svtkUndirectedGraph.h"
#include "svtkUniformGrid.h"
#include "svtkUniformHyperTreeGrid.h"
#include "svtkUnstructuredGrid.h"

#include "svtkArrayData.h"
#include "svtkMolecule.h"

svtkStandardNewMacro(svtkDataObjectTypes);

// This list should contain the data object class names in
// the same order as the #define's in svtkType.h. Make sure
// this list is nullptr terminated.
static const char* svtkDataObjectTypesStrings[] = {
  "svtkPolyData",
  "svtkStructuredPoints",
  "svtkStructuredGrid",
  "svtkRectilinearGrid",
  "svtkUnstructuredGrid",
  "svtkPiecewiseFunction",
  "svtkImageData",
  "svtkDataObject",
  "svtkDataSet",
  "svtkPointSet",
  "svtkUniformGrid",
  "svtkCompositeDataSet",
  "svtkMultiGroupDataSet", // OBSOLETE
  "svtkMultiBlockDataSet",
  "svtkHierarchicalDataSet",    // OBSOLETE
  "svtkHierarchicalBoxDataSet", // OBSOLETE
  "svtkGenericDataSet",
  "svtkHyperOctree",     // OBSOLETE
  "svtkTemporalDataSet", // OBSOLETE
  "svtkTable",
  "svtkGraph",
  "svtkTree",
  "svtkSelection",
  "svtkDirectedGraph",
  "svtkUndirectedGraph",
  "svtkMultiPieceDataSet",
  "svtkDirectedAcyclicGraph",
  "svtkArrayData",
  "svtkReebGraph",
  "svtkUniformGridAMR",
  "svtkNonOverlappingAMR",
  "svtkOverlappingAMR",
  "svtkHyperTreeGrid",
  "svtkMolecule",
  "svtkPistonDataObject", // OBSOLETE
  "svtkPath",
  "svtkUnstructuredGridBase",
  "svtkPartitionedDataSet",
  "svtkPartitionedDataSetCollection",
  "svtkUniformHyperTreeGrid",
  "svtkExplicitStructuredGrid",
  nullptr,
};

//----------------------------------------------------------------------------
const char* svtkDataObjectTypes::GetClassNameFromTypeId(int type)
{
  static int numClasses = 0;

  // find length of table
  if (numClasses == 0)
  {
    while (svtkDataObjectTypesStrings[numClasses] != nullptr)
    {
      numClasses++;
    }
  }

  if (type >= 0 && type < numClasses)
  {
    return svtkDataObjectTypesStrings[type];
  }
  else
  {
    return "UnknownClass";
  }
}

//----------------------------------------------------------------------------
int svtkDataObjectTypes::GetTypeIdFromClassName(const char* classname)
{
  if (!classname)
  {
    return -1;
  }

  for (int idx = 0; svtkDataObjectTypesStrings[idx] != nullptr; idx++)
  {
    if (strcmp(svtkDataObjectTypesStrings[idx], classname) == 0)
    {
      return idx;
    }
  }

  return -1;
}

//----------------------------------------------------------------------------
svtkDataObject* svtkDataObjectTypes::NewDataObject(int type)
{
  const char* className = svtkDataObjectTypes::GetClassNameFromTypeId(type);
  if (strcmp(className, "UnknownClass") != 0)
  {
    return svtkDataObjectTypes::NewDataObject(className);
  }

  return nullptr;
}

//----------------------------------------------------------------------------
svtkDataObject* svtkDataObjectTypes::NewDataObject(const char* type)
{

  if (!type)
  {
    svtkGenericWarningMacro("NewDataObject(): You are trying to instantiate DataObjectType \""
      << type << "\" which does not exist.");
    return nullptr;
  }

  // Check for some standard types.
  if (strcmp(type, "svtkImageData") == 0)
  {
    return svtkImageData::New();
  }
  else if (strcmp(type, "svtkDataObject") == 0)
  {
    return svtkDataObject::New();
  }
  else if (strcmp(type, "svtkPolyData") == 0)
  {
    return svtkPolyData::New();
  }
  else if (strcmp(type, "svtkRectilinearGrid") == 0)
  {
    return svtkRectilinearGrid::New();
  }
  else if (strcmp(type, "svtkStructuredGrid") == 0)
  {
    return svtkStructuredGrid::New();
  }
  else if (strcmp(type, "svtkStructuredPoints") == 0)
  {
    return svtkStructuredPoints::New();
  }
  else if (strcmp(type, "svtkUnstructuredGrid") == 0)
  {
    return svtkUnstructuredGrid::New();
  }
  else if (strcmp(type, "svtkUniformGrid") == 0)
  {
    return svtkUniformGrid::New();
  }
  else if (strcmp(type, "svtkMultiBlockDataSet") == 0)
  {
    return svtkMultiBlockDataSet::New();
  }
  else if (strcmp(type, "svtkHierarchicalBoxDataSet") == 0)
  {
    return svtkHierarchicalBoxDataSet::New();
  }
  else if (strcmp(type, "svtkOverlappingAMR") == 0)
  {
    return svtkOverlappingAMR::New();
  }
  else if (strcmp(type, "svtkNonOverlappingAMR") == 0)
  {
    return svtkNonOverlappingAMR::New();
  }
  else if (strcmp(type, "svtkHyperTreeGrid") == 0)
  {
    return svtkHyperTreeGrid::New();
  }
  else if (strcmp(type, "svtkUniformHyperTreeGrid") == 0)
  {
    return svtkUniformHyperTreeGrid::New();
  }
  else if (strcmp(type, "svtkTable") == 0)
  {
    return svtkTable::New();
  }
  else if (strcmp(type, "svtkTree") == 0)
  {
    return svtkTree::New();
  }
  else if (strcmp(type, "svtkSelection") == 0)
  {
    return svtkSelection::New();
  }
  else if (strcmp(type, "svtkDirectedGraph") == 0)
  {
    return svtkDirectedGraph::New();
  }
  else if (strcmp(type, "svtkUndirectedGraph") == 0)
  {
    return svtkUndirectedGraph::New();
  }
  else if (strcmp(type, "svtkMultiPieceDataSet") == 0)
  {
    return svtkMultiPieceDataSet::New();
  }
  else if (strcmp(type, "svtkDirectedAcyclicGraph") == 0)
  {
    return svtkDirectedAcyclicGraph::New();
  }
  else if (strcmp(type, "svtkAnnotation") == 0)
  {
    return svtkAnnotation::New();
  }
  else if (strcmp(type, "svtkAnnotationLayers") == 0)
  {
    return svtkAnnotationLayers::New();
  }
  else if (strcmp(type, "svtkReebGraph") == 0)
  {
    return svtkReebGraph::New();
  }
  else if (strcmp(type, "svtkMolecule") == 0)
  {
    return svtkMolecule::New();
  }
  else if (strcmp(type, "svtkArrayData") == 0)
  {
    return svtkArrayData::New();
  }
  else if (strcmp(type, "svtkPath") == 0)
  {
    return svtkPath::New();
  }
  else if (strcmp(type, "svtkPartitionedDataSet") == 0)
  {
    return svtkPartitionedDataSet::New();
  }
  else if (strcmp(type, "svtkPartitionedDataSetCollection") == 0)
  {
    return svtkPartitionedDataSetCollection::New();
  }
  else if (strcmp(type, "svtkExplicitStructuredGrid") == 0)
  {
    return svtkExplicitStructuredGrid::New();
  }

  svtkGenericWarningMacro("NewDataObject(): You are trying to instantiate DataObjectType \""
    << type << "\" which does not exist.");

  return nullptr;
}

//----------------------------------------------------------------------------
void svtkDataObjectTypes::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

int svtkDataObjectTypes::Validate()
{
  int rc = 0;

  for (int i = 0; svtkDataObjectTypesStrings[i] != nullptr; i++)
  {
    const char* cls = svtkDataObjectTypesStrings[i];
    svtkDataObject* obj = svtkDataObjectTypes::NewDataObject(cls);

    if (obj == nullptr)
    {
      continue;
    }

    int type = obj->GetDataObjectType();
    obj->Delete();

    if (strcmp(svtkDataObjectTypesStrings[type], cls) != 0)
    {
      cerr << "ERROR: In " __FILE__ ", line " << __LINE__ << endl;
      cerr << "Type mismatch for: " << cls << endl;
      cerr << "The value looked up in svtkDataObjectTypesStrings using ";
      cerr << "the index returned by GetDataObjectType() does not match the object type." << endl;
      cerr << "Value from svtkDataObjectTypesStrings[obj->GetDataObjectType()]): ";
      cerr << svtkDataObjectTypesStrings[type] << endl;
      cerr << "Check that the correct value is being returned by GetDataObjectType() ";
      cerr << "for this object type. Also check that the values in svtkDataObjectTypesStrings ";
      cerr << "are in the same order as the #define's in svtkType.h.";
      rc = 1;
      break;
    }
  }
  return rc;
}
