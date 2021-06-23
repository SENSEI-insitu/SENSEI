/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestSelectionSubtract

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// .NAME
// .SECTION Description
// this program tests the svtkSelection::Subtract method

#include "svtkIdTypeArray.h"
#include "svtkIndent.h"
#include "svtkSelection.h"
#include "svtkSelectionNode.h"
#include "svtkSmartPointer.h"

#include <iostream>
using namespace std;

#define SVTK_CREATE(type, name) svtkSmartPointer<type> name = svtkSmartPointer<type>::New()

#define DEBUG 0

#if DEBUG
// ------------------------------------------------------------------------------------------------
static void PrintSelectionNodes(svtkSmartPointer<svtkSelection>& sel, const char* tag = nullptr)
{
  svtkIdType numNodes = sel->GetNumberOfNodes();

  if (tag)
  {
    cout << tag << endl;
  }

  for (int iNode = 0; iNode < numNodes; iNode++)
  {
    if (tag)
      cout << "\t";
    cout << "Node: " << iNode << endl;
    svtkIdType listSize = sel->GetNode(iNode)->GetSelectionList()->GetNumberOfTuples();
    for (int iVal = 0; iVal < listSize; iVal++)
    {
      if (tag)
        cout << "\t";
      cout << "\t" << iVal << "\t" << sel->GetNode(iNode)->GetSelectionList()->GetVariantValue(iVal)
           << endl;
    }
  }
}
#endif

// ------------------------------------------------------------------------------------------------
int TestSelectionSubtract(int, char*[])
{
  // Create a selection, sel1, of PEDIGREEIDS containing {1, 2, 3}
  SVTK_CREATE(svtkSelection, sel1);
  SVTK_CREATE(svtkSelectionNode, sel1Node);
  SVTK_CREATE(svtkIdTypeArray, sel1Arr);
  sel1->AddNode(sel1Node);
  sel1Node->SetContentType(svtkSelectionNode::PEDIGREEIDS);
  sel1Node->SetFieldType(svtkSelectionNode::VERTEX);
  sel1Node->SetSelectionList(sel1Arr);
  sel1Arr->InsertNextValue(1);
  sel1Arr->InsertNextValue(2);
  sel1Arr->InsertNextValue(3);

  // Create a selection, sel2, of PEDIGREEIDS containing {2, 3}
  SVTK_CREATE(svtkSelection, sel2);
  SVTK_CREATE(svtkSelectionNode, sel2Node);
  SVTK_CREATE(svtkIdTypeArray, sel2Arr);
  sel2->AddNode(sel2Node);
  sel2Node->SetContentType(svtkSelectionNode::PEDIGREEIDS);
  sel2Node->SetFieldType(svtkSelectionNode::VERTEX);
  sel2Node->SetSelectionList(sel2Arr);
  sel2Arr->InsertNextValue(3);
  sel2Arr->InsertNextValue(1);

  // debugging
#if DEBUG
  PrintSelectionNodes(sel1, "sel1");
  PrintSelectionNodes(sel2, "sel2");
#endif

  // Subtract sel2 from sel1
#if DEBUG
  cout << endl << "Subtract sel2 from sel1 ..." << endl << endl;
#endif
  sel1->Subtract(sel2);

  // debugging
#if DEBUG
  PrintSelectionNodes(sel1, "sel1");
#endif

  // Correctness check.
  bool failed = false;
  cout << "Check # of nodes == 1 ....... ";
  if (sel1->GetNumberOfNodes() != 1)
  {
    cout << "FAILED" << endl;
    failed = true;
  }
  else
  {
    cout << "OK" << endl;
  }

  cout << "Check # of tuples == 1 ...... ";
  if (sel1->GetNode(0)->GetSelectionList()->GetNumberOfTuples() != 1)
  {
    cout << "FAILED" << endl;
    failed = true;
  }
  else
  {
    cout << "OK" << endl;
  }

  cout << "Check selection value is 2 .. ";
  if (sel1->GetNode(0)->GetSelectionList()->GetVariantValue(0) != 2)
  {
    cout << "FAILED" << endl;
    failed = true;
  }
  else
  {
    cout << "OK" << endl;
  }

  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
