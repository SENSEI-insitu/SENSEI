/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuadratureSchemeDefinition.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkQuadratureSchemeDefinition.h"

#include "svtkCellType.h"
#include "svtkInformationQuadratureSchemeDefinitionVectorKey.h"
#include "svtkInformationStringKey.h"
#include "svtkObjectFactory.h"
#include "svtkXMLDataElement.h"
#include <sstream>
using std::istringstream;
using std::ostringstream;
#include <string>
using std::string;

svtkStandardNewMacro(svtkQuadratureSchemeDefinition);

//-----------------------------------------------------------------------------
svtkInformationKeyMacro(svtkQuadratureSchemeDefinition, DICTIONARY, QuadratureSchemeDefinitionVector);

svtkInformationKeyMacro(svtkQuadratureSchemeDefinition, QUADRATURE_OFFSET_ARRAY_NAME, String);

//-----------------------------------------------------------------------------
svtkQuadratureSchemeDefinition::svtkQuadratureSchemeDefinition()
{
  this->ShapeFunctionWeights = nullptr;
  this->QuadratureWeights = nullptr;
  this->Clear();
}

//-----------------------------------------------------------------------------
svtkQuadratureSchemeDefinition::~svtkQuadratureSchemeDefinition()
{
  this->Clear();
}

//-----------------------------------------------------------------------------
int svtkQuadratureSchemeDefinition::DeepCopy(const svtkQuadratureSchemeDefinition* other)
{
  this->ShapeFunctionWeights = nullptr;
  this->QuadratureWeights = nullptr;
  this->Clear();
  //
  this->CellType = other->CellType;
  this->QuadratureKey = other->QuadratureKey;
  this->NumberOfNodes = other->NumberOfNodes;
  this->NumberOfQuadraturePoints = other->NumberOfQuadraturePoints;
  //
  this->SecureResources();
  //
  this->SetShapeFunctionWeights(other->GetShapeFunctionWeights());
  this->SetQuadratureWeights(other->GetQuadratureWeights());
  //
  return 1;
}

//-----------------------------------------------------------------------------
void svtkQuadratureSchemeDefinition::Clear()
{
  this->ReleaseResources();
  this->CellType = -1;
  this->QuadratureKey = -1;
  this->NumberOfNodes = 0;
  this->NumberOfQuadraturePoints = 0;
}

//-----------------------------------------------------------------------------
void svtkQuadratureSchemeDefinition::Initialize(
  int cellType, int numberOfNodes, int numberOfQuadraturePoints, double* shapeFunctionWeights)
{
  this->ReleaseResources();
  //
  this->CellType = cellType;
  this->QuadratureKey = -1;
  this->NumberOfNodes = numberOfNodes;
  this->NumberOfQuadraturePoints = numberOfQuadraturePoints;
  //
  this->SecureResources();
  //
  this->SetShapeFunctionWeights(shapeFunctionWeights);
}

//-----------------------------------------------------------------------------
void svtkQuadratureSchemeDefinition::Initialize(int cellType, int numberOfNodes,
  int numberOfQuadraturePoints, double* shapeFunctionWeights, double* quadratureWeights)
{
  this->ReleaseResources();
  //
  this->CellType = cellType;
  this->QuadratureKey = -1;
  this->NumberOfNodes = numberOfNodes;
  this->NumberOfQuadraturePoints = numberOfQuadraturePoints;
  //
  this->SecureResources();
  //
  this->SetShapeFunctionWeights(shapeFunctionWeights);
  this->SetQuadratureWeights(quadratureWeights);
}

//-----------------------------------------------------------------------------
void svtkQuadratureSchemeDefinition::ReleaseResources()
{
  delete[] this->ShapeFunctionWeights;
  this->ShapeFunctionWeights = nullptr;

  delete[] this->QuadratureWeights;
  this->QuadratureWeights = nullptr;
}

//-----------------------------------------------------------------------------
int svtkQuadratureSchemeDefinition::SecureResources()
{
  if ((this->NumberOfQuadraturePoints <= 0) || (this->NumberOfNodes <= 0))
  {
    svtkWarningMacro("Failed to allocate. Invalid buffer size.");
    return 0;
  }

  // Delete weights if they have been allocated
  this->ReleaseResources();

  // Shape function weights, one vector for each quad point.
  this->ShapeFunctionWeights = new double[this->NumberOfQuadraturePoints * this->NumberOfNodes];
  for (int i = 0; i < this->NumberOfQuadraturePoints * this->NumberOfNodes; i++)
  {
    this->ShapeFunctionWeights[i] = 0.0;
  }

  // Quadrature weights, one double for each quad point
  this->QuadratureWeights = new double[this->NumberOfQuadraturePoints];
  for (int i = 0; i < this->NumberOfQuadraturePoints; i++)
  {
    this->QuadratureWeights[i] = 0.0;
  }
  return 1;
}

//-----------------------------------------------------------------------------
void svtkQuadratureSchemeDefinition::SetShapeFunctionWeights(const double* W)
{
  if ((this->NumberOfQuadraturePoints <= 0) || (this->NumberOfNodes <= 0) ||
    (this->ShapeFunctionWeights == nullptr) || !W)
  {
    return;
  }
  // Copy
  int n = this->NumberOfQuadraturePoints * this->NumberOfNodes;
  for (int i = 0; i < n; ++i)
  {
    this->ShapeFunctionWeights[i] = W[i];
  }
}

//-----------------------------------------------------------------------------
void svtkQuadratureSchemeDefinition::SetQuadratureWeights(const double* W)
{
  if ((this->NumberOfQuadraturePoints <= 0) || (this->NumberOfNodes <= 0) ||
    (this->QuadratureWeights == nullptr) || !W)
  {
    return;
  }
  // Copy
  for (int i = 0; i < this->NumberOfQuadraturePoints; ++i)
  {
    this->QuadratureWeights[i] = W[i];
  }
}

//-----------------------------------------------------------------------------
void svtkQuadratureSchemeDefinition::PrintSelf(ostream& sout, svtkIndent indent)
{

  svtkObject::PrintSelf(sout, indent);

  double* pSfWt = this->ShapeFunctionWeights;

  for (int ptId = 0; ptId < this->NumberOfQuadraturePoints; ++ptId)
  {
    sout << indent << "(" << pSfWt[0];
    ++pSfWt;
    for (int nodeId = 1; nodeId < this->NumberOfNodes; ++nodeId)
    {
      sout << indent << ", " << pSfWt[0];
      ++pSfWt;
    }
    sout << ")" << endl;
  }
}

// NOTE: These are used by XML readers/writers.
//-----------------------------------------------------------------------------
ostream& operator<<(ostream& sout, const svtkQuadratureSchemeDefinition& def)
{
  /*
  stream will have this space delimited format:
  [cell type][number of cell nodes][number quadrature points][Qp1 ... QpN][Qwt1...QwtN]
  */

  // Size of arrays
  int nQuadPts = def.GetNumberOfQuadraturePoints();
  int nNodes = def.GetNumberOfNodes();

  // Write header
  sout << def.GetCellType() << " " << nNodes << " " << nQuadPts;

  if ((nNodes > 0) && (nQuadPts > 0))
  {
    sout.setf(ios::floatfield, ios::scientific);
    sout.precision(16);

    const double* pWt;
    // Write shape function weights
    pWt = def.GetShapeFunctionWeights();
    for (int ptId = 0; ptId < nQuadPts; ++ptId)
    {
      for (int nodeId = 0; nodeId < nNodes; ++nodeId)
      {
        sout << " " << pWt[0];
        ++pWt;
      }
    }
    // Write quadrature weights
    pWt = def.GetQuadratureWeights();
    for (int nodeId = 0; nodeId < nNodes; ++nodeId)
    {
      sout << " " << pWt[0];
      ++pWt;
    }
  }
  else
  {
    svtkGenericWarningMacro("Empty definition written to stream.");
  }
  return sout;
}

//-----------------------------------------------------------------------------
istream& operator>>(istream& sin, svtkQuadratureSchemeDefinition& def)
{
  /*
  stream will have this space delimited format:
  [cell type][number of cell nodes][number quadrature points][Qp1 ... QpN][Qwt1...QwtN]
  */

  // read the header
  int cellType, nNodes, nQuadPts;
  sin >> cellType >> nNodes >> nQuadPts;

  double *SfWt = nullptr, *QWt = nullptr, *pWt = nullptr;
  if ((nNodes > 0) && (nQuadPts > 0))
  {
    // read shape function weights
    SfWt = new double[nQuadPts * nNodes];
    pWt = SfWt;
    for (int ptId = 0; ptId < nQuadPts; ++ptId)
    {
      for (int nodeId = 0; nodeId < nNodes; ++nodeId)
      {
        sin >> pWt[0];
        ++pWt;
      }
    }
    // Write quadrature weights
    QWt = new double[nQuadPts];
    pWt = QWt;
    for (int nodeId = 0; nodeId < nNodes; ++nodeId)
    {
      sin >> pWt[0];
      ++pWt;
    }
  }
  else
  {
    svtkGenericWarningMacro("Empty definition found in stream.");
  }

  // initialize the object
  def.Initialize(cellType, nNodes, nQuadPts, SfWt, QWt);

  // clean up
  delete[] SfWt;
  delete[] QWt;

  return sin;
}

//-----------------------------------------------------------------------------
int svtkQuadratureSchemeDefinition::SaveState(svtkXMLDataElement* root)
{
  // Quick sanity check, we're not nesting rather treating
  // this as a root, to be nested by the caller as needed.
  if (root->GetName() != nullptr || root->GetNumberOfNestedElements() > 0)
  {
    svtkWarningMacro("Can't save state to non-empty element.");
    return 0;
  }

  root->SetName("svtkQuadratureSchemeDefinition");

  svtkXMLDataElement* e;
  e = svtkXMLDataElement::New();
  e->SetName("CellType");
  e->SetIntAttribute("value", this->CellType);
  root->AddNestedElement(e);
  e->Delete();

  e = svtkXMLDataElement::New();
  e->SetName("NumberOfNodes");
  e->SetIntAttribute("value", this->NumberOfNodes);
  root->AddNestedElement(e);
  e->Delete();

  e = svtkXMLDataElement::New();
  e->SetName("NumberOfQuadraturePoints");
  e->SetIntAttribute("value", this->NumberOfQuadraturePoints);
  root->AddNestedElement(e);
  e->Delete();

  svtkXMLDataElement* eShapeWts = svtkXMLDataElement::New();
  eShapeWts->SetName("ShapeFunctionWeights");
  eShapeWts->SetCharacterDataWidth(4);
  root->AddNestedElement(eShapeWts);
  eShapeWts->Delete();

  svtkXMLDataElement* eQuadWts = svtkXMLDataElement::New();
  eQuadWts->SetName("QuadratureWeights");
  eQuadWts->SetCharacterDataWidth(4);
  root->AddNestedElement(eQuadWts);
  eQuadWts->Delete();

  if ((this->NumberOfNodes > 0) && (this->NumberOfQuadraturePoints > 0))
  {
    // Write shape function weights
    ostringstream ssShapeWts;
    ssShapeWts.setf(ios::floatfield, ios::scientific);
    ssShapeWts.precision(16);
    ssShapeWts << this->ShapeFunctionWeights[0];
    int nIds = this->NumberOfNodes * this->NumberOfQuadraturePoints;
    for (int id = 1; id < nIds; ++id)
    {
      ssShapeWts << " " << this->ShapeFunctionWeights[id];
    }
    string sShapeWts = ssShapeWts.str();
    eShapeWts->SetCharacterData(sShapeWts.c_str(), static_cast<int>(sShapeWts.size()));

    // Write quadrature weights
    ostringstream ssQuadWts;
    ssQuadWts.setf(ios::floatfield, ios::scientific);
    ssQuadWts.precision(16);
    ssQuadWts << this->QuadratureWeights[0];
    for (int id = 1; id < this->NumberOfQuadraturePoints; ++id)
    {
      ssQuadWts << " " << this->QuadratureWeights[id];
    }
    string sQuadWts = ssQuadWts.str();
    eQuadWts->SetCharacterData(sQuadWts.c_str(), static_cast<int>(sQuadWts.size()));
  }
  else
  {
    svtkGenericWarningMacro("Empty definition written to stream.");
    return 0;
  }

  return 1;
}

//-----------------------------------------------------------------------------
int svtkQuadratureSchemeDefinition::RestoreState(svtkXMLDataElement* root)
{
  // A quick sanity check to be sure we have the correct tag.
  if (strcmp(root->GetName(), "svtkQuadratureSchemeDefinition") != 0)
  {
    svtkWarningMacro("Attempting to restore the state in "
      << root->GetName() << " into svtkQuadratureSchemeDefinition.");
    return 0;
  }

  svtkXMLDataElement* e;
  const char* value;
  // Transfer state from XML hierarchy.
  e = root->FindNestedElementWithName("CellType");
  if (e == nullptr)
  {
    svtkWarningMacro("Expected nested element \"CellType\" "
                    "is not present.");
    return 0;
  }
  value = e->GetAttribute("value");
  this->CellType = atoi(value);
  //
  e = root->FindNestedElementWithName("NumberOfNodes");
  if (e == nullptr)
  {
    svtkWarningMacro("Expected nested element \"NumberOfNodes\" "
                    "is not present.");
    return 0;
  }
  value = e->GetAttribute("value");
  this->NumberOfNodes = atoi(value);
  //
  e = root->FindNestedElementWithName("NumberOfQuadraturePoints");
  if (e == nullptr)
  {
    svtkWarningMacro("Expected nested element \"NumberOfQuadraturePoints\" "
                    "is not present.");
    return 0;
  }
  value = e->GetAttribute("value");
  this->NumberOfQuadraturePoints = atoi(value);
  // Extract the weights.
  if (this->SecureResources())
  {
    istringstream issWts;
    //
    e = root->FindNestedElementWithName("ShapeFunctionWeights");
    if (e == nullptr)
    {
      svtkWarningMacro("Expected nested element \"ShapeFunctionWeights\" "
                      "is not present.");
      return 0;
    }
    value = e->GetCharacterData();
    if (value == nullptr)
    {
      svtkWarningMacro("Character data in nested element"
                      " \"ShapeFunctionWeights\" is not present.");
      return 0;
    }
    issWts.str(value);
    int nWts = this->NumberOfNodes * this->NumberOfQuadraturePoints;
    for (int id = 0; id < nWts; ++id)
    {
      if (!issWts.good())
      {
        svtkWarningMacro("Character data for \"ShapeFunctionWeights\" "
                        "is short.");
        return 0;
      }
      issWts >> this->ShapeFunctionWeights[id];
    }
    //
    e = root->FindNestedElementWithName("QuadratureWeights");
    if (e == nullptr)
    {
      svtkWarningMacro("Expected element \"QuadratureWeights\" "
                      "is not present.");
      return 0;
    }
    value = e->GetCharacterData();
    if (value == nullptr)
    {
      svtkWarningMacro("Character data in expected nested element"
                      " \"QuadratureWeights\" is not present.");
      return 0;
    }
    issWts.str(value);
    for (int id = 0; id < this->NumberOfQuadraturePoints; ++id)
    {
      if (!issWts.good())
      {
        svtkWarningMacro("Character data for \"QuadratureWeights\" "
                        "is short.");
        return 0;
      }
      issWts >> this->QuadratureWeights[id];
    }
  }

  return 1;
}
