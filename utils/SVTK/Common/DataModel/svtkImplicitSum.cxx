/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImplicitSum.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkImplicitSum.h"

#include "svtkDoubleArray.h"
#include "svtkImplicitFunctionCollection.h"
#include "svtkObjectFactory.h"

#include <cmath>

svtkStandardNewMacro(svtkImplicitSum);

//----------------------------------------------------------------------------
// Constructor.
svtkImplicitSum::svtkImplicitSum()
{
  this->FunctionList = svtkImplicitFunctionCollection::New();
  this->Weights = svtkDoubleArray::New();
  this->Weights->SetNumberOfComponents(1);
  this->TotalWeight = 0.0;
  this->NormalizeByWeight = 0;
}

//----------------------------------------------------------------------------
svtkImplicitSum::~svtkImplicitSum()
{
  this->FunctionList->Delete();
  this->Weights->Delete();
}

//----------------------------------------------------------------------------
svtkMTimeType svtkImplicitSum::GetMTime()
{
  svtkMTimeType fMtime;
  svtkMTimeType mtime = this->svtkImplicitFunction::GetMTime();
  svtkImplicitFunction* f;

  fMtime = this->Weights->GetMTime();
  if (fMtime > mtime)
  {
    mtime = fMtime;
  }

  svtkCollectionSimpleIterator sit;
  for (this->FunctionList->InitTraversal(sit);
       (f = this->FunctionList->GetNextImplicitFunction(sit));)
  {
    fMtime = f->GetMTime();
    if (fMtime > mtime)
    {
      mtime = fMtime;
    }
  }
  return mtime;
}

//----------------------------------------------------------------------------
// Add another implicit function to the list of functions.
void svtkImplicitSum::AddFunction(svtkImplicitFunction* f, double scale)
{
  this->Modified();
  this->FunctionList->AddItem(f);
  this->Weights->InsertNextValue(scale);
  this->CalculateTotalWeight();
}

//----------------------------------------------------------------------------
void svtkImplicitSum::SetFunctionWeight(svtkImplicitFunction* f, double scale)
{
  int loc = this->FunctionList->IsItemPresent(f);
  if (!loc)
  {
    svtkWarningMacro("Function not found in function list");
    return;
  }
  loc--; // IsItemPresent returns index+1.

  if (this->Weights->GetValue(loc) != scale)
  {
    this->Modified();
    this->Weights->SetValue(loc, scale);
    this->CalculateTotalWeight();
  }
}

//----------------------------------------------------------------------------
void svtkImplicitSum::RemoveAllFunctions()
{
  this->Modified();
  this->FunctionList->RemoveAllItems();
  this->Weights->Initialize();
  this->TotalWeight = 0.0;
}

//----------------------------------------------------------------------------
void svtkImplicitSum::CalculateTotalWeight()
{
  this->TotalWeight = 0.0;

  for (int i = 0; i < this->Weights->GetNumberOfTuples(); ++i)
  {
    this->TotalWeight += this->Weights->GetValue(i);
  }
}

//----------------------------------------------------------------------------
// Evaluate sum of implicit functions.
double svtkImplicitSum::EvaluateFunction(double x[3])
{
  double sum = 0;
  double c;
  int i;
  svtkImplicitFunction* f;
  double* weights = this->Weights->GetPointer(0);

  svtkCollectionSimpleIterator sit;
  for (i = 0, this->FunctionList->InitTraversal(sit);
       (f = this->FunctionList->GetNextImplicitFunction(sit)); i++)
  {
    c = weights[i];
    if (c != 0.0)
    {
      sum += f->FunctionValue(x) * c;
    }
  }
  if (this->NormalizeByWeight && this->TotalWeight != 0.0)
  {
    sum /= this->TotalWeight;
  }
  return sum;
}

//----------------------------------------------------------------------------
// Evaluate gradient of sum of functions (valid only if linear)
void svtkImplicitSum::EvaluateGradient(double x[3], double g[3])
{
  double c;
  int i;
  double gtmp[3];
  svtkImplicitFunction* f;
  double* weights = this->Weights->GetPointer(0);

  g[0] = g[1] = g[2] = 0.0;
  svtkCollectionSimpleIterator sit;
  for (i = 0, this->FunctionList->InitTraversal(sit);
       (f = this->FunctionList->GetNextImplicitFunction(sit)); i++)
  {
    c = weights[i];
    if (c != 0.0)
    {
      f->FunctionGradient(x, gtmp);
      g[0] += gtmp[0] * c;
      g[1] += gtmp[1] * c;
      g[2] += gtmp[2] * c;
    }
  }

  if (this->NormalizeByWeight && this->TotalWeight != 0.0)
  {
    g[0] /= this->TotalWeight;
    g[1] /= this->TotalWeight;
    g[2] /= this->TotalWeight;
  }
}

//----------------------------------------------------------------------------
void svtkImplicitSum::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "NormalizeByWeight: " << (this->NormalizeByWeight ? "On\n" : "Off\n");

  os << indent << "Function List:\n";
  this->FunctionList->PrintSelf(os, indent.GetNextIndent());

  os << indent << "Weights:\n";
  this->Weights->PrintSelf(os, indent.GetNextIndent());
}
