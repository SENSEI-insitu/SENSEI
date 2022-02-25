/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImplicitBoolean.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkImplicitBoolean.h"

#include "svtkImplicitFunctionCollection.h"
#include "svtkObjectFactory.h"

#include <cmath>

svtkStandardNewMacro(svtkImplicitBoolean);

// Construct with union operation.
svtkImplicitBoolean::svtkImplicitBoolean()
{
  this->OperationType = SVTK_UNION;
  this->FunctionList = svtkImplicitFunctionCollection::New();
}

svtkImplicitBoolean::~svtkImplicitBoolean()
{
  this->FunctionList->Delete();
}

svtkMTimeType svtkImplicitBoolean::GetMTime()
{
  svtkMTimeType fMtime;
  svtkMTimeType mtime = this->svtkImplicitFunction::GetMTime();
  svtkImplicitFunction* f;

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

// Add another implicit function to the list of functions.
void svtkImplicitBoolean::AddFunction(svtkImplicitFunction* f)
{
  if (!this->FunctionList->IsItemPresent(f))
  {
    this->Modified();
    this->FunctionList->AddItem(f);
  }
}

// Remove a function from the list of implicit functions to boolean.
void svtkImplicitBoolean::RemoveFunction(svtkImplicitFunction* f)
{
  if (this->FunctionList->IsItemPresent(f))
  {
    this->Modified();
    this->FunctionList->RemoveItem(f);
  }
}

// Evaluate boolean combinations of implicit function using current operator.
double svtkImplicitBoolean::EvaluateFunction(double x[3])
{
  double value = 0;
  double v;
  svtkImplicitFunction* f;

  if (this->FunctionList->GetNumberOfItems() == 0)
  {
    return value;
  }

  svtkCollectionSimpleIterator sit;
  if (this->OperationType == SVTK_UNION)
  { // take minimum value
    for (value = SVTK_DOUBLE_MAX, this->FunctionList->InitTraversal(sit);
         (f = this->FunctionList->GetNextImplicitFunction(sit));)
    {
      if ((v = f->FunctionValue(x)) < value)
      {
        value = v;
      }
    }
  }

  else if (this->OperationType == SVTK_INTERSECTION)
  { // take maximum value
    for (value = -SVTK_DOUBLE_MAX, this->FunctionList->InitTraversal(sit);
         (f = this->FunctionList->GetNextImplicitFunction(sit));)
    {
      if ((v = f->FunctionValue(x)) > value)
      {
        value = v;
      }
    }
  }

  else if (this->OperationType == SVTK_UNION_OF_MAGNITUDES)
  { // take minimum absolute value
    for (value = SVTK_DOUBLE_MAX, this->FunctionList->InitTraversal(sit);
         (f = this->FunctionList->GetNextImplicitFunction(sit));)
    {
      if ((v = fabs(f->FunctionValue(x))) < value)
      {
        value = v;
      }
    }
  }

  else // difference
  {
    svtkImplicitFunction* firstF;
    this->FunctionList->InitTraversal(sit);
    if ((firstF = this->FunctionList->GetNextImplicitFunction(sit)) != nullptr)
    {
      value = firstF->FunctionValue(x);
    }

    for (this->FunctionList->InitTraversal(sit);
         (f = this->FunctionList->GetNextImplicitFunction(sit));)
    {
      if (f != firstF)
      {
        if ((v = (-1.0) * f->FunctionValue(x)) > value)
        {
          value = v;
        }
      }
    }
  } // else

  return value;
}

// Evaluate gradient of boolean combination.
void svtkImplicitBoolean::EvaluateGradient(double x[3], double g[3])
{
  double value = 0;
  double v;
  svtkImplicitFunction* f;
  svtkCollectionSimpleIterator sit;

  if (this->FunctionList->GetNumberOfItems() == 0)
  {
    g[0] = 0;
    g[1] = 0;
    g[2] = 0;
    return;
  }

  if (this->OperationType == SVTK_UNION)
  { // take minimum value
    for (value = SVTK_DOUBLE_MAX, this->FunctionList->InitTraversal(sit);
         (f = this->FunctionList->GetNextImplicitFunction(sit));)
    {
      if ((v = f->FunctionValue(x)) < value)
      {
        value = v;
        f->FunctionGradient(x, g);
      }
    }
  }

  else if (this->OperationType == SVTK_INTERSECTION)
  { // take maximum value
    for (value = -SVTK_DOUBLE_MAX, this->FunctionList->InitTraversal(sit);
         (f = this->FunctionList->GetNextImplicitFunction(sit));)
    {
      if ((v = f->FunctionValue(x)) > value)
      {
        value = v;
        f->FunctionGradient(x, g);
      }
    }
  }

  if (this->OperationType == SVTK_UNION_OF_MAGNITUDES)
  { // take minimum value
    for (value = SVTK_DOUBLE_MAX, this->FunctionList->InitTraversal(sit);
         (f = this->FunctionList->GetNextImplicitFunction(sit));)
    {
      if ((v = fabs(f->FunctionValue(x))) < value)
      {
        value = v;
        f->FunctionGradient(x, g);
      }
    }
  }

  else // difference
  {
    double gTemp[3];
    svtkImplicitFunction* firstF;
    this->FunctionList->InitTraversal(sit);
    if ((firstF = this->FunctionList->GetNextImplicitFunction(sit)) != nullptr)
    {
      value = firstF->FunctionValue(x);
      firstF->FunctionGradient(x, gTemp);
      g[0] = -1.0 * gTemp[0];
      g[1] = -1.0 * gTemp[1];
      g[2] = -1.0 * gTemp[2];
    }

    for (this->FunctionList->InitTraversal(sit);
         (f = this->FunctionList->GetNextImplicitFunction(sit));)
    {
      if (f != firstF)
      {
        if ((v = (-1.0) * f->FunctionValue(x)) > value)
        {
          value = v;
          f->FunctionGradient(x, gTemp);
          g[0] = -1.0 * gTemp[0];
          g[1] = -1.0 * gTemp[1];
          g[2] = -1.0 * gTemp[2];
        }
      }
    }
  } // else
}

void svtkImplicitBoolean::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Function List:\n";
  this->FunctionList->PrintSelf(os, indent.GetNextIndent());

  os << indent << "Operator Type: ";
  if (this->OperationType == SVTK_INTERSECTION)
  {
    os << "SVTK_INTERSECTION\n";
  }
  else if (this->OperationType == SVTK_UNION)
  {
    os << "SVTK_UNION\n";
  }
  else if (this->OperationType == SVTK_UNION_OF_MAGNITUDES)
  {
    os << "SVTK_UNION_OF_MAGNITUDES\n";
  }
  else
  {
    os << "SVTK_DIFFERENCE\n";
  }
}
