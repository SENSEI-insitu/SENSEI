/*=========================================================================

  Program:   Visualization Toolkit
  Module:    otherArrays.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkNew.h"
#include "svtkObject.h"
#include "svtkObjectFactory.h"
#include "svtkSMPThreadLocal.h"
#include "svtkSMPThreadLocalObject.h"
#include "svtkSMPTools.h"
#include <functional>
#include <vector>

static const int Target = 10000;

class ARangeFunctor
{
public:
  svtkSMPThreadLocal<int> Counter;

  ARangeFunctor()
    : Counter(0)
  {
  }

  void operator()(svtkIdType begin, svtkIdType end)
  {
    for (int i = begin; i < end; i++)
      this->Counter.Local()++;
  }
};

class MySVTKClass : public svtkObject
{
  int Value;

  MySVTKClass()
    : Value(0)
  {
  }

public:
  svtkTypeMacro(MySVTKClass, svtkObject);
  static MySVTKClass* New();

  void SetInitialValue(int value) { this->Value = value; }

  int GetValue() { return this->Value; }

  void Increment() { this->Value++; }
};

svtkStandardNewMacro(MySVTKClass);

class InitializableFunctor
{
public:
  svtkSMPThreadLocalObject<MySVTKClass> CounterObject;

  void Initialize() { CounterObject.Local()->SetInitialValue(5); }

  void operator()(svtkIdType begin, svtkIdType end)
  {
    for (int i = begin; i < end; i++)
      this->CounterObject.Local()->Increment();
  }

  void Reduce() {}
};

// For sorting comparison
bool myComp(double a, double b)
{
  return (a < b);
}

int TestSMP(int, char*[])
{
  // svtkSMPTools::Initialize(8);

  ARangeFunctor functor1;

  svtkSMPTools::For(0, Target, functor1);

  svtkSMPThreadLocal<int>::iterator itr1 = functor1.Counter.begin();
  svtkSMPThreadLocal<int>::iterator end1 = functor1.Counter.end();

  int total = 0;
  while (itr1 != end1)
  {
    total += *itr1;
    ++itr1;
  }

  if (total != Target)
  {
    cerr << "Error: ARangeFunctor did not generate " << Target << endl;
    return 1;
  }

  InitializableFunctor functor2;

  svtkSMPTools::For(0, Target, functor2);

  svtkSMPThreadLocalObject<MySVTKClass>::iterator itr2 = functor2.CounterObject.begin();
  svtkSMPThreadLocalObject<MySVTKClass>::iterator end2 = functor2.CounterObject.end();

  int newTarget = Target;
  total = 0;
  while (itr2 != end2)
  {
    newTarget += 5; // This is the initial value of each object
    total += (*itr2)->GetValue();
    ++itr2;
  }

  if (total != newTarget)
  {
    cerr << "Error: InitializableRangeFunctor did not generate " << newTarget << endl;
    return 1;
  }

  // Test sorting
  double data0[] = { 2, 1, 0, 3, 9, 6, 7, 3, 8, 4, 5 };
  std::vector<double> myvector(data0, data0 + 11);
  double data1[] = { 2, 1, 0, 3, 9, 6, 7, 3, 8, 4, 5 };
  double sdata[] = { 0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9 };

  // using default comparison (operator <):
  svtkSMPTools::Sort(myvector.begin(), myvector.begin() + 11);
  for (int i = 0; i < 11; ++i)
  {
    if (myvector[i] != sdata[i])
    {
      cerr << "Error: Bad vector sort!" << endl;
      return 1;
    }
  }

  svtkSMPTools::Sort(data1, data1 + 11, myComp);
  for (int i = 0; i < 11; ++i)
  {
    if (data1[i] != sdata[i])
    {
      cerr << "Error: Bad comparison sort!" << endl;
      return 1;
    }
  }

  return 0;
}
