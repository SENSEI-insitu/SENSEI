/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestGarbageCollector.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkCallbackCommand.h"
#include "svtkDebugLeaks.h"
#include "svtkGarbageCollector.h"
#include "svtkObject.h"
#include "svtkSmartPointer.h"

// A class that simulates a reference loop and participates in garbage
// collection.
class svtkTestReferenceLoop : public svtkObject
{
public:
  static svtkTestReferenceLoop* New()
  {
    svtkTestReferenceLoop* ret = new svtkTestReferenceLoop;
    ret->InitializeObjectBase();
    return ret;
  }
  svtkTypeMacro(svtkTestReferenceLoop, svtkObject);

  void Register(svtkObjectBase* o) override { this->RegisterInternal(o, 1); }
  void UnRegister(svtkObjectBase* o) override { this->UnRegisterInternal(o, 1); }

protected:
  svtkTestReferenceLoop()
  {
    this->Other = new svtkTestReferenceLoop(this);
    this->Other->InitializeObjectBase();
  }
  svtkTestReferenceLoop(svtkTestReferenceLoop* other)
  {
    this->Other = other;
    this->Other->Register(this);
  }
  ~svtkTestReferenceLoop() override
  {
    if (this->Other)
    {
      this->Other->UnRegister(this);
      this->Other = nullptr;
    }
  }

  void ReportReferences(svtkGarbageCollector* collector) override
  {
    svtkGarbageCollectorReport(collector, this->Other, "Other");
  }

  svtkTestReferenceLoop* Other;

private:
  svtkTestReferenceLoop(const svtkTestReferenceLoop&) = delete;
  void operator=(const svtkTestReferenceLoop&) = delete;
};

// A callback that reports when it is called.
static int called = 0;
static void MyDeleteCallback(svtkObject*, unsigned long, void*, void*)
{
  called = 1;
}

// Main test function.
int TestGarbageCollector(int, char*[])
{
  // Create a callback that reports when it is called.
  svtkSmartPointer<svtkCallbackCommand> cc = svtkSmartPointer<svtkCallbackCommand>::New();
  cc->SetCallback(MyDeleteCallback);

  // Create an object and delete it immediately.  It should be
  // immediately collected.
  svtkTestReferenceLoop* obj = svtkTestReferenceLoop::New();
  obj->AddObserver(svtkCommand::DeleteEvent, cc);
  called = 0;
  obj->Delete();
  if (!called)
  {
    cerr << "Object not immediately collected." << endl;
    return 1;
  }

  // Create an object, enable deferred collection, and delete it.  It
  // should not be collected yet.
  obj = svtkTestReferenceLoop::New();
  obj->AddObserver(svtkCommand::DeleteEvent, cc);
  svtkGarbageCollector::DeferredCollectionPush();
  called = 0;
  obj->Delete();
  if (called)
  {
    cerr << "Object collection not deferred." << endl;
    return 1;
  }

  // Disable deferred collection.  The object should be deleted now.
  svtkGarbageCollector::DeferredCollectionPop();
  if (!called)
  {
    cerr << "Deferred collection did not collect object." << endl;
    return 1;
  }

  return 0;
}
