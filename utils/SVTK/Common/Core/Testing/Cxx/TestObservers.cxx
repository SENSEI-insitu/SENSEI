/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestSmartPointer.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME Test of Observers.
// .SECTION Description
// Tests svtkObject::AddObserver templated API

#include "svtkObjectFactory.h"
#include "svtkSmartPointer.h"
#include <map>

class svtkHandler : public svtkObject
{
public:
  static std::map<int, int> EventCounts;
  static int VoidEventCounts;

public:
  static svtkHandler* New();
  svtkTypeMacro(svtkHandler, svtkObject);

  void VoidCallback() { this->VoidEventCounts++; }
  void CallbackWithArguments(svtkObject*, unsigned long event, void*) { this->EventCounts[event]++; }
};
svtkStandardNewMacro(svtkHandler);

int svtkHandler::VoidEventCounts = 0;
std::map<int, int> svtkHandler::EventCounts;

class OtherHandler
{
public:
  static std::map<int, int> EventCounts;
  static int VoidEventCounts;

public:
  void VoidCallback() { this->VoidEventCounts++; }
  void CallbackWithArguments(svtkObject*, unsigned long event, void*) { this->EventCounts[event]++; }
};

int OtherHandler::VoidEventCounts = 0;
std::map<int, int> OtherHandler::EventCounts;

int TestObservers(int, char*[])
{
  unsigned long event0 = 0;
  unsigned long event1 = 0;
  unsigned long event2 = 0;

  svtkObject* volcano = svtkObject::New();

  // First the base test, with a svtkObject pointer
  svtkHandler* handler = svtkHandler::New();

  event0 = volcano->AddObserver(1000, handler, &svtkHandler::VoidCallback);
  event1 = volcano->AddObserver(1001, handler, &svtkHandler::CallbackWithArguments);
  event2 = volcano->AddObserver(1002, handler, &svtkHandler::CallbackWithArguments);

  volcano->InvokeEvent(1000);
  volcano->InvokeEvent(1001);
  volcano->InvokeEvent(1002);

  // let's see if removing an observer works
  volcano->RemoveObserver(event2);
  volcano->InvokeEvent(1000);
  volcano->InvokeEvent(1001);
  volcano->InvokeEvent(1002);

  // now delete the observer, we shouldn't have any dangling pointers.
  handler->Delete();

  volcano->InvokeEvent(1000);
  volcano->InvokeEvent(1001);
  volcano->InvokeEvent(1002);

  // remove an observer after the handler has been deleted, should work.
  volcano->RemoveObserver(event1);
  volcano->InvokeEvent(1000);
  volcano->InvokeEvent(1001);
  volcano->InvokeEvent(1002);

  // remove the final observer
  volcano->RemoveObserver(event0);

  if (svtkHandler::VoidEventCounts == 2 && svtkHandler::EventCounts[1000] == 0 &&
    svtkHandler::EventCounts[1001] == 2 && svtkHandler::EventCounts[1002] == 1)
  {
    cout << "All svtkObject callback counts as expected." << endl;
  }
  else
  {
    cerr << "Mismatched callback counts for SVTK observer." << endl;
    volcano->Delete();
    return 1;
  }

  // ---------------------------------
  // Test again, with smart pointers
  svtkHandler::VoidEventCounts = 0;

  // Make a scope for the smart pointer
  {
    svtkSmartPointer<svtkHandler> handler2 = svtkSmartPointer<svtkHandler>::New();

    event0 = volcano->AddObserver(1003, handler2, &svtkHandler::VoidCallback);
    event1 = volcano->AddObserver(1004, handler2, &svtkHandler::CallbackWithArguments);
    event2 = volcano->AddObserver(1005, handler2, &svtkHandler::CallbackWithArguments);

    volcano->InvokeEvent(1003);
    volcano->InvokeEvent(1004);
    volcano->InvokeEvent(1005);

    // let's see if removing an observer works
    volcano->RemoveObserver(event2);
    volcano->InvokeEvent(1003);
    volcano->InvokeEvent(1004);
    volcano->InvokeEvent(1005);

    // end the scope, which deletes the observer
  }

  // continue invoking, to make sure that
  // no events to to the deleted observer
  volcano->InvokeEvent(1003);
  volcano->InvokeEvent(1004);
  volcano->InvokeEvent(1005);

  // remove an observer after the handler2 has been deleted, should work.
  volcano->RemoveObserver(event1);
  volcano->InvokeEvent(1003);
  volcano->InvokeEvent(1004);
  volcano->InvokeEvent(1005);

  // remove the final observer
  volcano->RemoveObserver(event0);

  if (svtkHandler::VoidEventCounts == 2 && svtkHandler::EventCounts[1003] == 0 &&
    svtkHandler::EventCounts[1004] == 2 && svtkHandler::EventCounts[1005] == 1)
  {
    cout << "All smart pointer callback counts as expected." << endl;
  }
  else
  {
    cerr << "Mismatched callback counts for smart pointer observer." << endl;
    volcano->Delete();
    return 1;
  }

  // ---------------------------------
  // Test yet again, this time with a non-SVTK object
  // (this _can_ leave dangling pointers!!!)

  OtherHandler* handler3 = new OtherHandler();

  event0 = volcano->AddObserver(1006, handler3, &OtherHandler::VoidCallback);
  event1 = volcano->AddObserver(1007, handler3, &OtherHandler::CallbackWithArguments);
  event2 = volcano->AddObserver(1008, handler3, &OtherHandler::CallbackWithArguments);

  volcano->InvokeEvent(1006);
  volcano->InvokeEvent(1007);
  volcano->InvokeEvent(1008);

  // let's see if removing an observer works
  volcano->RemoveObserver(event2);
  volcano->InvokeEvent(1006);
  volcano->InvokeEvent(1007);
  volcano->InvokeEvent(1008);

  // if we delete this non-svtkObject observer, we will
  // have dangling pointers and will see a crash...
  // so let's not do that until the events are removed

  volcano->RemoveObserver(event0);
  volcano->RemoveObserver(event1);
  delete handler3;

  // delete the observed object
  volcano->Delete();

  if (OtherHandler::VoidEventCounts == 2 && OtherHandler::EventCounts[1006] == 0 &&
    OtherHandler::EventCounts[1007] == 2 && OtherHandler::EventCounts[1008] == 1)
  {
    cout << "All non-SVTK observer callback counts as expected." << endl;
    return 0;
  }

  cerr << "Mismatched callback counts for non-SVTK observer." << endl;
  return 1;
}
