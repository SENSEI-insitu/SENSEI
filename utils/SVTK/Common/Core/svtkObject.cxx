/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkObject.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkObject.h"

#include "svtkCommand.h"
#include "svtkDebugLeaks.h"
#include "svtkGarbageCollector.h"
#include "svtkObjectFactory.h"
#include "svtkTimeStamp.h"
#include "svtkWeakPointer.h"

#include <algorithm>
#include <vector>

// Initialize static member that controls warning display
static int svtkObjectGlobalWarningDisplay = 1;

//----------------------------------------------------------------------------
// avoid dll boundary problems
#ifdef _WIN32
void* svtkObject::operator new(size_t nSize)
{
  void* p = malloc(nSize);
  return p;
}

//----------------------------------------------------------------------------
void svtkObject::operator delete(void* p)
{
  free(p);
}
#endif

//----------------------------------------------------------------------------
void svtkObject::SetGlobalWarningDisplay(int val)
{
  svtkObjectGlobalWarningDisplay = val;
}

//----------------------------------------------------------------------------
int svtkObject::GetGlobalWarningDisplay()
{
  return svtkObjectGlobalWarningDisplay;
}

//----------------------------------Command/Observer stuff-------------------
// The Command/Observer design pattern is used to invoke and dispatch events.
// The class svtkSubjectHelper keeps a list of observers (which in turn keep
// an instance of svtkCommand) which respond to registered events.
//
class svtkObserver
{
public:
  svtkObserver()
    : Command(nullptr)
    , Event(0)
    , Tag(0)
    , Next(nullptr)
    , Priority(0.0)
  {
  }
  ~svtkObserver();
  void PrintSelf(ostream& os, svtkIndent indent);

  svtkCommand* Command;
  unsigned long Event;
  unsigned long Tag;
  svtkObserver* Next;
  float Priority;
};

void svtkObserver::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "svtkObserver (" << this << ")\n";
  indent = indent.GetNextIndent();
  os << indent << "Event: " << this->Event << "\n";
  os << indent << "EventName: " << svtkCommand::GetStringFromEventId(this->Event) << "\n";
  os << indent << "Command: " << this->Command << "\n";
  os << indent << "Priority: " << this->Priority << "\n";
  os << indent << "Tag: " << this->Tag << "\n";
}

//----------------------------------------------------------------------------
// The svtkSubjectHelper keeps a list of observers and dispatches events to them.
// It also invokes the svtkCommands associated with the observers. Currently
// svtkSubjectHelper is an internal class to svtkObject. However, due to requirements
// from the SVTK widgets it may be necessary to break out the svtkSubjectHelper at
// some point (for reasons of event management, etc.)
class svtkSubjectHelper
{
public:
  svtkSubjectHelper()
    : ListModified(0)
    , Focus1(nullptr)
    , Focus2(nullptr)
    , Start(nullptr)
    , Count(1)
  {
  }
  ~svtkSubjectHelper();

  unsigned long AddObserver(unsigned long event, svtkCommand* cmd, float p);
  void RemoveObserver(unsigned long tag);
  void RemoveObservers(unsigned long event);
  void RemoveObservers(unsigned long event, svtkCommand* cmd);
  void RemoveAllObservers();
  int InvokeEvent(unsigned long event, void* callData, svtkObject* self);
  svtkCommand* GetCommand(unsigned long tag);
  unsigned long GetTag(svtkCommand*);
  svtkTypeBool HasObserver(unsigned long event);
  svtkTypeBool HasObserver(unsigned long event, svtkCommand* cmd);
  void GrabFocus(svtkCommand* c1, svtkCommand* c2)
  {
    this->Focus1 = c1;
    this->Focus2 = c2;
  }
  void ReleaseFocus()
  {
    this->Focus1 = nullptr;
    this->Focus2 = nullptr;
  }
  void PrintSelf(ostream& os, svtkIndent indent);

  int ListModified;

  // This is to support the GrabFocus() methods found in svtkInteractorObserver.
  svtkCommand* Focus1;
  svtkCommand* Focus2;

protected:
  svtkObserver* Start;
  unsigned long Count;
};

// ------------------------------------svtkObject----------------------

svtkObject* svtkObject::New()
{
  svtkObject* ret = new svtkObject;
  ret->InitializeObjectBase();
  return ret;
}

//----------------------------------------------------------------------------
// Create an object with Debug turned off and modified time initialized
// to zero.
svtkObject::svtkObject()
{
  this->Debug = false;
  this->SubjectHelper = nullptr;
  this->Modified(); // Insures modified time > than any other time
  // initial reference count = 1 and reference counting on.
}

//----------------------------------------------------------------------------
svtkObject::~svtkObject()
{
  svtkDebugMacro(<< "Destructing!");

  // warn user if reference counting is on and the object is being referenced
  // by another object
  if (this->ReferenceCount > 0)
  {
    svtkErrorMacro(<< "Trying to delete object with non-zero reference count.");
  }
  delete this->SubjectHelper;
  this->SubjectHelper = nullptr;
}

//----------------------------------------------------------------------------
// Return the modification for this object.
svtkMTimeType svtkObject::GetMTime()
{
  return this->MTime.GetMTime();
}

// Chaining method to print an object's instance variables, as well as
// its superclasses.
void svtkObject::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "Debug: " << (this->Debug ? "On\n" : "Off\n");
  os << indent << "Modified Time: " << this->GetMTime() << "\n";
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Registered Events: ";
  if (this->SubjectHelper)
  {
    os << endl;
    this->SubjectHelper->PrintSelf(os, indent.GetNextIndent());
  }
  else
  {
    os << "(none)\n";
  }
}

//----------------------------------------------------------------------------
// Turn debugging output on.
// The Modified() method is purposely not called since we do not want to affect
// the modification time when enabling debug output.
void svtkObject::DebugOn()
{
  this->Debug = true;
}

//----------------------------------------------------------------------------
// Turn debugging output off.
void svtkObject::DebugOff()
{
  this->Debug = false;
}

//----------------------------------------------------------------------------
// Get the value of the debug flag.
bool svtkObject::GetDebug()
{
  return this->Debug;
}

//----------------------------------------------------------------------------
// Set the value of the debug flag. A true value turns debugging on.
void svtkObject::SetDebug(bool debugFlag)
{
  this->Debug = debugFlag;
}

//----------------------------------------------------------------------------
// This method is called when svtkErrorMacro executes. It allows
// the debugger to break on error.
void svtkObject::BreakOnError() {}

//----------------------------------Command/Observer stuff-------------------
//

//----------------------------------------------------------------------------
svtkObserver::~svtkObserver()
{
  this->Command->UnRegister(nullptr);
}

//----------------------------------------------------------------------------
svtkSubjectHelper::~svtkSubjectHelper()
{
  svtkObserver* elem = this->Start;
  svtkObserver* next;
  while (elem)
  {
    next = elem->Next;
    delete elem;
    elem = next;
  }
  this->Start = nullptr;
  this->Focus1 = nullptr;
  this->Focus2 = nullptr;
}

//----------------------------------------------------------------------------
unsigned long svtkSubjectHelper::AddObserver(unsigned long event, svtkCommand* cmd, float p)
{
  svtkObserver* elem;

  // initialize the new observer element
  elem = new svtkObserver;
  elem->Priority = p;
  elem->Next = nullptr;
  elem->Event = event;
  elem->Command = cmd;
  cmd->Register(nullptr);
  elem->Tag = this->Count;
  this->Count++;

  // now insert into the list
  // if no other elements in the list then this is Start
  if (!this->Start)
  {
    this->Start = elem;
  }
  else
  {
    // insert high priority first
    svtkObserver* prev = nullptr;
    svtkObserver* pos = this->Start;
    while (pos->Priority >= elem->Priority && pos->Next)
    {
      prev = pos;
      pos = pos->Next;
    }
    // pos is Start and elem should not be start
    if (pos->Priority > elem->Priority)
    {
      pos->Next = elem;
    }
    else
    {
      if (prev)
      {
        prev->Next = elem;
      }
      elem->Next = pos;
      // check to see if the new element is the start
      if (pos == this->Start)
      {
        this->Start = elem;
      }
    }
  }
  return elem->Tag;
}

//----------------------------------------------------------------------------
void svtkSubjectHelper::RemoveObserver(unsigned long tag)
{
  svtkObserver* elem;
  svtkObserver* prev;
  svtkObserver* next;

  elem = this->Start;
  prev = nullptr;
  while (elem)
  {
    if (elem->Tag == tag)
    {
      if (prev)
      {
        prev->Next = elem->Next;
        next = prev->Next;
      }
      else
      {
        this->Start = elem->Next;
        next = this->Start;
      }
      delete elem;
      elem = next;
    }
    else
    {
      prev = elem;
      elem = elem->Next;
    }
  }

  this->ListModified = 1;
}

//----------------------------------------------------------------------------
void svtkSubjectHelper::RemoveObservers(unsigned long event)
{
  svtkObserver* elem;
  svtkObserver* prev;
  svtkObserver* next;

  elem = this->Start;
  prev = nullptr;
  while (elem)
  {
    if (elem->Event == event)
    {
      if (prev)
      {
        prev->Next = elem->Next;
        next = prev->Next;
      }
      else
      {
        this->Start = elem->Next;
        next = this->Start;
      }
      delete elem;
      elem = next;
    }
    else
    {
      prev = elem;
      elem = elem->Next;
    }
  }

  this->ListModified = 1;
}

//----------------------------------------------------------------------------
void svtkSubjectHelper::RemoveObservers(unsigned long event, svtkCommand* cmd)
{
  svtkObserver* elem;
  svtkObserver* prev;
  svtkObserver* next;

  elem = this->Start;
  prev = nullptr;
  while (elem)
  {
    if (elem->Event == event && elem->Command == cmd)
    {
      if (prev)
      {
        prev->Next = elem->Next;
        next = prev->Next;
      }
      else
      {
        this->Start = elem->Next;
        next = this->Start;
      }
      delete elem;
      elem = next;
    }
    else
    {
      prev = elem;
      elem = elem->Next;
    }
  }

  this->ListModified = 1;
}

//----------------------------------------------------------------------------
void svtkSubjectHelper::RemoveAllObservers()
{
  svtkObserver* elem = this->Start;
  svtkObserver* next;
  while (elem)
  {
    next = elem->Next;
    delete elem;
    elem = next;
  }
  this->Start = nullptr;
}

//----------------------------------------------------------------------------
svtkTypeBool svtkSubjectHelper::HasObserver(unsigned long event)
{
  svtkObserver* elem = this->Start;
  while (elem)
  {
    if (elem->Event == event || elem->Event == svtkCommand::AnyEvent)
    {
      return 1;
    }
    elem = elem->Next;
  }
  return 0;
}

//----------------------------------------------------------------------------
svtkTypeBool svtkSubjectHelper::HasObserver(unsigned long event, svtkCommand* cmd)
{
  svtkObserver* elem = this->Start;
  while (elem)
  {
    if ((elem->Event == event || elem->Event == svtkCommand::AnyEvent) && elem->Command == cmd)
    {
      return 1;
    }
    elem = elem->Next;
  }
  return 0;
}

//----------------------------------------------------------------------------
int svtkSubjectHelper::InvokeEvent(unsigned long event, void* callData, svtkObject* self)
{
  int focusHandled = 0;

  // When we invoke an event, the observer may add or remove observers.  To make
  // sure that the iteration over the observers goes smoothly, we capture any
  // change to the list with the ListModified ivar.  However, an observer may
  // also do something that causes another event to be invoked in this object.
  // That means that this method will be called recursively, which means that we
  // will obliterate the ListModified flag that the first call is relying on.
  // To get around this, save the previous ListModified value on the stack and
  // then restore it before leaving.
  int saveListModified = this->ListModified;
  this->ListModified = 0;

  // We also need to save what observers we have called on the stack (lest it
  // get overridden in the event invocation).  Also make sure that we do not
  // invoke any new observers that were added during another observer's
  // invocation.
  typedef std::vector<unsigned long> VisitedListType;
  VisitedListType visited;
  svtkObserver* elem = this->Start;
  // If an element with a tag greater than maxTag is found, that means it has
  // been added after InvokeEvent is called (as a side effect of calling an
  // element command. In that case, the element is discarded and not executed.
  const unsigned long maxTag = this->Count;

  // Loop two or three times, giving preference to passive observers
  // and focus holders, if any.
  //
  // 0. Passive observer loop
  //   Loop over all observers and execute those that are passive observers.
  //   These observers should not affect the state of the system in any way,
  //   and should not be allowed to abort the event.
  //
  // 1. Focus loop
  //   If there is a focus holder, loop over all observers and execute
  //   those associated with either focus holder. Set focusHandled to
  //   indicate that a focus holder handled the event.
  //
  // 2. Remainder loop
  //   If no focus holder handled the event already, loop over the
  //   remaining observers. This loop will always get executed when there
  //   is no focus holder.

  // 0. Passive observer loop
  //
  svtkObserver* next;
  while (elem)
  {
    // store the next pointer because elem could disappear due to Command
    next = elem->Next;
    if (elem->Command->GetPassiveObserver() &&
      (elem->Event == event || elem->Event == svtkCommand::AnyEvent) && elem->Tag < maxTag)
    {
      VisitedListType::iterator vIter = std::lower_bound(visited.begin(), visited.end(), elem->Tag);
      if (vIter == visited.end() || *vIter != elem->Tag)
      {
        // Sorted insertion by tag to speed-up future searches at limited
        // insertion cost because it reuses the search iterator already at the
        // correct location
        visited.insert(vIter, elem->Tag);
        svtkCommand* command = elem->Command;
        command->Register(command);
        elem->Command->Execute(self, event, callData);
        command->UnRegister();
      }
    }
    if (this->ListModified)
    {
      svtkGenericWarningMacro(
        << "Passive observer should not call AddObserver or RemoveObserver in callback.");
      elem = this->Start;
      this->ListModified = 0;
    }
    else
    {
      elem = next;
    }
  }

  // 1. Focus loop
  //
  if (this->Focus1 || this->Focus2)
  {
    elem = this->Start;
    while (elem)
    {
      // store the next pointer because elem could disappear due to Command
      next = elem->Next;
      if (((this->Focus1 == elem->Command) || (this->Focus2 == elem->Command)) &&
        (elem->Event == event || elem->Event == svtkCommand::AnyEvent) && elem->Tag < maxTag)
      {
        VisitedListType::iterator vIter =
          std::lower_bound(visited.begin(), visited.end(), elem->Tag);
        if (vIter == visited.end() || *vIter != elem->Tag)
        {
          // Don't execute the remainder loop
          focusHandled = 1;
          // Sorted insertion by tag to speed-up future searches at limited
          // insertion cost because it reuses the search iterator already at the
          // correct location
          visited.insert(vIter, elem->Tag);
          svtkCommand* command = elem->Command;
          command->Register(command);
          command->SetAbortFlag(0);
          elem->Command->Execute(self, event, callData);
          // if the command set the abort flag, then stop firing events
          // and return
          if (command->GetAbortFlag())
          {
            command->UnRegister();
            this->ListModified = saveListModified;
            return 1;
          }
          command->UnRegister();
        }
      }
      if (this->ListModified)
      {
        elem = this->Start;
        this->ListModified = 0;
      }
      else
      {
        elem = next;
      }
    }
  }

  // 2. Remainder loop
  //
  if (!focusHandled)
  {
    elem = this->Start;
    while (elem)
    {
      // store the next pointer because elem could disappear due to Command
      next = elem->Next;
      if ((elem->Event == event || elem->Event == svtkCommand::AnyEvent) && elem->Tag < maxTag)
      {
        VisitedListType::iterator vIter =
          std::lower_bound(visited.begin(), visited.end(), elem->Tag);
        if (vIter == visited.end() || *vIter != elem->Tag)
        {
          // Sorted insertion by tag to speed-up future searches at limited
          // insertion cost because it reuses the search iterator already at the
          // correct location
          visited.insert(vIter, elem->Tag);
          svtkCommand* command = elem->Command;
          command->Register(command);
          command->SetAbortFlag(0);
          elem->Command->Execute(self, event, callData);
          // if the command set the abort flag, then stop firing events
          // and return
          if (command->GetAbortFlag())
          {
            command->UnRegister();
            this->ListModified = saveListModified;
            return 1;
          }
          command->UnRegister();
        }
      }
      if (this->ListModified)
      {
        elem = this->Start;
        this->ListModified = 0;
      }
      else
      {
        elem = next;
      }
    }
  }

  this->ListModified = saveListModified;
  return 0;
}

//----------------------------------------------------------------------------
unsigned long svtkSubjectHelper::GetTag(svtkCommand* cmd)
{
  svtkObserver* elem = this->Start;
  while (elem)
  {
    if (elem->Command == cmd)
    {
      return elem->Tag;
    }
    elem = elem->Next;
  }
  return 0;
}

//----------------------------------------------------------------------------
svtkCommand* svtkSubjectHelper::GetCommand(unsigned long tag)
{
  svtkObserver* elem = this->Start;
  while (elem)
  {
    if (elem->Tag == tag)
    {
      return elem->Command;
    }
    elem = elem->Next;
  }
  return nullptr;
}

//----------------------------------------------------------------------------
void svtkSubjectHelper::PrintSelf(ostream& os, svtkIndent indent)
{
  os << indent << "Registered Observers:\n";
  indent = indent.GetNextIndent();
  svtkObserver* elem = this->Start;
  if (!elem)
  {
    os << indent << "(none)\n";
    return;
  }

  for (; elem; elem = elem->Next)
  {
    elem->PrintSelf(os, indent);
  }
}

//--------------------------------svtkObject observer-----------------------
unsigned long svtkObject::AddObserver(unsigned long event, svtkCommand* cmd, float p)
{
  if (!this->SubjectHelper)
  {
    this->SubjectHelper = new svtkSubjectHelper;
  }
  return this->SubjectHelper->AddObserver(event, cmd, p);
}

//----------------------------------------------------------------------------
unsigned long svtkObject::AddObserver(const char* event, svtkCommand* cmd, float p)
{
  return this->AddObserver(svtkCommand::GetEventIdFromString(event), cmd, p);
}

//----------------------------------------------------------------------------
svtkCommand* svtkObject::GetCommand(unsigned long tag)
{
  if (this->SubjectHelper)
  {
    return this->SubjectHelper->GetCommand(tag);
  }
  return nullptr;
}

//----------------------------------------------------------------------------
void svtkObject::RemoveObserver(unsigned long tag)
{
  if (this->SubjectHelper)
  {
    this->SubjectHelper->RemoveObserver(tag);
  }
}

//----------------------------------------------------------------------------
void svtkObject::RemoveObserver(svtkCommand* c)
{
  if (this->SubjectHelper)
  {
    unsigned long tag = this->SubjectHelper->GetTag(c);
    while (tag)
    {
      this->SubjectHelper->RemoveObserver(tag);
      tag = this->SubjectHelper->GetTag(c);
    }
  }
}

//----------------------------------------------------------------------------
void svtkObject::RemoveObservers(unsigned long event)
{
  if (this->SubjectHelper)
  {
    this->SubjectHelper->RemoveObservers(event);
  }
}

//----------------------------------------------------------------------------
void svtkObject::RemoveObservers(const char* event)
{
  this->RemoveObservers(svtkCommand::GetEventIdFromString(event));
}

//----------------------------------------------------------------------------
void svtkObject::RemoveObservers(unsigned long event, svtkCommand* cmd)
{
  if (this->SubjectHelper)
  {
    this->SubjectHelper->RemoveObservers(event, cmd);
  }
}

//----------------------------------------------------------------------------
void svtkObject::RemoveObservers(const char* event, svtkCommand* cmd)
{
  this->RemoveObservers(svtkCommand::GetEventIdFromString(event), cmd);
}

//----------------------------------------------------------------------------
void svtkObject::RemoveAllObservers()
{
  if (this->SubjectHelper)
  {
    this->SubjectHelper->RemoveAllObservers();
  }
}

//----------------------------------------------------------------------------
int svtkObject::InvokeEvent(unsigned long event, void* callData)
{
  if (this->SubjectHelper)
  {
    return this->SubjectHelper->InvokeEvent(event, callData, this);
  }
  return 0;
}

//----------------------------------------------------------------------------
int svtkObject::InvokeEvent(const char* event, void* callData)
{
  return this->InvokeEvent(svtkCommand::GetEventIdFromString(event), callData);
}

//----------------------------------------------------------------------------
svtkTypeBool svtkObject::HasObserver(unsigned long event)
{
  if (this->SubjectHelper)
  {
    return this->SubjectHelper->HasObserver(event);
  }
  return 0;
}

//----------------------------------------------------------------------------
svtkTypeBool svtkObject::HasObserver(const char* event)
{
  return this->HasObserver(svtkCommand::GetEventIdFromString(event));
}

//----------------------------------------------------------------------------
svtkTypeBool svtkObject::HasObserver(unsigned long event, svtkCommand* cmd)
{
  if (this->SubjectHelper)
  {
    return this->SubjectHelper->HasObserver(event, cmd);
  }
  return 0;
}

//----------------------------------------------------------------------------
svtkTypeBool svtkObject::HasObserver(const char* event, svtkCommand* cmd)
{
  return this->HasObserver(svtkCommand::GetEventIdFromString(event), cmd);
}

//----------------------------------------------------------------------------
void svtkObject::InternalGrabFocus(svtkCommand* mouseEvents, svtkCommand* keypressEvents)
{
  if (this->SubjectHelper)
  {
    this->SubjectHelper->GrabFocus(mouseEvents, keypressEvents);
  }
}

//----------------------------------------------------------------------------
void svtkObject::InternalReleaseFocus()
{
  if (this->SubjectHelper)
  {
    this->SubjectHelper->ReleaseFocus();
  }
}

//----------------------------------------------------------------------------
void svtkObject::Modified()
{
  this->MTime.Modified();
  this->InvokeEvent(svtkCommand::ModifiedEvent, nullptr);
}

//----------------------------------------------------------------------------
void svtkObject::RegisterInternal(svtkObjectBase* o, svtkTypeBool check)
{
  // Print debugging messages.
  if (o)
  {
    svtkDebugMacro(<< "Registered by " << o->GetClassName() << " (" << o
                  << "), ReferenceCount = " << this->ReferenceCount + 1);
  }
  else
  {
    svtkDebugMacro(<< "Registered by nullptr, ReferenceCount = " << this->ReferenceCount + 1);
  }

  // Increment the reference count.
  this->Superclass::RegisterInternal(o, check);
}

//----------------------------------------------------------------------------
void svtkObject::UnRegisterInternal(svtkObjectBase* o, svtkTypeBool check)
{
  // Print debugging messages.
  if (o)
  {
    svtkDebugMacro(<< "UnRegistered by " << o->GetClassName() << " (" << o
                  << "), ReferenceCount = " << (this->ReferenceCount - 1));
  }
  else
  {
    svtkDebugMacro(<< "UnRegistered by nullptr, ReferenceCount = " << (this->ReferenceCount - 1));
  }

  if (this->ReferenceCount == 1)
  {
    // The reference count is 1, so the object is about to be deleted.
    // Invoke the delete event.
    this->InvokeEvent(svtkCommand::DeleteEvent, nullptr);

    // Clean out observers prior to entering destructor
    this->RemoveAllObservers();
  }

  // Decrement the reference count.
  this->Superclass::UnRegisterInternal(o, check);
}

//----------------------------------------------------------------------------
// Internal observer used by svtkObject::AddTemplatedObserver to add a
// svtkClassMemberCallbackBase instance as an observer to an event.
class svtkObjectCommandInternal : public svtkCommand
{
  svtkObject::svtkClassMemberCallbackBase* Callable;

public:
  static svtkObjectCommandInternal* New() { return new svtkObjectCommandInternal(); }

  svtkTypeMacro(svtkObjectCommandInternal, svtkCommand);
  void Execute(svtkObject* caller, unsigned long eventId, void* callData) override
  {
    if (this->Callable)
    {
      this->AbortFlagOff();
      if ((*this->Callable)(caller, eventId, callData))
      {
        this->AbortFlagOn();
      }
    }
  }

  // Takes over the ownership of \c callable.
  void SetCallable(svtkObject::svtkClassMemberCallbackBase* callable)
  {
    delete this->Callable;
    this->Callable = callable;
  }

protected:
  svtkObjectCommandInternal() { this->Callable = nullptr; }
  ~svtkObjectCommandInternal() override { delete this->Callable; }
};

//----------------------------------------------------------------------------
unsigned long svtkObject::AddTemplatedObserver(
  unsigned long event, svtkObject::svtkClassMemberCallbackBase* callable, float priority)
{
  svtkObjectCommandInternal* command = svtkObjectCommandInternal::New();
  // Takes over the ownership of \c callable.
  command->SetCallable(callable);
  unsigned long id = this->AddObserver(event, command, priority);
  command->Delete();
  return id;
}
