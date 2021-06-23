/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGarbageCollector.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkGarbageCollector.h"

#include "svtkMultiThreader.h"
#include "svtkObjectFactory.h"
#include "svtkSmartPointerBase.h"

#include <queue>
#include <sstream>
#include <stack>
#include <vector>

// Leave the hashing version off for now.
#define SVTK_GARBAGE_COLLECTOR_HASH 0

#if SVTK_GARBAGE_COLLECTOR_HASH
#include <unordered_map>
#include <unordered_set>
#else
#include <map>
#include <set>
#endif

#include <cassert>

svtkStandardNewMacro(svtkGarbageCollector);

#if SVTK_GARBAGE_COLLECTOR_HASH
struct svtkGarbageCollectorHash
{
  size_t operator()(void* p) const { return reinterpret_cast<size_t>(p); }
};
#endif

class svtkGarbageCollectorSingleton;

//----------------------------------------------------------------------------
// The garbage collector singleton.  In order to support delayed
// collection svtkObjectBase::UnRegister passes references to the
// singleton instead of decrementing the reference count.  At some
// point collection occurs and accounts for these references.  This
// MUST be default initialized to zero by the compiler and is
// therefore not initialized here.  The ClassInitialize and
// ClassFinalize methods handle this instance.
static svtkGarbageCollectorSingleton* svtkGarbageCollectorSingletonInstance;

//----------------------------------------------------------------------------
// Global debug setting.  This flag specifies whether a collector
// should print debugging output.  This must be default initialized to
// false by the compiler and is therefore not initialized here.  The
// ClassInitialize and ClassFinalize methods handle it.
static bool svtkGarbageCollectorGlobalDebugFlag;

//----------------------------------------------------------------------------
// The thread identifier of the main thread.  Delayed garbage
// collection is supported only for objects in the main thread.  This
// is initialized when the program loads.  All garbage collection
// calls test whether they are called from this thread.  If not, no
// references are accepted by the singleton.  This must be default
// initialized to zero by the compiler and is therefore not
// initialized here.  The ClassInitialize and ClassFinalize methods
// handle it.
static svtkMultiThreaderIDType svtkGarbageCollectorMainThread;

//----------------------------------------------------------------------------
svtkGarbageCollector::svtkGarbageCollector() = default;

//----------------------------------------------------------------------------
svtkGarbageCollector::~svtkGarbageCollector()
{
  this->SetReferenceCount(0);
}

//----------------------------------------------------------------------------
void svtkGarbageCollector::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
void svtkGarbageCollector::SetGlobalDebugFlag(bool flag)
{
  svtkGarbageCollectorGlobalDebugFlag = flag;
}

//----------------------------------------------------------------------------
bool svtkGarbageCollector::GetGlobalDebugFlag()
{
  return svtkGarbageCollectorGlobalDebugFlag;
}

//----------------------------------------------------------------------------
// Friendship interface listing non-public methods the garbage
// collector can call on svtkObjectBase.
class svtkGarbageCollectorToObjectBaseFriendship
{
public:
  static void ReportReferences(svtkGarbageCollector* self, svtkObjectBase* obj)
  {
    obj->ReportReferences(self);
  }
  static void RegisterBase(svtkObjectBase* obj)
  {
    // Call svtkObjectBase::RegisterInternal directly to make sure the
    // object does not try to report the call back to the garbage
    // collector and no debugging output is shown.
    obj->svtkObjectBase::RegisterInternal(nullptr, 0);
  }
  static void UnRegisterBase(svtkObjectBase* obj)
  {
    // Call svtkObjectBase::UnRegisterInternal directly to make sure
    // the object does not try to report the call back to the garbage
    // collector and no debugging output is shown.
    obj->svtkObjectBase::UnRegisterInternal(nullptr, 0);
  }
  static void Register(svtkObjectBase* obj, svtkObjectBase* from)
  {
    // Call RegisterInternal directly to make sure the object does not
    // try to report the call back to the garbage collector.
    obj->RegisterInternal(from, 0);
  }
  static void UnRegister(svtkObjectBase* obj, svtkObjectBase* from)
  {
    // Call UnRegisterInternal directly to make sure the object does
    // not try to report the call back to the garbage collector.
    obj->UnRegisterInternal(from, 0);
  }
};

//----------------------------------------------------------------------------
// Function to test whether caller is the main thread.
static svtkTypeBool svtkGarbageCollectorIsMainThread()
{
  return svtkMultiThreader::ThreadsEqual(
    svtkGarbageCollectorMainThread, svtkMultiThreader::GetCurrentThreadID());
}

//----------------------------------------------------------------------------
// Singleton to hold discarded references.
class svtkGarbageCollectorSingleton
{
public:
  svtkGarbageCollectorSingleton();
  ~svtkGarbageCollectorSingleton();

  // Internal implementation of svtkGarbageCollector::GiveReference.
  int GiveReference(svtkObjectBase* obj);

  // Internal implementation of svtkGarbageCollector::TakeReference.
  int TakeReference(svtkObjectBase* obj);

  // Called by GiveReference to decide whether to accept a reference.
  svtkTypeBool CheckAccept();

  // Push/Pop deferred collection.
  void DeferredCollectionPush();
  void DeferredCollectionPop();

  // Map from object to number of stored references.
#if SVTK_GARBAGE_COLLECTOR_HASH
  typedef std::unordered_map<svtkObjectBase*, int, svtkGarbageCollectorHash> ReferencesType;
#else
  typedef std::map<svtkObjectBase*, int> ReferencesType;
#endif
  ReferencesType References;

  // The number of references stored in the map.
  int TotalNumberOfReferences;

  // The number of times DeferredCollectionPush has been called not
  // matched by a DeferredCollectionPop.
  int DeferredCollectionCount;
};

//----------------------------------------------------------------------------
// Internal implementation subclass.
class svtkGarbageCollectorImpl : public svtkGarbageCollector
{
public:
  svtkTypeMacro(svtkGarbageCollectorImpl, svtkGarbageCollector);

  svtkGarbageCollectorImpl();
  ~svtkGarbageCollectorImpl() override;

  // Description:
  // Prevent normal svtkObject reference counting behavior.
  void Register(svtkObjectBase*) override;

  // Description:
  // Prevent normal svtkObject reference counting behavior.
  void UnRegister(svtkObjectBase*) override;

  // Perform a collection check.
  void CollectInternal(svtkObjectBase* root);

  // Sun's compiler is broken and does not allow access to protected members from
  // nested class
  // protected:
  //--------------------------------------------------------------------------
  // Internal data structure types.

#if SVTK_GARBAGE_COLLECTOR_HASH
  typedef std::unordered_map<svtkObjectBase*, int, svtkGarbageCollectorHash> ReferencesType;
#else
  typedef std::map<svtkObjectBase*, int> ReferencesType;
#endif
  struct ComponentType;

  struct Entry;
  struct EntryEdge
  {
    Entry* Reference;
    void* Pointer;
    EntryEdge(Entry* r, void* p)
      : Reference(r)
      , Pointer(p)
    {
    }
  };

  // Store garbage collection entries keyed by object.
  struct Entry
  {
    Entry(svtkObjectBase* obj)
      : Object(obj)
      , Root(nullptr)
      , Component(nullptr)
      , VisitOrder(0)
      , Count(0)
      , GarbageCount(0)
      , References()
    {
    }
    ~Entry() { assert(this->GarbageCount == 0); }

    // The object corresponding to this entry.
    svtkObjectBase* Object;

    // The candidate root for the component containing this object.
    Entry* Root;

    // The component to which the object is assigned, if any.
    ComponentType* Component;

    // Mark the order in which object's are visited by Tarjan's algorithm.
    int VisitOrder;

    // The number of references from outside the component not
    // counting the garbage collector references.
    int Count;

    // The number of references held by the garbage collector.
    int GarbageCount;

    // The list of references reported by this entry's object.
    typedef std::vector<EntryEdge> ReferencesType;
    ReferencesType References;
  };

  // Compare entries by object pointer for quick lookup.
#if SVTK_GARBAGE_COLLECTOR_HASH
  struct EntryCompare
  {
    bool operator()(Entry* l, Entry* r) const { return l->Object == r->Object; }
  };
  struct EntryHash
  {
    size_t operator()(Entry* e) const { return e ? reinterpret_cast<size_t>(e->Object) : 0; }
  };
#else
  struct EntryCompare
  {
    std::less<svtkObjectBase*> Compare;
    bool operator()(Entry* l, Entry* r) const { return Compare(l->Object, r->Object); }
  };
#endif

  // Represent a strongly connected component of the reference graph.
  typedef std::vector<Entry*> ComponentBase;
  struct ComponentType : public ComponentBase
  {
    typedef ComponentBase::iterator iterator;
    ComponentType()
      : NetCount(0)
      , Identifier(0)
    {
    }
    ~ComponentType()
    {
      for (iterator i = begin(), iend = end(); i != iend; ++i)
      {
        (*i)->Component = nullptr;
      }
    }

    // The net reference count of the component.
    int NetCount;

    // The component identifier.
    int Identifier;
  };

  //--------------------------------------------------------------------------
  // Internal data objects.

  // The set of objects that have been visited.
#if SVTK_GARBAGE_COLLECTOR_HASH
  typedef std::unordered_set<Entry*, EntryHash, EntryCompare> VisitedType;
#else
  typedef std::set<Entry*, EntryCompare> VisitedType;
#endif
  VisitedType Visited;

  // Count the number of components found to give each an identifier
  // for use in debugging messages.
  int NumberOfComponents;

  // The set of components found that have not yet leaked.
#if SVTK_GARBAGE_COLLECTOR_HASH
  typedef std::unordered_set<ComponentType*, svtkGarbageCollectorHash> ComponentsType;
#else
  typedef std::set<ComponentType*> ComponentsType;
#endif
  ComponentsType ReferencedComponents;

  // Queue leaked components for deletion.
  std::queue<ComponentType*> LeakedComponents;

  // The stack of objects forming the connected components.  This is
  // used in the implementation of Tarjan's algorithm.
  std::stack<Entry*> Stack;

  // The object whose references are currently being traced by
  // Tarjan's algorithm.  Used during the ReportReferences callback.
  Entry* Current;

  // Count for visit order of Tarjan's algorithm.
  int VisitCount;

  // The singleton instance from which to take references when passing
  // references to the entries.
  svtkGarbageCollectorSingleton* Singleton;

  //--------------------------------------------------------------------------
  // Internal implementation methods.

  // Walk the reference graph using Tarjan's algorithm to identify
  // strongly connected components.
  void FindComponents(svtkObjectBase* root);

  // Get the entry for the given object.  This may visit the object.
  Entry* MaybeVisit(svtkObjectBase*);

  // Node visitor for Tarjan's algorithm.
  Entry* VisitTarjan(svtkObjectBase*);

  // Callback from objects to report references.
  void Report(svtkObjectBase* obj, void* ptr);
  void Report(svtkObjectBase* obj, void* ptr, const char* desc) override;

  // Collect the objects of the given leaked component.
  void CollectComponent(ComponentType* c);

  // Print the given component as a debugging message.
  void PrintComponent(ComponentType* c);

  // Subtract references the component holds to itself.
  void SubtractInternalReferences(ComponentType* c);

  // Subtract references the component holds to other components.
  void SubtractExternalReferences(ComponentType* c);

  // Subtract one reference from the given entry.  If the entry's
  // component is left with no references, it is queued as a leaked
  // component.
  void SubtractReference(Entry* e);

  // Transfer references from the garbage collector to the entry for
  // its object.
  void PassReferencesToEntry(Entry* e);

  // Flush all collector references to the object in an entry.
  void FlushEntryReferences(Entry* e);

private:
  svtkGarbageCollectorImpl(const svtkGarbageCollectorImpl&) = delete;
  void operator=(const svtkGarbageCollectorImpl&) = delete;
};

//----------------------------------------------------------------------------
svtkGarbageCollectorImpl::svtkGarbageCollectorImpl()
{
  // Set debugging state.
  this->SetDebug(svtkGarbageCollectorGlobalDebugFlag);

  // Take references from the singleton only in the main thread.
  if (svtkGarbageCollectorIsMainThread())
  {
    this->Singleton = svtkGarbageCollectorSingletonInstance;
  }
  else
  {
    this->Singleton = nullptr;
  }

  // Initialize reference graph walk implementation.
  this->VisitCount = 0;
  this->Current = nullptr;
  this->NumberOfComponents = 0;
}

//----------------------------------------------------------------------------
svtkGarbageCollectorImpl::~svtkGarbageCollectorImpl()
{
  // The collector implementation should have left these empty.
  assert(this->Current == nullptr);
  assert(this->Stack.empty());
  assert(this->LeakedComponents.empty());

  // Clear component list.
  for (ComponentsType::iterator c = this->ReferencedComponents.begin(),
                                cend = this->ReferencedComponents.end();
       c != cend; ++c)
  {
    delete *c;
  }
  this->ReferencedComponents.clear();

  // Clear visited list.
  for (VisitedType::iterator v = this->Visited.begin(), vend = this->Visited.end(); v != vend;)
  {
    // Increment the iterator before deleting because the hash table
    // compare function dereferences the pointer.
    delete *v++;
  }
  this->Visited.clear();

  // Disable debugging to avoid destruction message.
  this->SetDebug(false);
}

//----------------------------------------------------------------------------
void svtkGarbageCollectorImpl::Register(svtkObjectBase*) {}

//----------------------------------------------------------------------------
void svtkGarbageCollectorImpl::UnRegister(svtkObjectBase*) {}

//----------------------------------------------------------------------------
void svtkGarbageCollectorImpl::CollectInternal(svtkObjectBase* root)
{
  // Identify strong components.
  this->FindComponents(root);

  // Delete all the leaked components.
  while (!this->LeakedComponents.empty())
  {
    // Get the next leaked component.
    ComponentType* c = this->LeakedComponents.front();
    this->LeakedComponents.pop();

    // Subtract this component's references to other components.  This
    // may cause others to be queued.
    this->SubtractExternalReferences(c);

    // Collect the members of this component.
    this->CollectComponent(c);

    // We are done with this component.
    delete c;
  }

#ifndef NDEBUG
  // Print remaining referenced components for debugging.
  for (ComponentsType::iterator i = this->ReferencedComponents.begin(),
                                iend = this->ReferencedComponents.end();
       i != iend; ++i)
  {
    this->PrintComponent(*i);
  }
#endif

  // Flush remaining references owned by entries in referenced
  // components.
  for (ComponentsType::iterator c = this->ReferencedComponents.begin(),
                                cend = this->ReferencedComponents.end();
       c != cend; ++c)
  {
    for (ComponentType::iterator j = (*c)->begin(), jend = (*c)->end(); j != jend; ++j)
    {
      this->FlushEntryReferences(*j);
    }
  }
}

//----------------------------------------------------------------------------
void svtkGarbageCollectorImpl::FindComponents(svtkObjectBase* root)
{
  // Walk the references from the given object, if any.
  if (root)
  {
    this->MaybeVisit(root);
  }
}

//----------------------------------------------------------------------------
svtkGarbageCollectorImpl::Entry* svtkGarbageCollectorImpl::MaybeVisit(svtkObjectBase* obj)
{
  // Check for an existing entry.
  assert(obj != nullptr);
  Entry e(obj);
  VisitedType::iterator i = this->Visited.find(&e);
  if (i == this->Visited.end())
  {
    // Visit the object to create the entry.
    return this->VisitTarjan(obj);
  }
  // else Return the existing entry.
  return *i;
}

//----------------------------------------------------------------------------
svtkGarbageCollectorImpl::Entry* svtkGarbageCollectorImpl::VisitTarjan(svtkObjectBase* obj)
{
  // Create an entry for the object.
  Entry* v = new Entry(obj);
  this->Visited.insert(v);

  // Initialize the entry and push it onto the stack of graph nodes.
  v->Root = v;
  v->Component = nullptr;
  v->VisitOrder = ++this->VisitCount;
  this->PassReferencesToEntry(v);
  this->Stack.push(v);

  svtkDebugMacro("Requesting references from "
    << v->Object->GetClassName() << "(" << v->Object << ") with reference count "
    << (v->Object->GetReferenceCount() - v->GarbageCount));

  // Process the references from this node.
  Entry* saveCurrent = this->Current;
  this->Current = v;
  svtkGarbageCollectorToObjectBaseFriendship::ReportReferences(this, v->Object);
  this->Current = saveCurrent;

  // Check if we have found a component.
  if (v->Root == v)
  {
    // Found a new component.
    ComponentType* c = new ComponentType;
    c->Identifier = ++this->NumberOfComponents;
    Entry* w;
    do
    {
      // Get the next member of the component.
      w = this->Stack.top();
      this->Stack.pop();

      // Assign the member to the component.
      w->Component = c;
      w->Root = v;
      c->push_back(w);

      // Include this member's reference count in the component total.
      c->NetCount += w->Count;
    } while (w != v);

    // Save the component.
    this->ReferencedComponents.insert(c);

    // Print the component for debugging.
    this->PrintComponent(c);

    // Remove internal references from the component.
    this->SubtractInternalReferences(c);
  }

  return v;
}

//----------------------------------------------------------------------------
#ifdef NDEBUG
void svtkGarbageCollectorImpl::Report(svtkObjectBase* obj, void* ptr, const char*)
{
  // All calls should be given the pointer.
  assert(ptr != 0);

  // Forward call to the internal implementation.
  if (obj)
  {
    this->Report(obj, ptr);
  }
}
#else
void svtkGarbageCollectorImpl::Report(svtkObjectBase* obj, void* ptr, const char* desc)
{
  // All calls should be given the pointer.
  assert(ptr != nullptr);

  if (obj)
  {
    // Report debugging information if requested.
    if (this->Debug && svtkObject::GetGlobalWarningDisplay())
    {
      svtkObjectBase* current = this->Current->Object;
      std::ostringstream msg;
      msg << "Report: " << current->GetClassName() << "(" << current << ") " << (desc ? desc : "")
          << " -> " << obj->GetClassName() << "(" << obj << ")";
      svtkDebugMacro(<< msg.str().c_str());
    }

    // Forward call to the internal implementation.
    this->Report(obj, ptr);
  }
}
#endif

//----------------------------------------------------------------------------
void svtkGarbageCollectorImpl::Report(svtkObjectBase* obj, void* ptr)
{
  // Get the source and destination of this reference.
  Entry* v = this->Current;
  Entry* w = this->MaybeVisit(obj);

  // If the destination has not yet been assigned to a component,
  // check if it is a better potential root for the current object.
  if (!w->Component)
  {
    if (w->Root->VisitOrder < v->Root->VisitOrder)
    {
      v->Root = w->Root;
    }
  }

  // Save this reference.
  v->References.push_back(EntryEdge(w, ptr));
}

//----------------------------------------------------------------------------
void svtkGarbageCollectorImpl::CollectComponent(ComponentType* c)
{
  ComponentType::iterator e, eend;

  // Print out the component for debugging.
  this->PrintComponent(c);

  // Get an extra reference to all objects in the component so that
  // they are not deleted until all references are removed.
  for (e = c->begin(), eend = c->end(); e != eend; ++e)
  {
    svtkGarbageCollectorToObjectBaseFriendship::Register((*e)->Object, this);
  }

  // Disconnect the reference graph.
  for (e = c->begin(), eend = c->end(); e != eend; ++e)
  {
    // Loop over all references made by this entry's object.
    Entry* entry = *e;
    for (unsigned int i = 0; i < entry->References.size(); ++i)
    {
      // Get a pointer to the object referenced.
      svtkObjectBase* obj = entry->References[i].Reference->Object;

      // Get a pointer to the pointer holding the reference.
      void** ptr = static_cast<void**>(entry->References[i].Pointer);

      // Set the pointer holding the reference to nullptr.  The
      // destructor of the object that reported this reference must
      // deal with this.
      *ptr = nullptr;

      // Remove the reference to the object referenced without
      // recursively collecting.  We already know about the object.
      svtkGarbageCollectorToObjectBaseFriendship::UnRegister(obj, entry->Object);
    }
  }

  // Remove the Entries' references to objects.
  for (e = c->begin(), eend = c->end(); e != eend; ++e)
  {
    this->FlushEntryReferences(*e);
  }

  // Only our extra reference to each object remains.  Delete the
  // objects.
  for (e = c->begin(), eend = c->end(); e != eend; ++e)
  {
    assert((*e)->Object->GetReferenceCount() == 1);
    svtkGarbageCollectorToObjectBaseFriendship::UnRegister((*e)->Object, this);
  }
}

//----------------------------------------------------------------------------
#ifndef NDEBUG
void svtkGarbageCollectorImpl::PrintComponent(ComponentType* c)
{
  if (this->Debug && svtkObject::GetGlobalWarningDisplay())
  {
    std::ostringstream msg;
    msg << "Identified strongly connected component " << c->Identifier
        << " with net reference count " << c->NetCount << ":";
    for (ComponentType::iterator i = c->begin(), iend = c->end(); i != iend; ++i)
    {
      svtkObjectBase* obj = (*i)->Object;
      int count = (*i)->Count;
      msg << "\n  " << obj->GetClassName() << "(" << obj << ")"
          << " with " << count << " external " << ((count == 1) ? "reference" : "references");
    }
    svtkDebugMacro(<< msg.str().c_str());
  }
}
#else
void svtkGarbageCollectorImpl::PrintComponent(ComponentType*) {}
#endif

//----------------------------------------------------------------------------
void svtkGarbageCollectorImpl::SubtractInternalReferences(ComponentType* c)
{
  // Loop over all members of the component.
  for (ComponentType::iterator i = c->begin(), iend = c->end(); i != iend; ++i)
  {
    Entry* v = *i;

    // Loop over all references from this member.
    for (Entry::ReferencesType::iterator r = v->References.begin(), rend = v->References.end();
         r != rend; ++r)
    {
      Entry* w = r->Reference;

      // If this reference points inside the component, subtract it.
      if (v->Component == w->Component)
      {
        this->SubtractReference(w);
      }
    }
  }
}

//----------------------------------------------------------------------------
void svtkGarbageCollectorImpl::SubtractExternalReferences(ComponentType* c)
{
  // Loop over all members of the component.
  for (ComponentType::iterator i = c->begin(), iend = c->end(); i != iend; ++i)
  {
    Entry* v = *i;

    // Loop over all references from this member.
    for (Entry::ReferencesType::iterator r = v->References.begin(), rend = v->References.end();
         r != rend; ++r)
    {
      Entry* w = r->Reference;

      // If this reference points outside the component, subtract it.
      if (v->Component != w->Component)
      {
        this->SubtractReference(w);
      }
    }
  }
}

//----------------------------------------------------------------------------
void svtkGarbageCollectorImpl::SubtractReference(Entry* e)
{
  // The component should not be leaked before we get here.
  assert(e->Component != nullptr);
  assert(e->Component->NetCount > 0);

  svtkDebugMacro("Subtracting reference to object "
    << e->Object->GetClassName() << "(" << e->Object << ")"
    << " in component " << e->Component->Identifier << ".");

  // Decrement the entry's reference count.
  --e->Count;

  // If the component's net count is now zero, move it to the queue of
  // leaked component.
  if (--e->Component->NetCount == 0)
  {
    this->ReferencedComponents.erase(e->Component);
    this->LeakedComponents.push(e->Component);
    svtkDebugMacro("Component " << e->Component->Identifier << " is leaked.");
  }
}

//----------------------------------------------------------------------------
void svtkGarbageCollectorImpl::PassReferencesToEntry(Entry* e)
{
  // Get the number of references the collector holds.
  e->GarbageCount = 0;
  if (this->Singleton)
  {
    ReferencesType::iterator i = this->Singleton->References.find(e->Object);
    if (i != this->Singleton->References.end())
    {
      // Pass these references from the singleton to the entry.
      e->GarbageCount = i->second;
      this->Singleton->References.erase(i);
      this->Singleton->TotalNumberOfReferences -= e->GarbageCount;
    }
  }

  // Make sure the entry has at least one reference to the object.
  // This ensures the object in components of size 1 is not deleted
  // until we delete the component.
  if (e->GarbageCount == 0)
  {
    svtkGarbageCollectorToObjectBaseFriendship::RegisterBase(e->Object);
    ++e->GarbageCount;
  }

  // Subtract the garbage count from the object's reference count.
  e->Count = e->Object->GetReferenceCount() - e->GarbageCount;
}

//----------------------------------------------------------------------------
void svtkGarbageCollectorImpl::FlushEntryReferences(Entry* e)
{
  while (e->GarbageCount > 0)
  {
    svtkGarbageCollectorToObjectBaseFriendship::UnRegisterBase(e->Object);
    --e->GarbageCount;
  }
}

//----------------------------------------------------------------------------
void svtkGarbageCollector::ClassInitialize()
{
  // Set default debugging state.
  svtkGarbageCollectorGlobalDebugFlag = false;

  // Record the id of the main thread.
  svtkGarbageCollectorMainThread = svtkMultiThreader::GetCurrentThreadID();

  // Allocate the singleton used for delayed collection in the main
  // thread.
  svtkGarbageCollectorSingletonInstance = new svtkGarbageCollectorSingleton;
}

//----------------------------------------------------------------------------
void svtkGarbageCollector::ClassFinalize()
{
  // We are done with the singleton.  Delete it and reset the pointer.
  // Other singletons may still cause garbage collection of SVTK
  // objects, they just will not have the option of deferred
  // collection.  In order to get it they need only to include
  // svtkGarbageCollectorManager.h so that this singleton stays around
  // longer.
  delete svtkGarbageCollectorSingletonInstance;
  svtkGarbageCollectorSingletonInstance = nullptr;
}

//----------------------------------------------------------------------------
void svtkGarbageCollector::Report(svtkObjectBase*, void*, const char*)
{
  svtkErrorMacro("svtkGarbageCollector::Report should be overridden.");
}

//----------------------------------------------------------------------------
void svtkGarbageCollector::Collect()
{
  // This must be called only from the main thread.
  assert(svtkGarbageCollectorIsMainThread());

  // Keep collecting until no deferred checks exist.
  while (svtkGarbageCollectorSingletonInstance &&
    svtkGarbageCollectorSingletonInstance->TotalNumberOfReferences > 0)
  {
    // Collect starting from one deferred object at a time.  Each
    // check will remove at least the starting object and possibly
    // other objects from the singleton's references.
    svtkObjectBase* root = svtkGarbageCollectorSingletonInstance->References.begin()->first;
    svtkGarbageCollector::Collect(root);
  }
}

//----------------------------------------------------------------------------
void svtkGarbageCollector::Collect(svtkObjectBase* root)
{
  // Create a collector instance.
  svtkGarbageCollectorImpl collector;

  svtkDebugWithObjectMacro((&collector), "Starting collection check.");

  // Collect leaked objects.
  collector.CollectInternal(root);

  svtkDebugWithObjectMacro((&collector), "Finished collection check.");
}

//----------------------------------------------------------------------------
void svtkGarbageCollector::DeferredCollectionPush()
{
  // This must be called only from the main thread.
  assert(svtkGarbageCollectorIsMainThread());

  // Forward the call to the singleton.
  if (svtkGarbageCollectorSingletonInstance)
  {
    svtkGarbageCollectorSingletonInstance->DeferredCollectionPush();
  }
}

//----------------------------------------------------------------------------
void svtkGarbageCollector::DeferredCollectionPop()
{
  // This must be called only from the main thread.
  assert(svtkGarbageCollectorIsMainThread());

  // Forward the call to the singleton.
  if (svtkGarbageCollectorSingletonInstance)
  {
    svtkGarbageCollectorSingletonInstance->DeferredCollectionPop();
  }
}

//----------------------------------------------------------------------------
int svtkGarbageCollector::GiveReference(svtkObjectBase* obj)
{
  // We must have an object.
  assert(obj != nullptr);

  // See if the singleton will accept a reference.
  if (svtkGarbageCollectorIsMainThread() && svtkGarbageCollectorSingletonInstance)
  {
    return svtkGarbageCollectorSingletonInstance->GiveReference(obj);
  }

  // Could not accept the reference.
  return 0;
}

//----------------------------------------------------------------------------
int svtkGarbageCollector::TakeReference(svtkObjectBase* obj)
{
  // We must have an object.
  assert(obj != nullptr);

  // See if the singleton has a reference.
  if (svtkGarbageCollectorIsMainThread() && svtkGarbageCollectorSingletonInstance)
  {
    return svtkGarbageCollectorSingletonInstance->TakeReference(obj);
  }

  // No reference is available.
  return 0;
}

//----------------------------------------------------------------------------
svtkGarbageCollectorSingleton::svtkGarbageCollectorSingleton()
{
  this->TotalNumberOfReferences = 0;
  this->DeferredCollectionCount = 0;
}

//----------------------------------------------------------------------------
svtkGarbageCollectorSingleton::~svtkGarbageCollectorSingleton()
{
  // There should be no deferred collections left.
  assert(this->TotalNumberOfReferences == 0);
}

//----------------------------------------------------------------------------
int svtkGarbageCollectorSingleton::GiveReference(svtkObjectBase* obj)
{
  // Check if we can store a reference to the object in the map.
  if (this->CheckAccept())
  {
    // Create a reference to the object.
    ReferencesType::iterator i = this->References.find(obj);
    if (i == this->References.end())
    {
      // This is a new object.  Create a map entry for it.
      this->References.insert(ReferencesType::value_type(obj, 1));
    }
    else
    {
      ++i->second;
    }
    ++this->TotalNumberOfReferences;
    return 1;
  }

  // We did not accept the reference.
  return 0;
}

//----------------------------------------------------------------------------
int svtkGarbageCollectorSingleton::TakeReference(svtkObjectBase* obj)
{
  // If we have a reference to the object hand it back to the caller.
  ReferencesType::iterator i = this->References.find(obj);
  if (i != this->References.end())
  {
    // Remove our reference to the object.
    --this->TotalNumberOfReferences;
    if (--i->second == 0)
    {
      // If we have no more references to the object, remove its map
      // entry.
      this->References.erase(i);
    }
    return 1;
  }

  // We do not have a reference to the object.
  return 0;
}

//----------------------------------------------------------------------------
svtkTypeBool svtkGarbageCollectorSingleton::CheckAccept()
{
  // Accept the reference only if deferred collection is enabled.  It
  // is tempting to put a check against TotalNumberOfReferences here
  // to collect every so many deferred calls, but this will NOT work.
  // Some objects call UnRegister on other objects during
  // construction.  We do not want to perform deferred collection
  // while an object is under construction because the reference walk
  // might call ReportReferences on a partially constructed object!
  return this->DeferredCollectionCount > 0;
}

//----------------------------------------------------------------------------
void svtkGarbageCollectorSingleton::DeferredCollectionPush()
{
  if (++this->DeferredCollectionCount <= 0)
  {
    // Deferred collection is disabled.  Collect immediately.
    svtkGarbageCollector::Collect();
  }
}

//----------------------------------------------------------------------------
void svtkGarbageCollectorSingleton::DeferredCollectionPop()
{
  if (--this->DeferredCollectionCount <= 0)
  {
    // Deferred collection is disabled.  Collect immediately.
    svtkGarbageCollector::Collect();
  }
}

//----------------------------------------------------------------------------
void svtkGarbageCollectorReportInternal(
  svtkGarbageCollector* collector, svtkObjectBase* obj, void* ptr, const char* desc)
{
  collector->Report(obj, ptr, desc);
}

//----------------------------------------------------------------------------
void svtkGarbageCollectorReport(
  svtkGarbageCollector* collector, svtkSmartPointerBase& ptr, const char* desc)
{
  ptr.Report(collector, desc);
}
