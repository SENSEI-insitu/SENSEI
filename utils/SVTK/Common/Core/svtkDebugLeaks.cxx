/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDebugLeaks.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkDebugLeaks.h"

#include "svtkCriticalSection.h"
#include "svtkObjectFactory.h"
#include "svtkWindows.h"

#include <svtksys/SystemInformation.hxx>
#include <svtksys/SystemTools.hxx>

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

static const char* svtkDebugLeaksIgnoreClasses[] = { nullptr };

//----------------------------------------------------------------------------
// return 1 if the class should be ignored
static int svtkDebugLeaksIgnoreClassesCheck(const char* s)
{
  int i = 0;
  while (svtkDebugLeaksIgnoreClasses[i])
  {
    if (strcmp(s, svtkDebugLeaksIgnoreClasses[i]) == 0)
    {
      return 1;
    }
    i++;
  }
  return 0;
}

svtkStandardNewMacro(svtkDebugLeaks);

//----------------------------------------------------------------------------
class svtkDebugLeaksHashTable
{
public:
  svtkDebugLeaksHashTable() {}
  ~svtkDebugLeaksHashTable() {}
  void IncrementCount(const char* name);
  svtkTypeBool DecrementCount(const char* name);
  void PrintTable(std::string& os);
  bool IsEmpty();

private:
  std::unordered_map<const char*, unsigned int> CountMap;
};

//----------------------------------------------------------------------------
void svtkDebugLeaksHashTable::IncrementCount(const char* key)
{
  this->CountMap[key]++;
}

//----------------------------------------------------------------------------
bool svtkDebugLeaksHashTable::IsEmpty()
{
  return this->CountMap.empty();
}

//----------------------------------------------------------------------------
svtkTypeBool svtkDebugLeaksHashTable::DecrementCount(const char* key)
{
  if (this->CountMap.count(key) > 0)
  {
    this->CountMap[key]--;
    if (this->CountMap[key] == 0)
    {
      this->CountMap.erase(key);
    }
    return 1;
  }
  else
  {
    return 0;
  }
}

//----------------------------------------------------------------------------
void svtkDebugLeaksHashTable::PrintTable(std::string& os)
{
  auto iter = this->CountMap.begin();
  while (iter != this->CountMap.end())
  {
    if (iter->second > 0 && !svtkDebugLeaksIgnoreClassesCheck(iter->first))
    {
      char tmp[256];
      snprintf(tmp, 256, "\" has %i %s still around.\n", iter->second,
        (iter->second == 1) ? "instance" : "instances");
      os += "Class \"";
      os += iter->first;
      os += tmp;
    }
    ++iter;
  }
}

//----------------------------------------------------------------------------
class svtkDebugLeaksTraceManager
{
public:
  svtkDebugLeaksTraceManager()
  {
    const char* debugLeaksTraceClasses =
      svtksys::SystemTools::GetEnv("SVTK_DEBUG_LEAKS_TRACE_CLASSES");
    if (debugLeaksTraceClasses)
    {
      std::vector<std::string> classes;
      svtksys::SystemTools::Split(debugLeaksTraceClasses, classes, ',');
      this->ClassesToTrace.insert(classes.begin(), classes.end());
    }
  }
  ~svtkDebugLeaksTraceManager() {}

  void RegisterObject(svtkObjectBase* obj);
  void UnRegisterObject(svtkObjectBase* obj);
  void PrintObjects(std::ostream& os);

private:
  std::set<std::string> ClassesToTrace;
  std::map<svtkObjectBase*, std::string> ObjectTraceMap;
};

//----------------------------------------------------------------------------
#ifdef SVTK_DEBUG_LEAKS
void svtkDebugLeaksTraceManager::RegisterObject(svtkObjectBase* obj)
{
  // Get the current stack trace
  if (this->ClassesToTrace.find(obj->GetClassName()) != this->ClassesToTrace.end())
  {
    const int firstFrame = 5; // skip debug leaks frames and start at the call to New()
    const int wholePath = 1;  // produce the whole path to the file if available
    std::string trace = svtksys::SystemInformation::GetProgramStack(firstFrame, wholePath);
    this->ObjectTraceMap[obj] = trace;
  }
}
#else
void svtkDebugLeaksTraceManager::RegisterObject(svtkObjectBase* svtkNotUsed(obj)) {}
#endif

//----------------------------------------------------------------------------
#ifdef SVTK_DEBUG_LEAKS
void svtkDebugLeaksTraceManager::UnRegisterObject(svtkObjectBase* obj)
{
  this->ObjectTraceMap.erase(obj);
}
#else
void svtkDebugLeaksTraceManager::UnRegisterObject(svtkObjectBase* svtkNotUsed(obj)) {}
#endif

//----------------------------------------------------------------------------
#ifdef SVTK_DEBUG_LEAKS
void svtkDebugLeaksTraceManager::PrintObjects(std::ostream& os)
{
  // Iterate over any remaining object traces and print them
  auto iter = this->ObjectTraceMap.begin();
  while (iter != this->ObjectTraceMap.end())
  {
    os << "Remaining instance of object '" << iter->first->GetClassName();
    os << "' was allocated at:\n";
    os << iter->second << "\n";
    ++iter;
  }
}
#else
void svtkDebugLeaksTraceManager::PrintObjects(std::ostream& svtkNotUsed(os)) {}
#endif

//----------------------------------------------------------------------------
#ifdef SVTK_DEBUG_LEAKS
void svtkDebugLeaks::ConstructClass(svtkObjectBase* object)
{
  svtkDebugLeaks::CriticalSection->Lock();
  svtkDebugLeaks::MemoryTable->IncrementCount(object->GetClassName());
  svtkDebugLeaks::TraceManager->RegisterObject(object);
  svtkDebugLeaks::CriticalSection->Unlock();
}
#else
void svtkDebugLeaks::ConstructClass(svtkObjectBase* svtkNotUsed(object)) {}
#endif

//----------------------------------------------------------------------------
#ifdef SVTK_DEBUG_LEAKS
void svtkDebugLeaks::ConstructClass(const char* className)
{
  svtkDebugLeaks::CriticalSection->Lock();
  svtkDebugLeaks::MemoryTable->IncrementCount(className);
  svtkDebugLeaks::CriticalSection->Unlock();
}
#else
void svtkDebugLeaks::ConstructClass(const char* svtkNotUsed(className)) {}
#endif

//----------------------------------------------------------------------------
#ifdef SVTK_DEBUG_LEAKS
void svtkDebugLeaks::DestructClass(svtkObjectBase* object)
{
  svtkDebugLeaks::CriticalSection->Lock();

  // Ensure the trace manager has not yet been deleted.
  if (svtkDebugLeaks::TraceManager)
  {
    svtkDebugLeaks::TraceManager->UnRegisterObject(object);
  }

  // Due to globals being deleted, this table may already have
  // been deleted.
  if (svtkDebugLeaks::MemoryTable &&
    !svtkDebugLeaks::MemoryTable->DecrementCount(object->GetClassName()))
  {
    svtkGenericWarningMacro("Deleting unknown object: " << object->GetClassName());
  }
  svtkDebugLeaks::CriticalSection->Unlock();
}
#else
void svtkDebugLeaks::DestructClass(svtkObjectBase* svtkNotUsed(object)) {}
#endif

//----------------------------------------------------------------------------
#ifdef SVTK_DEBUG_LEAKS
void svtkDebugLeaks::DestructClass(const char* className)
{
  svtkDebugLeaks::CriticalSection->Lock();

  // Due to globals being deleted, this table may already have
  // been deleted.
  if (svtkDebugLeaks::MemoryTable && !svtkDebugLeaks::MemoryTable->DecrementCount(className))
  {
    svtkGenericWarningMacro("Deleting unknown object: " << className);
  }
  svtkDebugLeaks::CriticalSection->Unlock();
}
#else
void svtkDebugLeaks::DestructClass(const char* svtkNotUsed(className)) {}
#endif

//----------------------------------------------------------------------------
void svtkDebugLeaks::SetDebugLeaksObserver(svtkDebugLeaksObserver* observer)
{
  svtkDebugLeaks::Observer = observer;
}

//----------------------------------------------------------------------------
svtkDebugLeaksObserver* svtkDebugLeaks::GetDebugLeaksObserver()
{
  return svtkDebugLeaks::Observer;
}

//----------------------------------------------------------------------------
void svtkDebugLeaks::ConstructingObject(svtkObjectBase* object)
{
  if (svtkDebugLeaks::Observer)
  {
    svtkDebugLeaks::Observer->ConstructingObject(object);
  }
}

//----------------------------------------------------------------------------
void svtkDebugLeaks::DestructingObject(svtkObjectBase* object)
{
  if (svtkDebugLeaks::Observer)
  {
    svtkDebugLeaks::Observer->DestructingObject(object);
  }
}

//----------------------------------------------------------------------------
int svtkDebugLeaks::PrintCurrentLeaks()
{
#ifdef SVTK_DEBUG_LEAKS
  if (svtkDebugLeaks::MemoryTable->IsEmpty())
  {
    // Log something anyway, so users know svtkDebugLeaks is active/working.
    cerr << "svtkDebugLeaks has found no leaks.\n";
    return 0;
  }

  std::string leaks;
  std::string msg = "svtkDebugLeaks has detected LEAKS!\n";
  svtkDebugLeaks::MemoryTable->PrintTable(leaks);
  cerr << msg;
  cerr << leaks << endl << std::flush;

  svtkDebugLeaks::TraceManager->PrintObjects(std::cerr);

#ifdef _WIN32
  if (getenv("DASHBOARD_TEST_FROM_CTEST") || getenv("DART_TEST_FROM_DART"))
  {
    // Skip dialogs when running on dashboard.
    return 1;
  }
  std::string::size_type myPos = 0;
  int cancel = 0;
  int count = 0;
  while (!cancel && myPos != leaks.npos)
  {
    std::string::size_type newPos = leaks.find('\n', myPos);
    if (newPos != leaks.npos)
    {
      msg += leaks.substr(myPos, newPos - myPos);
      msg += "\n";
      myPos = newPos;
      myPos++;
    }
    else
    {
      myPos = newPos;
    }
    count++;
    if (count == 10)
    {
      count = 0;
      cancel = svtkDebugLeaks::DisplayMessageBox(msg.c_str());
      msg = "";
    }
  }
  if (!cancel && count > 0)
  {
    svtkDebugLeaks::DisplayMessageBox(msg.c_str());
  }
#endif
#endif
  return 1;
}

//----------------------------------------------------------------------------
#ifdef _WIN32
int svtkDebugLeaks::DisplayMessageBox(const char* msg)
{
#ifdef UNICODE
  wchar_t* wmsg = new wchar_t[mbstowcs(nullptr, msg, 32000) + 1];
  mbstowcs(wmsg, msg, 32000);
  int result = (MessageBox(nullptr, wmsg, L"Error", MB_ICONERROR | MB_OKCANCEL) == IDCANCEL);
  delete[] wmsg;
#else
  int result = (MessageBox(nullptr, msg, "Error", MB_ICONERROR | MB_OKCANCEL) == IDCANCEL);
#endif
  return result;
}
#else
int svtkDebugLeaks::DisplayMessageBox(const char*)
{
  return 0;
}
#endif

//----------------------------------------------------------------------------
int svtkDebugLeaks::GetExitError()
{
  return svtkDebugLeaks::ExitError;
}

//----------------------------------------------------------------------------
void svtkDebugLeaks::SetExitError(int flag)
{
  svtkDebugLeaks::ExitError = flag;
}

//----------------------------------------------------------------------------
void svtkDebugLeaks::ClassInitialize()
{
#ifdef SVTK_DEBUG_LEAKS
  // Create the hash table.
  svtkDebugLeaks::MemoryTable = new svtkDebugLeaksHashTable;

  // Create the trace manager.
  svtkDebugLeaks::TraceManager = new svtkDebugLeaksTraceManager;

  // Create the lock for the critical sections.
  svtkDebugLeaks::CriticalSection = new svtkSimpleCriticalSection;

  // Default to error when leaks occur while running tests.
  svtkDebugLeaks::ExitError = 1;
  svtkDebugLeaks::Observer = nullptr;
#else
  svtkDebugLeaks::MemoryTable = nullptr;
  svtkDebugLeaks::CriticalSection = nullptr;
  svtkDebugLeaks::ExitError = 0;
  svtkDebugLeaks::Observer = nullptr;
#endif
}

//----------------------------------------------------------------------------
void svtkDebugLeaks::ClassFinalize()
{
#ifdef SVTK_DEBUG_LEAKS
  // Report leaks.
  int leaked = svtkDebugLeaks::PrintCurrentLeaks();

  // Destroy the hash table.
  delete svtkDebugLeaks::MemoryTable;
  svtkDebugLeaks::MemoryTable = nullptr;

  // Destroy the trace manager.
  delete svtkDebugLeaks::TraceManager;
  svtkDebugLeaks::TraceManager = nullptr;

  // Destroy the lock for the critical sections.
  delete svtkDebugLeaks::CriticalSection;
  svtkDebugLeaks::CriticalSection = nullptr;

  // Exit with error if leaks occurred and error mode is on.
  if (leaked && svtkDebugLeaks::ExitError)
  {
    exit(1);
  }
#endif
}

//----------------------------------------------------------------------------

// Purposely not initialized.  ClassInitialize will handle it.
svtkDebugLeaksHashTable* svtkDebugLeaks::MemoryTable;

svtkDebugLeaksTraceManager* svtkDebugLeaks::TraceManager;

// Purposely not initialized.  ClassInitialize will handle it.
svtkSimpleCriticalSection* svtkDebugLeaks::CriticalSection;

// Purposely not initialized.  ClassInitialize will handle it.
int svtkDebugLeaks::ExitError;

// Purposely not initialized.  ClassInitialize will handle it.
svtkDebugLeaksObserver* svtkDebugLeaks::Observer;
