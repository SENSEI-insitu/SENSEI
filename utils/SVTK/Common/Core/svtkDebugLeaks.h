/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDebugLeaks.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class svtkDebugLeaks
 * @brief identify memory leaks at program termination
 * svtkDebugLeaks is used to report memory leaks at the exit of the program. It
 * uses svtkObjectBase::InitializeObjectBase() (called via svtkObjectFactory
 * macros) to intercept the construction of all SVTK objects. It uses the
 * UnRegisterInternal method of svtkObjectBase to intercept the destruction of
 * all objects.
 *
 * If not using the svtkObjectFactory macros to implement New(), be sure to call
 * svtkObjectBase::InitializeObjectBase() explicitly on the constructed
 * instance. The rule of thumb is that wherever "new [some svtkObjectBase
 * subclass]" is called, svtkObjectBase::InitializeObjectBase() must be called
 * as well.
 *
 * There are exceptions to this:
 *
 * - svtkCommand subclasses traditionally do not fully participate in
 * svtkDebugLeaks registration, likely because they typically do not use
 * svtkTypeMacro to configure GetClassName. InitializeObjectBase should not be
 * called on svtkCommand subclasses, and all such classes will be automatically
 * registered with svtkDebugLeaks as "svtkCommand or subclass".
 *
 * - svtkInformationKey subclasses are not reference counted. They are allocated
 * statically and registered automatically with a singleton "manager" instance.
 * The manager ensures that all keys are cleaned up before exiting, and
 * registration/deregistration with svtkDebugLeaks is bypassed.
 *
 * A table of object name to number of instances is kept. At the exit of the
 * program if there are still SVTK objects around it will print them out. To
 * enable this class add the flag -DSVTK_DEBUG_LEAKS to the compile line, and
 * rebuild svtkObject and svtkObjectFactory.
 */

#ifndef svtkDebugLeaks_h
#define svtkDebugLeaks_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

#include "svtkDebugLeaksManager.h" // Needed for proper singleton initialization
#include "svtkToolkits.h"          // Needed for SVTK_DEBUG_LEAKS macro setting.

class svtkDebugLeaksHashTable;
class svtkDebugLeaksTraceManager;
class svtkSimpleCriticalSection;
class svtkDebugLeaksObserver;

class SVTKCOMMONCORE_EXPORT svtkDebugLeaks : public svtkObject
{
public:
  static svtkDebugLeaks* New();
  svtkTypeMacro(svtkDebugLeaks, svtkObject);

  /**
   * Call this when creating a class.
   */
  static void ConstructClass(svtkObjectBase* object);

  /**
   * Call this when creating a svtkCommand or subclasses.
   */
  static void ConstructClass(const char* className);

  /**
   * Call this when deleting a class.
   */
  static void DestructClass(svtkObjectBase* object);

  /**
   * Call this when deleting svtkCommand or a subclass.
   */
  static void DestructClass(const char* className);

  /**
   * Print all the values in the table.  Returns non-zero if there
   * were leaks.
   */
  static int PrintCurrentLeaks();

  //@{
  /**
   * Get/Set flag for exiting with an error when leaks are present.
   * Default is on when SVTK_DEBUG_LEAKS is on and off otherwise.
   */
  static int GetExitError();
  static void SetExitError(int);
  //@}

  static void SetDebugLeaksObserver(svtkDebugLeaksObserver* observer);
  static svtkDebugLeaksObserver* GetDebugLeaksObserver();

protected:
  svtkDebugLeaks() {}
  ~svtkDebugLeaks() override {}

  static int DisplayMessageBox(const char*);

  static void ClassInitialize();
  static void ClassFinalize();

  static void ConstructingObject(svtkObjectBase* object);
  static void DestructingObject(svtkObjectBase* object);

  friend class svtkDebugLeaksManager;
  friend class svtkObjectBase;

private:
  static svtkDebugLeaksHashTable* MemoryTable;
  static svtkDebugLeaksTraceManager* TraceManager;
  static svtkSimpleCriticalSection* CriticalSection;
  static svtkDebugLeaksObserver* Observer;
  static int ExitError;

  svtkDebugLeaks(const svtkDebugLeaks&) = delete;
  void operator=(const svtkDebugLeaks&) = delete;
};

// This class defines callbacks for debugging tools. The callbacks are not for general use.
// The objects passed as arguments to the callbacks are in partially constructed or destructed
// state and accessing them may cause undefined behavior.
class SVTKCOMMONCORE_EXPORT svtkDebugLeaksObserver
{
public:
  virtual ~svtkDebugLeaksObserver() {}
  virtual void ConstructingObject(svtkObjectBase*) = 0;
  virtual void DestructingObject(svtkObjectBase*) = 0;
};

#endif // svtkDebugLeaks_h
// SVTK-HeaderTest-Exclude: svtkDebugLeaks.h
