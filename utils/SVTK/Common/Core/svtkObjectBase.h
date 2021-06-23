/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkObjectBase.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkObjectBase
 * @brief   abstract base class for most SVTK objects
 *
 * svtkObjectBase is the base class for all reference counted classes
 * in the SVTK. These classes include svtkCommand classes, svtkInformationKey
 * classes, and svtkObject classes.
 *
 * svtkObjectBase performs reference counting: objects that are
 * reference counted exist as long as another object uses them. Once
 * the last reference to a reference counted object is removed, the
 * object will spontaneously destruct.
 *
 * Constructor and destructor of the subclasses of svtkObjectBase
 * should be protected, so that only New() and UnRegister() actually
 * call them. Debug leaks can be used to see if there are any objects
 * left with nonzero reference count.
 *
 * @warning
 * Note: Objects of subclasses of svtkObjectBase should always be
 * created with the New() method and deleted with the Delete()
 * method. They cannot be allocated off the stack (i.e., automatic
 * objects) because the constructor is a protected method.
 *
 * @sa
 * svtkObject svtkCommand svtkInformationKey
 */

#ifndef svtkObjectBase_h
#define svtkObjectBase_h

// Semantics around svtkDebugLeaks usage has changed. Now just call
// svtkObjectBase::InitializeObjectBase() after creating an object with New().
// The object factory methods take care of this automatically.
#define SVTK_HAS_INITIALIZE_OBJECT_BASE

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkIndent.h"
#include "svtkSystemIncludes.h"
#include "svtkType.h"

#include <atomic> // For std::atomic

class svtkGarbageCollector;
class svtkGarbageCollectorToObjectBaseFriendship;
class svtkWeakPointerBase;
class svtkWeakPointerBaseToObjectBaseFriendship;

class SVTKCOMMONCORE_EXPORT svtkObjectBase
{
  /**
   * Return the class name as a string. This method is overridden
   * in all subclasses of svtkObjectBase with the svtkTypeMacro found
   * in svtkSetGet.h.
   */
  virtual const char* GetClassNameInternal() const { return "svtkObjectBase"; }

public:
#ifdef SVTK_WORKAROUND_WINDOWS_MANGLE
// Avoid windows name mangling.
#define GetClassNameA GetClassName
#define GetClassNameW GetClassName
#endif

  /**
   * Return the class name as a string.
   */
  const char* GetClassName() const;

#ifdef SVTK_WORKAROUND_WINDOWS_MANGLE
#undef GetClassNameW
#undef GetClassNameA

  // Define possible mangled names.
  const char* GetClassNameA() const;
  const char* GetClassNameW() const;

#endif

  /**
   * Return 1 if this class type is the same type of (or a subclass of)
   * the named class. Returns 0 otherwise. This method works in
   * combination with svtkTypeMacro found in svtkSetGet.h.
   */
  static svtkTypeBool IsTypeOf(const char* name);

  /**
   * Return 1 if this class is the same type of (or a subclass of)
   * the named class. Returns 0 otherwise. This method works in
   * combination with svtkTypeMacro found in svtkSetGet.h.
   */
  virtual svtkTypeBool IsA(const char* name);

  /**
   * Given a the name of a base class of this class type, return the distance
   * of inheritance between this class type and the named class (how many
   * generations of inheritance are there between this class and the named
   * class). If the named class is not in this class's inheritance tree, return
   * a negative value. Valid responses will always be nonnegative. This method
   * works in combination with svtkTypeMacro found in svtkSetGet.h.
   */
  static svtkIdType GetNumberOfGenerationsFromBaseType(const char* name);

  /**
   * Given a the name of a base class of this class type, return the distance
   * of inheritance between this class type and the named class (how many
   * generations of inheritance are there between this class and the named
   * class). If the named class is not in this class's inheritance tree, return
   * a negative value. Valid responses will always be nonnegative. This method
   * works in combination with svtkTypeMacro found in svtkSetGet.h.
   */
  virtual svtkIdType GetNumberOfGenerationsFromBase(const char* name);

  /**
   * Delete a SVTK object.  This method should always be used to delete
   * an object when the New() method was used to create it. Using the
   * C++ delete method will not work with reference counting.
   */
  virtual void Delete();

  /**
   * Delete a reference to this object.  This version will not invoke
   * garbage collection and can potentially leak the object if it is
   * part of a reference loop.  Use this method only when it is known
   * that the object has another reference and would not be collected
   * if a full garbage collection check were done.
   */
  virtual void FastDelete();

  /**
   * Create an object with Debug turned off, modified time initialized
   * to zero, and reference counting on.
   */
  static svtkObjectBase* New()
  {
    svtkObjectBase* o = new svtkObjectBase;
    o->InitializeObjectBase();
    return o;
  }

  // Called by implementations of svtkObject::New(). Centralized location for
  // svtkDebugLeaks registration:
  void InitializeObjectBase();

#ifdef _WIN32
  // avoid dll boundary problems
  void* operator new(size_t tSize);
  void operator delete(void* p);
#endif

  /**
   * Print an object to an ostream. This is the method to call
   * when you wish to see print the internal state of an object.
   */
  void Print(ostream& os);

  //@{
  /**
   * Methods invoked by print to print information about the object
   * including superclasses. Typically not called by the user (use
   * Print() instead) but used in the hierarchical print process to
   * combine the output of several classes.
   */
  virtual void PrintSelf(ostream& os, svtkIndent indent);
  virtual void PrintHeader(ostream& os, svtkIndent indent);
  virtual void PrintTrailer(ostream& os, svtkIndent indent);
  //@}

  /**
   * Increase the reference count (mark as used by another object).
   */
  virtual void Register(svtkObjectBase* o);

  /**
   * Decrease the reference count (release by another object). This
   * has the same effect as invoking Delete() (i.e., it reduces the
   * reference count by 1).
   */
  virtual void UnRegister(svtkObjectBase* o);

  /**
   * Return the current reference count of this object.
   */
  int GetReferenceCount() { return this->ReferenceCount; }

  /**
   * Sets the reference count. (This is very dangerous, use with care.)
   */
  void SetReferenceCount(int);

/**
 * Legacy.  Do not call.
 */
#ifndef SVTK_LEGACY_REMOVE
  void PrintRevisions(ostream&) {}
#endif

protected:
  svtkObjectBase();
  virtual ~svtkObjectBase();

#ifndef SVTK_LEGACY_REMOVE
  virtual void CollectRevisions(ostream&) {} // Legacy; do not use!
#endif

  std::atomic<int32_t> ReferenceCount;
  svtkWeakPointerBase** WeakPointers;

  // Internal Register/UnRegister implementation that accounts for
  // possible garbage collection participation.  The second argument
  // indicates whether to participate in garbage collection.
  virtual void RegisterInternal(svtkObjectBase*, svtkTypeBool check);
  virtual void UnRegisterInternal(svtkObjectBase*, svtkTypeBool check);

  // See svtkGarbageCollector.h:
  virtual void ReportReferences(svtkGarbageCollector*);

private:
  friend SVTKCOMMONCORE_EXPORT ostream& operator<<(ostream& os, svtkObjectBase& o);
  friend class svtkGarbageCollectorToObjectBaseFriendship;
  friend class svtkWeakPointerBaseToObjectBaseFriendship;

protected:
  svtkObjectBase(const svtkObjectBase&) {}
  void operator=(const svtkObjectBase&) {}
};

#endif

// SVTK-HeaderTest-Exclude: svtkObjectBase.h
