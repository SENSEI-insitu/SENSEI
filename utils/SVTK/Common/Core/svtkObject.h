/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkObject.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkObject
 * @brief   abstract base class for most SVTK objects
 *
 * svtkObject is the base class for most objects in the visualization
 * toolkit. svtkObject provides methods for tracking modification time,
 * debugging, printing, and event callbacks. Most objects created
 * within the SVTK framework should be a subclass of svtkObject or one
 * of its children.  The few exceptions tend to be very small helper
 * classes that usually never get instantiated or situations where
 * multiple inheritance gets in the way.  svtkObject also performs
 * reference counting: objects that are reference counted exist as
 * long as another object uses them. Once the last reference to a
 * reference counted object is removed, the object will spontaneously
 * destruct.
 *
 * @warning
 * Note: in SVTK objects should always be created with the New() method
 * and deleted with the Delete() method. SVTK objects cannot be
 * allocated off the stack (i.e., automatic objects) because the
 * constructor is a protected method.
 *
 * @sa
 * svtkCommand svtkTimeStamp
 */

#ifndef svtkObject_h
#define svtkObject_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObjectBase.h"
#include "svtkSetGet.h"
#include "svtkTimeStamp.h"
#include "svtkWeakPointerBase.h" // needed for svtkWeakPointer

class svtkSubjectHelper;
class svtkCommand;

class SVTKCOMMONCORE_EXPORT svtkObject : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkObject, svtkObjectBase);

  /**
   * Create an object with Debug turned off, modified time initialized
   * to zero, and reference counting on.
   */
  static svtkObject* New();

#ifdef _WIN32
  // avoid dll boundary problems
  void* operator new(size_t tSize);
  void operator delete(void* p);
#endif

  /**
   * Turn debugging output on.
   */
  virtual void DebugOn();

  /**
   * Turn debugging output off.
   */
  virtual void DebugOff();

  /**
   * Get the value of the debug flag.
   */
  bool GetDebug();

  /**
   * Set the value of the debug flag. A true value turns debugging on.
   */
  void SetDebug(bool debugFlag);

  /**
   * This method is called when svtkErrorMacro executes. It allows
   * the debugger to break on error.
   */
  static void BreakOnError();

  /**
   * Update the modification time for this object. Many filters rely on
   * the modification time to determine if they need to recompute their
   * data. The modification time is a unique monotonically increasing
   * unsigned long integer.
   */
  virtual void Modified();

  /**
   * Return this object's modified time.
   */
  virtual svtkMTimeType GetMTime();

  /**
   * Methods invoked by print to print information about the object
   * including superclasses. Typically not called by the user (use
   * Print() instead) but used in the hierarchical print process to
   * combine the output of several classes.
   */
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * This is a global flag that controls whether any debug, warning
   * or error messages are displayed.
   */
  static void SetGlobalWarningDisplay(int val);
  static void GlobalWarningDisplayOn() { svtkObject::SetGlobalWarningDisplay(1); }
  static void GlobalWarningDisplayOff() { svtkObject::SetGlobalWarningDisplay(0); }
  static int GetGlobalWarningDisplay();
  //@}

  //@{
  /**
   * Allow people to add/remove/invoke observers (callbacks) to any SVTK
   * object.  This is an implementation of the subject/observer design
   * pattern. An observer is added by specifying an event to respond to
   * and a svtkCommand to execute. It returns an unsigned long tag which
   * can be used later to remove the event or retrieve the command.
   * When events are invoked, the observers are called in the order they
   * were added. If a priority value is specified, then the higher
   * priority commands are called first. A command may set an abort
   * flag to stop processing of the event. (See svtkCommand.h for more
   * information.)
   */
  unsigned long AddObserver(unsigned long event, svtkCommand*, float priority = 0.0f);
  unsigned long AddObserver(const char* event, svtkCommand*, float priority = 0.0f);
  svtkCommand* GetCommand(unsigned long tag);
  void RemoveObserver(svtkCommand*);
  void RemoveObservers(unsigned long event, svtkCommand*);
  void RemoveObservers(const char* event, svtkCommand*);
  svtkTypeBool HasObserver(unsigned long event, svtkCommand*);
  svtkTypeBool HasObserver(const char* event, svtkCommand*);
  //@}

  void RemoveObserver(unsigned long tag);
  void RemoveObservers(unsigned long event);
  void RemoveObservers(const char* event);
  void RemoveAllObservers(); // remove every last one of them
  svtkTypeBool HasObserver(unsigned long event);
  svtkTypeBool HasObserver(const char* event);

  //@{
  /**
   * Overloads to AddObserver that allow developers to add class member
   * functions as callbacks for events.  The callback function can
   * be one of these two types:
   * \code
   * void foo(void);\n
   * void foo(svtkObject*, unsigned long, void*);
   * \endcode
   * If the callback is a member of a svtkObjectBase-derived object,
   * then the callback will automatically be disabled if the object
   * destructs (but the observer will not automatically be removed).
   * If the callback is a member of any other type of object, then
   * the observer must be removed before the object destructs or else
   * its dead pointer will be used the next time the event occurs.
   * Typical usage of these functions is as follows:
   * \code
   * SomeClassOfMine* observer = SomeClassOfMine::New();\n
   * to_observe->AddObserver(event, observer, \&SomeClassOfMine::SomeMethod);
   * \endcode
   * Note that this does not affect the reference count of a
   * svtkObjectBase-derived \c observer, which can be safely deleted
   * with the observer still in place. For non-svtkObjectBase observers,
   * the observer should never be deleted before it is removed.
   * Return value is a tag that can be used to remove the observer.
   */
  template <class U, class T>
  unsigned long AddObserver(
    unsigned long event, U observer, void (T::*callback)(), float priority = 0.0f)
  {
    svtkClassMemberCallback<T>* callable = new svtkClassMemberCallback<T>(observer, callback);
    // callable is deleted when the observer is cleaned up (look at
    // svtkObjectCommandInternal)
    return this->AddTemplatedObserver(event, callable, priority);
  }
  template <class U, class T>
  unsigned long AddObserver(unsigned long event, U observer,
    void (T::*callback)(svtkObject*, unsigned long, void*), float priority = 0.0f)
  {
    svtkClassMemberCallback<T>* callable = new svtkClassMemberCallback<T>(observer, callback);
    // callable is deleted when the observer is cleaned up (look at
    // svtkObjectCommandInternal)
    return this->AddTemplatedObserver(event, callable, priority);
  }
  //@}

  //@{
  /**
   * Allow user to set the AbortFlagOn() with the return value of the callback
   * method.
   */
  template <class U, class T>
  unsigned long AddObserver(unsigned long event, U observer,
    bool (T::*callback)(svtkObject*, unsigned long, void*), float priority = 0.0f)
  {
    svtkClassMemberCallback<T>* callable = new svtkClassMemberCallback<T>(observer, callback);
    // callable is deleted when the observer is cleaned up (look at
    // svtkObjectCommandInternal)
    return this->AddTemplatedObserver(event, callable, priority);
  }
  //@}

  //@{
  /**
   * This method invokes an event and return whether the event was
   * aborted or not. If the event was aborted, the return value is 1,
   * otherwise it is 0.
   */
  int InvokeEvent(unsigned long event, void* callData);
  int InvokeEvent(const char* event, void* callData);
  //@}

  int InvokeEvent(unsigned long event) { return this->InvokeEvent(event, nullptr); }
  int InvokeEvent(const char* event) { return this->InvokeEvent(event, nullptr); }

protected:
  svtkObject();
  ~svtkObject() override;

  // See svtkObjectBase.h.
  void RegisterInternal(svtkObjectBase*, svtkTypeBool check) override;
  void UnRegisterInternal(svtkObjectBase*, svtkTypeBool check) override;

  bool Debug;                      // Enable debug messages
  svtkTimeStamp MTime;              // Keep track of modification time
  svtkSubjectHelper* SubjectHelper; // List of observers on this object

  //@{
  /**
   * These methods allow a command to exclusively grab all events. (This
   * method is typically used by widgets to grab events once an event
   * sequence begins.)  These methods are provided in support of the
   * public methods found in the class svtkInteractorObserver. Note that
   * these methods are designed to support svtkInteractorObservers since
   * they use two separate svtkCommands to watch for mouse and keypress events.
   */
  void InternalGrabFocus(svtkCommand* mouseEvents, svtkCommand* keypressEvents = nullptr);
  void InternalReleaseFocus();
  //@}

private:
  svtkObject(const svtkObject&) = delete;
  void operator=(const svtkObject&) = delete;

  /**
   * Following classes (svtkClassMemberCallbackBase,
   * svtkClassMemberCallback, and svtkClassMemberHanderPointer)
   * along with svtkObjectCommandInternal are for supporting
   * templated AddObserver() overloads that allow developers
   * to add event callbacks that are class member functions.
   */
  class svtkClassMemberCallbackBase
  {
  public:
    //@{
    /**
     * Called when the event is invoked
     */
    virtual bool operator()(svtkObject*, unsigned long, void*) = 0;
    virtual ~svtkClassMemberCallbackBase() {}
    //@}
  };

  //@{
  /**
   * This is a weak pointer for svtkObjectBase and a regular
   * void pointer for everything else
   */
  template <class T>
  class svtkClassMemberHandlerPointer
  {
  public:
    void operator=(svtkObjectBase* o)
    {
      // The cast is needed in case "o" has multi-inheritance,
      // to offset the pointer to get the svtkObjectBase.
      if ((this->VoidPointer = dynamic_cast<T*>(o)) == nullptr)
      {
        // fallback to just using its svtkObjectBase as-is.
        this->VoidPointer = o;
      }
      this->WeakPointer = o;
      this->UseWeakPointer = true;
    }
    void operator=(void* o)
    {
      this->VoidPointer = o;
      this->WeakPointer = nullptr;
      this->UseWeakPointer = false;
    }
    T* GetPointer()
    {
      if (this->UseWeakPointer && !this->WeakPointer.GetPointer())
      {
        return nullptr;
      }
      return static_cast<T*>(this->VoidPointer);
    }

  private:
    svtkWeakPointerBase WeakPointer;
    void* VoidPointer;
    bool UseWeakPointer;
  };
  //@}

  //@{
  /**
   * Templated member callback.
   */
  template <class T>
  class svtkClassMemberCallback : public svtkClassMemberCallbackBase
  {
    svtkClassMemberHandlerPointer<T> Handler;
    void (T::*Method1)();
    void (T::*Method2)(svtkObject*, unsigned long, void*);
    bool (T::*Method3)(svtkObject*, unsigned long, void*);

  public:
    svtkClassMemberCallback(T* handler, void (T::*method)())
    {
      this->Handler = handler;
      this->Method1 = method;
      this->Method2 = nullptr;
      this->Method3 = nullptr;
    }

    svtkClassMemberCallback(T* handler, void (T::*method)(svtkObject*, unsigned long, void*))
    {
      this->Handler = handler;
      this->Method1 = nullptr;
      this->Method2 = method;
      this->Method3 = nullptr;
    }

    svtkClassMemberCallback(T* handler, bool (T::*method)(svtkObject*, unsigned long, void*))
    {
      this->Handler = handler;
      this->Method1 = nullptr;
      this->Method2 = nullptr;
      this->Method3 = method;
    }
    ~svtkClassMemberCallback() override {}

    // Called when the event is invoked
    bool operator()(svtkObject* caller, unsigned long event, void* calldata) override
    {
      T* handler = this->Handler.GetPointer();
      if (handler)
      {
        if (this->Method1)
        {
          (handler->*this->Method1)();
        }
        else if (this->Method2)
        {
          (handler->*this->Method2)(caller, event, calldata);
        }
        else if (this->Method3)
        {
          return (handler->*this->Method3)(caller, event, calldata);
        }
      }
      return false;
    }
  };
  //@}

  //@{
  /**
   * Called by templated variants of AddObserver.
   */
  unsigned long AddTemplatedObserver(
    unsigned long event, svtkClassMemberCallbackBase* callable, float priority);
  // Friend to access AddTemplatedObserver().
  friend class svtkObjectCommandInternal;
  //@}
};

#endif
// SVTK-HeaderTest-Exclude: svtkObject.h
