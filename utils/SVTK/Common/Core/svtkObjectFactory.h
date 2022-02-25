/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkObjectFactory.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkObjectFactory
 * @brief   abstract base class for svtkObjectFactories
 *
 * svtkObjectFactory is used to create svtk objects.   The base class
 * svtkObjectFactory contains a static method CreateInstance which is used
 * to create svtk objects from the list of registered svtkObjectFactory
 * sub-classes.   The first time CreateInstance is called, all dll's or shared
 * libraries in the environment variable SVTK_AUTOLOAD_PATH are loaded into
 * the current process.   The C functions svtkLoad, svtkGetFactoryCompilerUsed,
 * and svtkGetFactoryVersion are called on each dll.  To implement these
 * functions in a shared library or dll, use the macro:
 * SVTK_FACTORY_INTERFACE_IMPLEMENT.
 * SVTK_AUTOLOAD_PATH is an environment variable
 * containing a colon separated (semi-colon on win32) list of paths.
 *
 * The svtkObjectFactory can be use to override the creation of any object
 * in SVTK with a sub-class of that object.  The factories can be registered
 * either at run time with the SVTK_AUTOLOAD_PATH, or at compile time
 * with the svtkObjectFactory::RegisterFactory method.
 *
 */

#ifndef svtkObjectFactory_h
#define svtkObjectFactory_h

#include "svtkCommonCoreModule.h"  // For export macro
#include "svtkDebugLeaksManager.h" // Must be included before singletons
#include "svtkObject.h"

#include <string> // for std::string

class svtkObjectFactoryCollection;
class svtkOverrideInformationCollection;
class svtkCollection;

class SVTKCOMMONCORE_EXPORT svtkObjectFactory : public svtkObject
{
public:
  // Class Methods used to interface with the registered factories

  /**
   * Create and return an instance of the named svtk object.
   * Each loaded svtkObjectFactory will be asked in the order
   * the factory was in the SVTK_AUTOLOAD_PATH.  After the
   * first factory returns the object no other factories are asked.
   * isAbstract is no longer used. This method calls
   * svtkObjectBase::InitializeObjectBase() on the instance when the
   * return value is non-nullptr.
   */
  SVTK_NEWINSTANCE
  static svtkObject* CreateInstance(const char* svtkclassname, bool isAbstract = false);

  /**
   * Create all possible instances of the named svtk object.
   * Each registered svtkObjectFactory will be asked, and the
   * result will be stored in the user allocated svtkCollection
   * passed in to the function.
   */
  static void CreateAllInstance(const char* svtkclassname, svtkCollection* retList);
  /**
   * Re-check the SVTK_AUTOLOAD_PATH for new factory libraries.
   * This calls UnRegisterAll before re-loading
   */
  static void ReHash();
  /**
   * Register a factory so it can be used to create svtk objects
   */
  static void RegisterFactory(svtkObjectFactory*);
  /**
   * Remove a factory from the list of registered factories
   */
  static void UnRegisterFactory(svtkObjectFactory*);
  /**
   * Unregister all factories
   */
  static void UnRegisterAllFactories();

  /**
   * Return the list of all registered factories.  This is NOT a copy,
   * do not remove items from this list!
   */
  static svtkObjectFactoryCollection* GetRegisteredFactories();

  /**
   * return 1 if one of the registered factories
   * overrides the given class name
   */
  static int HasOverrideAny(const char* className);

  /**
   * Fill the given collection with all the overrides for
   * the class with the given name.
   */
  static void GetOverrideInformation(const char* name, svtkOverrideInformationCollection*);

  /**
   * Set the enable flag for a given named class for all registered
   * factories.
   */
  static void SetAllEnableFlags(svtkTypeBool flag, const char* className);
  /**
   * Set the enable flag for a given named class subclass pair
   * for all registered factories.
   */
  static void SetAllEnableFlags(svtkTypeBool flag, const char* className, const char* subclassName);

  // Instance methods to be used on individual instances of svtkObjectFactory

  // Methods from svtkObject
  svtkTypeMacro(svtkObjectFactory, svtkObject);
  /**
   * Print ObjectFactory to stream.
   */
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * All sub-classes of svtkObjectFactory should must return the version of
   * SVTK they were built with.  This should be implemented with the macro
   * SVTK_SOURCE_VERSION and NOT a call to svtkVersion::GetSVTKSourceVersion.
   * As the version needs to be compiled into the file as a string constant.
   * This is critical to determine possible incompatible dynamic factory loads.
   */
  virtual const char* GetSVTKSourceVersion() = 0;

  /**
   * Return a descriptive string describing the factory.
   */
  virtual const char* GetDescription() = 0;

  /**
   * Return number of overrides this factory can create.
   */
  virtual int GetNumberOfOverrides();

  /**
   * Return the name of a class override at the given index.
   */
  virtual const char* GetClassOverrideName(int index);

  /**
   * Return the name of the class that will override the class
   * at the given index
   */
  virtual const char* GetClassOverrideWithName(int index);

  /**
   * Return the enable flag for the class at the given index.
   */
  virtual svtkTypeBool GetEnableFlag(int index);

  /**
   * Return the description for a the class override at the given
   * index.
   */
  virtual const char* GetOverrideDescription(int index);

  //@{
  /**
   * Set and Get the Enable flag for the specific override of className.
   * if subclassName is null, then it is ignored.
   */
  virtual void SetEnableFlag(svtkTypeBool flag, const char* className, const char* subclassName);
  virtual svtkTypeBool GetEnableFlag(const char* className, const char* subclassName);
  //@}

  /**
   * Return 1 if this factory overrides the given class name, 0 otherwise.
   */
  virtual int HasOverride(const char* className);
  /**
   * Return 1 if this factory overrides the given class name, 0 otherwise.
   */
  virtual int HasOverride(const char* className, const char* subclassName);

  /**
   * Set all enable flags for the given class to 0.  This will
   * mean that the factory will stop producing class with the given
   * name.
   */
  virtual void Disable(const char* className);

  //@{
  /**
   * This returns the path to a dynamically loaded factory.
   */
  svtkGetStringMacro(LibraryPath);
  //@}

  typedef svtkObject* (*CreateFunction)();

protected:
  /**
   * Register object creation information with the factory.
   */
  void RegisterOverride(const char* classOverride, const char* overrideClassName,
    const char* description, int enableFlag, CreateFunction createFunction);

  /**
   * This method is provided by sub-classes of svtkObjectFactory.
   * It should create the named svtk object or return 0 if that object
   * is not supported by the factory implementation.
   */
  virtual svtkObject* CreateObject(const char* svtkclassname);

  svtkObjectFactory();
  ~svtkObjectFactory() override;

  struct OverrideInformation
  {
    char* Description;
    char* OverrideWithName;
    svtkTypeBool EnabledFlag;
    CreateFunction CreateCallback;
  };

  OverrideInformation* OverrideArray;
  char** OverrideClassNames;
  int SizeOverrideArray;
  int OverrideArrayLength;

private:
  void GrowOverrideArray();

  /**
   * Initialize the static members of svtkObjectFactory.   RegisterDefaults
   * is called here.
   */
  static void Init();
  /**
   * Register default factories which are not loaded at run time.
   */
  static void RegisterDefaults();
  /**
   * Load dynamic factories from the SVTK_AUTOLOAD_PATH
   */
  static void LoadDynamicFactories();
  /**
   * Load all dynamic libraries in the given path
   */
  static void LoadLibrariesInPath(const std::string&);

  // list of registered factories
  static svtkObjectFactoryCollection* RegisteredFactories;

  // member variables for a factory set by the base class
  // at load or register time
  void* LibraryHandle;
  char* LibrarySVTKVersion;
  char* LibraryCompilerUsed;
  char* LibraryPath;

private:
  svtkObjectFactory(const svtkObjectFactory&) = delete;
  void operator=(const svtkObjectFactory&) = delete;
};

// Implementation detail for Schwarz counter idiom.
class SVTKCOMMONCORE_EXPORT svtkObjectFactoryRegistryCleanup
{
public:
  svtkObjectFactoryRegistryCleanup();
  ~svtkObjectFactoryRegistryCleanup();

private:
  svtkObjectFactoryRegistryCleanup(const svtkObjectFactoryRegistryCleanup& other) = delete;
  svtkObjectFactoryRegistryCleanup& operator=(const svtkObjectFactoryRegistryCleanup& rhs) = delete;
};
static svtkObjectFactoryRegistryCleanup svtkObjectFactoryRegistryCleanupInstance;

// Macro to create an object creation function.
// The name of the function will by svtkObjectFactoryCreateclassname
// where classname is the name of the class being created
#define SVTK_CREATE_CREATE_FUNCTION(classname)                                                      \
  static svtkObject* svtkObjectFactoryCreate##classname() { return classname::New(); }

#endif

#define SVTK_FACTORY_INTERFACE_EXPORT SVTKCOMMONCORE_EXPORT

// Macro to create the interface "C" functions used in
// a dll or shared library that contains a SVTK object factory.
// Put this function in the .cxx file of your object factory,
// and pass in the name of the factory sub-class that you want
// the dll to create.
#define SVTK_FACTORY_INTERFACE_IMPLEMENT(factoryName)                                               \
  extern "C" SVTK_FACTORY_INTERFACE_EXPORT const char* svtkGetFactoryCompilerUsed()                  \
  {                                                                                                \
    return SVTK_CXX_COMPILER;                                                                       \
  }                                                                                                \
  extern "C" SVTK_FACTORY_INTERFACE_EXPORT const char* svtkGetFactoryVersion()                       \
  {                                                                                                \
    return SVTK_SOURCE_VERSION;                                                                     \
  }                                                                                                \
  extern "C" SVTK_FACTORY_INTERFACE_EXPORT svtkObjectFactory* svtkLoad()                              \
  {                                                                                                \
    return factoryName ::New();                                                                    \
  }

// Macro to implement the body of the object factory form of the New() method.
#define SVTK_OBJECT_FACTORY_NEW_BODY(thisClass)                                                     \
  svtkObject* ret = svtkObjectFactory::CreateInstance(#thisClass, false);                            \
  if (ret)                                                                                         \
  {                                                                                                \
    return static_cast<thisClass*>(ret);                                                           \
  }                                                                                                \
  auto result = new thisClass;                                                                     \
  result->InitializeObjectBase();                                                                  \
  return result

// Macro to implement the body of the abstract object factory form of the New()
// method, i.e. an abstract base class that can only be instantiated if the
// object factory overrides it.
#define SVTK_ABSTRACT_OBJECT_FACTORY_NEW_BODY(thisClass)                                            \
  svtkObject* ret = svtkObjectFactory::CreateInstance(#thisClass, true);                             \
  if (ret)                                                                                         \
  {                                                                                                \
    return static_cast<thisClass*>(ret);                                                           \
  }                                                                                                \
  svtkGenericWarningMacro("Error: no override found for '" #thisClass "'.");                        \
  return nullptr

// Macro to implement the body of the standard form of the New() method.
#if defined(SVTK_ALL_NEW_OBJECT_FACTORY)
#define SVTK_STANDARD_NEW_BODY(thisClass) SVTK_OBJECT_FACTORY_NEW_BODY(thisClass)
#else
#define SVTK_STANDARD_NEW_BODY(thisClass)                                                           \
  auto result = new thisClass;                                                                     \
  result->InitializeObjectBase();                                                                  \
  return result
#endif

// Macro to implement the standard form of the New() method.
#define svtkStandardNewMacro(thisClass)                                                             \
  thisClass* thisClass::New() { SVTK_STANDARD_NEW_BODY(thisClass); }

// Macro to implement the object factory form of the New() method.
#define svtkObjectFactoryNewMacro(thisClass)                                                        \
  thisClass* thisClass::New() { SVTK_OBJECT_FACTORY_NEW_BODY(thisClass); }

// Macro to implement the abstract object factory form of the New() method.
// That is an abstract base class that can only be instantiated if the
// object factory overrides it.
#define svtkAbstractObjectFactoryNewMacro(thisClass)                                                \
  thisClass* thisClass::New() { SVTK_ABSTRACT_OBJECT_FACTORY_NEW_BODY(thisClass); }
