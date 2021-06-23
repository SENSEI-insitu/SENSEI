/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkObjectFactory.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkObjectFactory.h"

#include "svtkDebugLeaks.h"
#include "svtkDynamicLoader.h"
#include "svtkObjectFactoryCollection.h"
#include "svtkOverrideInformation.h"
#include "svtkOverrideInformationCollection.h"
#include "svtkVersion.h"

#include "svtksys/Directory.hxx"

#include <cctype>

svtkObjectFactoryCollection* svtkObjectFactory::RegisteredFactories = nullptr;
static unsigned int svtkObjectFactoryRegistryCleanupCounter = 0;

svtkObjectFactoryRegistryCleanup::svtkObjectFactoryRegistryCleanup()
{
  ++svtkObjectFactoryRegistryCleanupCounter;
}

svtkObjectFactoryRegistryCleanup::~svtkObjectFactoryRegistryCleanup()
{
  if (--svtkObjectFactoryRegistryCleanupCounter == 0)
  {
    svtkObjectFactory::UnRegisterAllFactories();
  }
}

// Create an instance of a named svtk object using the loaded
// factories

svtkObject* svtkObjectFactory::CreateInstance(const char* svtkclassname, bool)
{
  if (!svtkObjectFactory::RegisteredFactories)
  {
    svtkObjectFactory::Init();
  }

  svtkObjectFactory* factory;
  svtkCollectionSimpleIterator osit;
  for (svtkObjectFactory::RegisteredFactories->InitTraversal(osit);
       (factory = svtkObjectFactory::RegisteredFactories->GetNextObjectFactory(osit));)
  {
    svtkObject* newobject = factory->CreateObject(svtkclassname);
    if (newobject)
    {
      return newobject;
    }
  }

  return nullptr;
}

// A one time initialization method.
void svtkObjectFactory::Init()
{
  // Don't do anything if we are already initialized
  if (svtkObjectFactory::RegisteredFactories)
  {
    return;
  }

  svtkObjectFactory::RegisteredFactories = svtkObjectFactoryCollection::New();
  svtkObjectFactory::RegisterDefaults();
  svtkObjectFactory::LoadDynamicFactories();
}

// Register any factories that are always present in SVTK like
// the OpenGL factory, currently this is not done.

void svtkObjectFactory::RegisterDefaults() {}

// Load all libraries in SVTK_AUTOLOAD_PATH

void svtkObjectFactory::LoadDynamicFactories()
{
  // follow PATH convensions
#ifdef _WIN32
  char PathSeparator = ';';
#else
  char PathSeparator = ':';
#endif

  char* LoadPath = nullptr;

#ifndef _WIN32_WCE
  LoadPath = getenv("SVTK_AUTOLOAD_PATH");
#endif
  if (LoadPath == nullptr || LoadPath[0] == 0)
  {
    return;
  }
  std::string CurrentPath;
  CurrentPath.reserve(strlen(LoadPath) + 1);
  char* SeparatorPosition = LoadPath; // initialize to env variable
  while (SeparatorPosition)
  {
    CurrentPath.clear();

    size_t PathLength = 0;
    // find PathSeparator in LoadPath
    SeparatorPosition = strchr(LoadPath, PathSeparator);
    // if not found then use the whole string
    if (SeparatorPosition == nullptr)
    {
      PathLength = strlen(LoadPath);
    }
    else
    {
      PathLength = static_cast<size_t>(SeparatorPosition - LoadPath);
    }
    // copy the path out of LoadPath into CurrentPath
    CurrentPath.append(LoadPath, PathLength);
    // Get ready for the next path
    LoadPath = SeparatorPosition + 1;
    // Load the libraries in the current path
    svtkObjectFactory::LoadLibrariesInPath(CurrentPath);
  }
}

// A file scope helper function to concat path and file into
// a full path
static char* CreateFullPath(const std::string& path, const char* file)
{
  size_t lenpath = path.size();
  char* ret = new char[lenpath + strlen(file) + 2];
#ifdef _WIN32
  const char sep = '\\';
#else
  const char sep = '/';
#endif
  // make sure the end of path is a separator
  strcpy(ret, path.c_str());
  if (ret[lenpath - 1] != sep)
  {
    ret[lenpath] = sep;
    ret[lenpath + 1] = 0;
  }
  strcat(ret, file);
  return ret;
}

// A file scope typedef to make the cast code to the load
// function cleaner to read.
typedef svtkObjectFactory* (*SVTK_LOAD_FUNCTION)();
typedef const char* (*SVTK_VERSION_FUNCTION)();
typedef const char* (*SVTK_COMPILER_FUNCTION)();

// A file scoped function to determine if a file has
// the shared library extension in its name, this converts name to lower
// case before the compare, svtkDynamicLoader always uses
// lower case for LibExtension values.

inline int svtkNameIsSharedLibrary(const char* name)
{
  int len = static_cast<int>(strlen(name));
  char* copy = new char[len + 1];

  for (int i = 0; i < len; i++)
  {
    copy[i] = static_cast<char>(tolower(name[i]));
  }
  copy[len] = 0;
  char* ret = strstr(copy, svtkDynamicLoader::LibExtension());
  delete[] copy;
  return (ret != nullptr);
}

void svtkObjectFactory::LoadLibrariesInPath(const std::string& path)
{
  svtksys::Directory dir;
  if (!dir.Load(path))
  {
    return;
  }

  // Attempt to load each file in the directory as a shared library
  for (unsigned long i = 0; i < dir.GetNumberOfFiles(); i++)
  {
    const char* file = dir.GetFile(i);
    // try to make sure the file has at least the extension
    // for a shared library in it.
    if (svtkNameIsSharedLibrary(file))
    {
      char* fullpath = CreateFullPath(path, file);
      svtkLibHandle lib = svtkDynamicLoader::OpenLibrary(fullpath);
      if (lib)
      {
        // Look for the symbol svtkLoad, svtkGetFactoryCompilerUsed,
        // and svtkGetFactoryVersion in the library
        SVTK_LOAD_FUNCTION loadfunction =
          (SVTK_LOAD_FUNCTION)(svtkDynamicLoader::GetSymbolAddress(lib, "svtkLoad"));
        SVTK_COMPILER_FUNCTION compilerFunction = (SVTK_COMPILER_FUNCTION)(
          svtkDynamicLoader::GetSymbolAddress(lib, "svtkGetFactoryCompilerUsed"));
        SVTK_VERSION_FUNCTION versionFunction =
          (SVTK_VERSION_FUNCTION)(svtkDynamicLoader::GetSymbolAddress(lib, "svtkGetFactoryVersion"));
        // if the symbol is found call it to create the factory
        // from the library
        if (loadfunction && compilerFunction && versionFunction)
        {
          const char* compiler = (*compilerFunction)();
          const char* version = (*versionFunction)();
          if (strcmp(compiler, SVTK_CXX_COMPILER) ||
            strcmp(version, svtkVersion::GetSVTKSourceVersion()))
          {
            svtkGenericWarningMacro(<< "Incompatible factory rejected:"
                                   << "\nRunning SVTK compiled with: " << SVTK_CXX_COMPILER
                                   << "\nFactory compiled with: " << compiler
                                   << "\nRunning SVTK version: " << svtkVersion::GetSVTKSourceVersion()
                                   << "\nFactory version: " << version
                                   << "\nPath to rejected factory: " << fullpath << "\n");
          }
          else
          {
            svtkObjectFactory* newfactory = (*loadfunction)();
            newfactory->LibrarySVTKVersion = strcpy(new char[strlen(version) + 1], version);
            newfactory->LibraryCompilerUsed = strcpy(new char[strlen(compiler) + 1], compiler);
            // initialize class members if load worked
            newfactory->LibraryHandle = static_cast<void*>(lib);
            newfactory->LibraryPath = strcpy(new char[strlen(fullpath) + 1], fullpath);
            svtkObjectFactory::RegisterFactory(newfactory);
            newfactory->Delete();
          }
        }
        // if only the loadfunction is found, then warn
        else if (loadfunction)
        {
          svtkGenericWarningMacro(
            "Old Style Factory not loaded.  Shared object has svtkLoad, but is missing "
            "svtkGetFactoryCompilerUsed and svtkGetFactoryVersion.  Recompile factory: "
            << fullpath << ", and use SVTK_FACTORY_INTERFACE_IMPLEMENT macro.");
        }
      }
      delete[] fullpath;
    }
  }
}

// Recheck the SVTK_AUTOLOAD_PATH for new libraries

void svtkObjectFactory::ReHash()
{
  svtkObjectFactory::UnRegisterAllFactories();
  svtkObjectFactory::Init();
}

// initialize class members
svtkObjectFactory::svtkObjectFactory()
{
  this->LibraryHandle = nullptr;
  this->LibraryPath = nullptr;
  this->OverrideArray = nullptr;
  this->OverrideClassNames = nullptr;
  this->SizeOverrideArray = 0;
  this->OverrideArrayLength = 0;
  this->LibrarySVTKVersion = nullptr;
  this->LibraryCompilerUsed = nullptr;
}

// Unload the library and free the path string
svtkObjectFactory::~svtkObjectFactory()
{
  delete[] this->LibrarySVTKVersion;
  delete[] this->LibraryCompilerUsed;
  delete[] this->LibraryPath;
  this->LibraryPath = nullptr;

  for (int i = 0; i < this->OverrideArrayLength; i++)
  {
    delete[] this->OverrideClassNames[i];
    delete[] this->OverrideArray[i].Description;
    delete[] this->OverrideArray[i].OverrideWithName;
  }
  delete[] this->OverrideArray;
  delete[] this->OverrideClassNames;
  this->OverrideArray = nullptr;
  this->OverrideClassNames = nullptr;
}

// Add a factory to the registered list
void svtkObjectFactory::RegisterFactory(svtkObjectFactory* factory)
{
  if (factory->LibraryHandle == nullptr)
  {
    const char* nonDynamicName = "Non-Dynamicly loaded factory";
    factory->LibraryPath = strcpy(new char[strlen(nonDynamicName) + 1], nonDynamicName);
    factory->LibraryCompilerUsed = strcpy(new char[strlen(SVTK_CXX_COMPILER) + 1], SVTK_CXX_COMPILER);
    factory->LibrarySVTKVersion = strcpy(
      new char[strlen(svtkVersion::GetSVTKSourceVersion()) + 1], svtkVersion::GetSVTKSourceVersion());
  }
  else
  {
    if (strcmp(factory->LibraryCompilerUsed, SVTK_CXX_COMPILER) != 0)
    {
      svtkGenericWarningMacro(<< "Possible incompatible factory load:"
                             << "\nRunning svtk compiled with :\n"
                             << SVTK_CXX_COMPILER << "\nLoaded Factory compiled with:\n"
                             << factory->LibraryCompilerUsed << "\nRejecting factory:\n"
                             << factory->LibraryPath << "\n");
      return;
    }
    if (strcmp(factory->LibrarySVTKVersion, svtkVersion::GetSVTKSourceVersion()) != 0)
    {
      svtkGenericWarningMacro(<< "Possible incompatible factory load:"
                             << "\nRunning svtk version :\n"
                             << svtkVersion::GetSVTKSourceVersion() << "\nLoaded Factory version:\n"
                             << factory->LibrarySVTKVersion << "\nRejecting factory:\n"
                             << factory->LibraryPath << "\n");
      return;
    }
    if (strcmp(factory->GetSVTKSourceVersion(), svtkVersion::GetSVTKSourceVersion()) != 0)
    {
      svtkGenericWarningMacro(<< "Possible incompatible factory load:"
                             << "\nRunning svtk version :\n"
                             << svtkVersion::GetSVTKSourceVersion() << "\nLoaded Factory version:\n"
                             << factory->GetSVTKSourceVersion() << "\nRejecting factory:\n"
                             << factory->LibraryPath << "\n");
      return;
    }
  }

  svtkObjectFactory::Init();
  svtkObjectFactory::RegisteredFactories->AddItem(factory);
}

// print ivars to stream
void svtkObjectFactory::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  if (this->LibraryPath)
  {
    os << indent << "Factory DLL path: " << this->LibraryPath << "\n";
  }
  if (this->LibrarySVTKVersion)
  {
    os << indent << "Library version: " << this->LibrarySVTKVersion << "\n";
  }
  if (this->LibraryCompilerUsed)
  {
    os << indent << "Compiler used: " << this->LibraryCompilerUsed << "\n";
  }
  os << indent << "Factory description: " << this->GetDescription() << endl;
  int num = this->GetNumberOfOverrides();
  os << indent << "Factory overrides " << num << " classes:" << endl;
  indent = indent.GetNextIndent();
  for (int i = 0; i < num; i++)
  {
    os << indent << "Class : " << this->GetClassOverrideName(i) << endl;
    os << indent << "Overridden with: " << this->GetClassOverrideWithName(i) << endl;
    os << indent << "Enable flag: " << this->GetEnableFlag(i) << endl;
    os << endl;
  }
}

// Remove a factory from the list of registered factories.
void svtkObjectFactory::UnRegisterFactory(svtkObjectFactory* factory)
{
  void* lib = factory->LibraryHandle;
  svtkObjectFactory::RegisteredFactories->RemoveItem(factory);
  if (lib)
  {
    svtkDynamicLoader::CloseLibrary(static_cast<svtkLibHandle>(lib));
  }
}

// unregister all factories and delete the RegisteredFactories list
void svtkObjectFactory::UnRegisterAllFactories()
{
  // do not do anything if this is null
  if (!svtkObjectFactory::RegisteredFactories)
  {
    return;
  }
  int num = svtkObjectFactory::RegisteredFactories->GetNumberOfItems();
  // collect up all the library handles so they can be closed
  // AFTER the factory has been deleted.
  void** libs = new void*[num + 1];
  svtkObjectFactory* factory;
  svtkCollectionSimpleIterator osit;
  svtkObjectFactory::RegisteredFactories->InitTraversal(osit);
  int index = 0;
  while ((factory = svtkObjectFactory::RegisteredFactories->GetNextObjectFactory(osit)))
  {
    libs[index++] = factory->LibraryHandle;
  }
  // delete the factory list and its factories
  svtkObjectFactory::RegisteredFactories->Delete();
  svtkObjectFactory::RegisteredFactories = nullptr;
  // now close the libraries
  for (int i = 0; i < num; i++)
  {
    void* lib = libs[i];
    if (lib)
    {
      svtkDynamicLoader::CloseLibrary(reinterpret_cast<svtkLibHandle>(lib));
    }
  }
  delete[] libs;
}

// Register an override function with a factory.
void svtkObjectFactory::RegisterOverride(const char* classOverride, const char* subclass,
  const char* description, int enableFlag, CreateFunction createFunction)
{
  this->GrowOverrideArray();
  int nextIndex = this->OverrideArrayLength;
  this->OverrideArrayLength++;
  char* className = strcpy(new char[strlen(classOverride) + 1], classOverride);
  char* desc = strcpy(new char[strlen(description) + 1], description);
  char* ocn = strcpy(new char[strlen(subclass) + 1], subclass);
  this->OverrideClassNames[nextIndex] = className;
  this->OverrideArray[nextIndex].Description = desc;
  this->OverrideArray[nextIndex].OverrideWithName = ocn;
  this->OverrideArray[nextIndex].EnabledFlag = enableFlag;
  this->OverrideArray[nextIndex].CreateCallback = createFunction;
}

// Create an instance of an object
svtkObject* svtkObjectFactory::CreateObject(const char* svtkclassname)
{
  for (int i = 0; i < this->OverrideArrayLength; i++)
  {
    if (this->OverrideArray[i].EnabledFlag &&
      strcmp(this->OverrideClassNames[i], svtkclassname) == 0)
    {
      return (*this->OverrideArray[i].CreateCallback)();
    }
  }
  return nullptr;
}

// grow the array if the length is greater than the size.
void svtkObjectFactory::GrowOverrideArray()
{
  if (this->OverrideArrayLength + 1 > this->SizeOverrideArray)
  {
    int newLength = this->OverrideArrayLength + 50;
    OverrideInformation* newArray = new OverrideInformation[newLength];
    char** newNameArray = new char*[newLength];
    for (int i = 0; i < this->OverrideArrayLength; i++)
    {
      newNameArray[i] = this->OverrideClassNames[i];
      newArray[i] = this->OverrideArray[i];
    }
    delete[] this->OverrideClassNames;
    this->OverrideClassNames = newNameArray;
    delete[] this->OverrideArray;
    this->OverrideArray = newArray;
  }
}

int svtkObjectFactory::GetNumberOfOverrides()
{
  return this->OverrideArrayLength;
}

const char* svtkObjectFactory::GetClassOverrideName(int index)
{
  return this->OverrideClassNames[index];
}

const char* svtkObjectFactory::GetClassOverrideWithName(int index)
{
  return this->OverrideArray[index].OverrideWithName;
}

svtkTypeBool svtkObjectFactory::GetEnableFlag(int index)
{
  return this->OverrideArray[index].EnabledFlag;
}

const char* svtkObjectFactory::GetOverrideDescription(int index)
{
  return this->OverrideArray[index].Description;
}

// Set the enable flag for a class / subclassName pair
void svtkObjectFactory::SetEnableFlag(
  svtkTypeBool flag, const char* className, const char* subclassName)
{
  for (int i = 0; i < this->OverrideArrayLength; i++)
  {
    if (strcmp(this->OverrideClassNames[i], className) == 0)
    {
      // if subclassName is null, then set on className match
      if (!subclassName)
      {
        this->OverrideArray[i].EnabledFlag = flag;
      }
      else
      {
        if (strcmp(this->OverrideArray[i].OverrideWithName, subclassName) == 0)
        {
          this->OverrideArray[i].EnabledFlag = flag;
        }
      }
    }
  }
}

// Get the enable flag for a className/subclassName pair
svtkTypeBool svtkObjectFactory::GetEnableFlag(const char* className, const char* subclassName)
{
  for (int i = 0; i < this->OverrideArrayLength; i++)
  {
    if (strcmp(this->OverrideClassNames[i], className) == 0)
    {
      if (strcmp(this->OverrideArray[i].OverrideWithName, subclassName) == 0)
      {
        return this->OverrideArray[i].EnabledFlag;
      }
    }
  }
  return 0;
}

// Set the EnabledFlag to 0 for a given classname
void svtkObjectFactory::Disable(const char* className)
{
  for (int i = 0; i < this->OverrideArrayLength; i++)
  {
    if (strcmp(this->OverrideClassNames[i], className) == 0)
    {
      this->OverrideArray[i].EnabledFlag = 0;
    }
  }
}

// 1,0 is the class overridden by className
int svtkObjectFactory::HasOverride(const char* className)
{
  for (int i = 0; i < this->OverrideArrayLength; i++)
  {
    if (strcmp(this->OverrideClassNames[i], className) == 0)
    {
      return 1;
    }
  }
  return 0;
}

// 1,0 is the class overridden by className/subclassName pair
int svtkObjectFactory::HasOverride(const char* className, const char* subclassName)
{
  for (int i = 0; i < this->OverrideArrayLength; i++)
  {
    if (strcmp(this->OverrideClassNames[i], className) == 0)
    {
      if (strcmp(this->OverrideArray[i].OverrideWithName, subclassName) == 0)
      {
        return 1;
      }
    }
  }
  return 0;
}

svtkObjectFactoryCollection* svtkObjectFactory::GetRegisteredFactories()
{
  if (!svtkObjectFactory::RegisteredFactories)
  {
    svtkObjectFactory::Init();
  }

  return svtkObjectFactory::RegisteredFactories;
}

// 1,0 is the className overridden by any registered factories
int svtkObjectFactory::HasOverrideAny(const char* className)
{
  svtkObjectFactory* factory;
  svtkCollectionSimpleIterator osit;
  for (svtkObjectFactory::RegisteredFactories->InitTraversal(osit);
       (factory = svtkObjectFactory::RegisteredFactories->GetNextObjectFactory(osit));)
  {
    if (factory->HasOverride(className))
    {
      return 1;
    }
  }
  return 0;
}

// collect up information about current registered factories
void svtkObjectFactory::GetOverrideInformation(
  const char* name, svtkOverrideInformationCollection* ret)
{
  // create the collection to return
  svtkOverrideInformation* overInfo; // info object pointer
  svtkObjectFactory* factory;        // factory pointer for traversal
  svtkCollectionSimpleIterator osit;
  svtkObjectFactory::RegisteredFactories->InitTraversal(osit);
  for (; (factory = svtkObjectFactory::RegisteredFactories->GetNextObjectFactory(osit));)
  {
    for (int i = 0; i < factory->OverrideArrayLength; i++)
    {
      if (strcmp(name, factory->OverrideClassNames[i]) == 0)
      {
        // Create a new override info class
        overInfo = svtkOverrideInformation::New();
        // Set the class name
        overInfo->SetClassOverrideName(factory->OverrideClassNames[i]);
        // Set the override class name
        overInfo->SetClassOverrideWithName(factory->OverrideArray[i].OverrideWithName);
        // Set the Description for the override
        overInfo->SetDescription(factory->OverrideArray[i].Description);
        // Set the factory for the override
        overInfo->SetObjectFactory(factory);
        // add the item to the collection
        ret->AddItem(overInfo);
        overInfo->Delete();
      }
    }
  }
}

// set enable flag for all registered factories for the given className
void svtkObjectFactory::SetAllEnableFlags(svtkTypeBool flag, const char* className)
{
  svtkObjectFactory* factory;
  svtkCollectionSimpleIterator osit;
  for (svtkObjectFactory::RegisteredFactories->InitTraversal(osit);
       (factory = svtkObjectFactory::RegisteredFactories->GetNextObjectFactory(osit));)
  {
    factory->SetEnableFlag(flag, className, nullptr);
  }
}

// set enable flag for the first factory that that
// has an override for className/subclassName pair
void svtkObjectFactory::SetAllEnableFlags(
  svtkTypeBool flag, const char* className, const char* subclassName)
{
  svtkObjectFactory* factory;
  svtkCollectionSimpleIterator osit;
  for (svtkObjectFactory::RegisteredFactories->InitTraversal(osit);
       (factory = svtkObjectFactory::RegisteredFactories->GetNextObjectFactory(osit));)
  {
    factory->SetEnableFlag(flag, className, subclassName);
  }
}

void svtkObjectFactory::CreateAllInstance(const char* svtkclassname, svtkCollection* retList)
{
  svtkObjectFactory* f;
  svtkObjectFactoryCollection* collection = svtkObjectFactory::GetRegisteredFactories();
  svtkCollectionSimpleIterator osit;
  for (collection->InitTraversal(osit); (f = collection->GetNextObjectFactory(osit));)
  {
    svtkObject* o = f->CreateObject(svtkclassname);
    if (o)
    {
      retList->AddItem(o);
      o->Delete();
    }
  }
}
