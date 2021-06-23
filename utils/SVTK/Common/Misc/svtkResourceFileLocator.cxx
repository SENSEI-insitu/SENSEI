/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkResourceFileLocator.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkResourceFileLocator.h"

#include "svtkLogger.h"
#include "svtkObjectFactory.h"

#include <svtksys/SystemTools.hxx>

#if defined(_WIN32) && !defined(__CYGWIN__)
#define SVTK_PATH_SEPARATOR "\\"
#else
#define SVTK_PATH_SEPARATOR "/"
#endif

#define SVTK_FILE_LOCATOR_DEBUG_MESSAGE(...)                                                        \
  svtkVLogF(static_cast<svtkLogger::Verbosity>(this->LogVerbosity), __VA_ARGS__)

#if defined(_WIN32) && !defined(__CYGWIN__)
// Implementation for Windows win32 code but not cygwin
#include <windows.h>
#else
#include <dlfcn.h>
#endif

svtkStandardNewMacro(svtkResourceFileLocator);
//----------------------------------------------------------------------------
svtkResourceFileLocator::svtkResourceFileLocator()
  : LogVerbosity(svtkLogger::VERBOSITY_TRACE)
{
}

//----------------------------------------------------------------------------
svtkResourceFileLocator::~svtkResourceFileLocator() = default;

//----------------------------------------------------------------------------
std::string svtkResourceFileLocator::Locate(
  const std::string& anchor, const std::string& landmark, const std::string& defaultDir)
{
  return this->Locate(anchor, { std::string() }, landmark, defaultDir);
}

//----------------------------------------------------------------------------
std::string svtkResourceFileLocator::Locate(const std::string& anchor,
  const std::vector<std::string>& landmark_prefixes, const std::string& landmark,
  const std::string& defaultDir)
{
  svtkVLogScopeF(
    static_cast<svtkLogger::Verbosity>(this->LogVerbosity), "looking for '%s'", landmark.c_str());
  std::vector<std::string> path_components;
  svtksys::SystemTools::SplitPath(anchor, path_components);
  while (!path_components.empty())
  {
    std::string curanchor = svtksys::SystemTools::JoinPath(path_components);
    for (const std::string& curprefix : landmark_prefixes)
    {
      const std::string landmarkdir =
        curprefix.empty() ? curanchor : curanchor + SVTK_PATH_SEPARATOR + curprefix;
      const std::string landmarktocheck = landmarkdir + SVTK_PATH_SEPARATOR + landmark;
      if (svtksys::SystemTools::FileExists(landmarktocheck))
      {
        SVTK_FILE_LOCATOR_DEBUG_MESSAGE("trying file %s -- success!", landmarktocheck.c_str());
        return landmarkdir;
      }
      else
      {
        SVTK_FILE_LOCATOR_DEBUG_MESSAGE("trying file %s -- failed!", landmarktocheck.c_str());
      }
    }
    path_components.pop_back();
  }
  return defaultDir;
}

//----------------------------------------------------------------------------
std::string svtkResourceFileLocator::GetLibraryPathForSymbolUnix(const char* symbolname)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
  (void)symbolname;
  return std::string();
#else
  void* handle = dlsym(RTLD_DEFAULT, symbolname);
  if (!handle)
  {
    return std::string();
  }

  Dl_info di;
  int ret = dladdr(handle, &di);
  if (ret == 0 || !di.dli_saddr || !di.dli_fname)
  {
    return std::string();
  }

  return std::string(di.dli_fname);
#endif
}

//----------------------------------------------------------------------------
std::string svtkResourceFileLocator::GetLibraryPathForSymbolWin32(const void* fptr)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
  MEMORY_BASIC_INFORMATION mbi;
  VirtualQuery(fptr, &mbi, sizeof(mbi));
  char pathBuf[16384];
  if (!GetModuleFileName(static_cast<HMODULE>(mbi.AllocationBase), pathBuf, sizeof(pathBuf)))
  {
    return std::string();
  }

  return std::string(pathBuf);
#else
  (void)fptr;
  return std::string();
#endif
}

//----------------------------------------------------------------------------
void svtkResourceFileLocator::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "LogVerbosity: " << this->LogVerbosity << endl;
}

#if !defined(SVTK_LEGACY_REMOVE)
void svtkResourceFileLocator::SetPrintDebugInformation(bool val)
{
  SVTK_LEGACY_REPLACED_BODY(svtkResourceFileLocator::SetPrintDebugInformation, "SVTK 9.0",
    svtkResourceFileLocator::SetLogVerbosity);
  this->SetLogVerbosity(val ? svtkLogger::VERBOSITY_INFO : svtkLogger::VERBOSITY_TRACE);
}

bool svtkResourceFileLocator::GetPrintDebugInformation()
{
  SVTK_LEGACY_REPLACED_BODY(svtkResourceFileLocator::GetPrintDebugInformation, "SVTK 9.0",
    svtkResourceFileLocator::GetLogVerbosity);
  return (this->GetLogVerbosity() == svtkLogger::VERBOSITY_INFO);
}

void svtkResourceFileLocator::PrintDebugInformationOn()
{
  SVTK_LEGACY_REPLACED_BODY(svtkResourceFileLocator::PrintDebugInformationOn, "SVTK 9.0",
    svtkResourceFileLocator::SetLogVerbosity);
  this->SetLogVerbosity(svtkLogger::VERBOSITY_INFO);
}

void svtkResourceFileLocator::PrintDebugInformationOff()
{
  SVTK_LEGACY_REPLACED_BODY(svtkResourceFileLocator::PrintDebugInformationOff, "SVTK 9.0",
    svtkResourceFileLocator::SetLogVerbosity);
  this->SetLogVerbosity(svtkLogger::VERBOSITY_TRACE);
}

#endif
