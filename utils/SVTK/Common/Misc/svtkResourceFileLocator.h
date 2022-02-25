/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkResourceFileLocator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class svtkResourceFileLocator
 * @brief utility to locate resource files.
 *
 * SVTK based application often need to locate resource files, such configuration
 * files, Python modules, etc. svtkResourceFileLocator provides methods that can
 * be used to locate such resource files at runtime.
 *
 * Using `Locate`, one can locate files relative to an
 * anchor directory such as the executable directory, or the library directory.
 *
 * `GetLibraryPathForSymbolUnix` and `GetLibraryPathForSymbolWin32` methods can
 * be used to locate the library that provides a particular symbol. For example,
 * this is used by `svtkPythonInterpreter` to ensure that the `svtk` Python package
 * is located relative the SVTK libraries, irrespective of the application location.
 */

#ifndef svtkResourceFileLocator_h
#define svtkResourceFileLocator_h

#include "svtkCommonMiscModule.h" // For export macro
#include "svtkObject.h"

#include <string> // needed for std::string
#include <vector> // needed for std::vector

class SVTKCOMMONMISC_EXPORT svtkResourceFileLocator : public svtkObject
{
public:
  static svtkResourceFileLocator* New();
  svtkTypeMacro(svtkResourceFileLocator, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Enable/disable printing of testing of various path during `Locate`
   * to `stdout`.
   *
   * @deprecated Instead use `SetLogVerbosity` to specify the verbosity at which
   * this instance should log trace information. Default is
   * `svtkLogger::VERBOSITY_TRACE`.
   */
  SVTK_LEGACY(void SetPrintDebugInformation(bool));
  SVTK_LEGACY(bool GetPrintDebugInformation());
  SVTK_LEGACY(void PrintDebugInformationOn());
  SVTK_LEGACY(void PrintDebugInformationOff());
  //@}

  //@{
  /**
   * The log verbosity to use when logging information about the resource
   * searching. Default is `svtkLogger::VERBOSITY_TRACE`.
   */
  svtkSetMacro(LogVerbosity, int);
  svtkGetMacro(LogVerbosity, int);
  //@}

  //@{
  /**
   * Given a starting anchor directory, look for the landmark file relative to
   * the anchor. If found return the anchor. If not found, go one directory up
   * and then look the landmark file again.
   */
  virtual std::string Locate(const std::string& anchor, const std::string& landmark,
    const std::string& defaultDir = std::string());
  //@}

  //@{
  /**
   * This variant is used to look for landmark relative to the anchor using
   * additional prefixes for the landmark file. For example, if you're looking for
   * `svtk/__init__.py`, but it can be placed relative to your anchor location
   * (let's say the executable directory), under "lib" or "lib/python", then
   * use this variant with "lib", and "lib/python" passed in as the landmark
   * prefixes. On success, the returned value will be anchor + matching prefix.
   */
  virtual std::string Locate(const std::string& anchor,
    const std::vector<std::string>& landmark_prefixes, const std::string& landmark,
    const std::string& defaultDir = std::string());
  //@}

  //@{
  /**
   * Returns the name of the library providing the symbol. For example, if you
   * want to locate where the SVTK libraries located call
   * `GetLibraryPathForSymbolUnix("GetSVTKVersion")` on Unixes and
   * `GetLibraryPathForSymbolWin32(GetSVTKVersion)` on Windows. Alternatively, you
   * can simply use the `svtkGetLibraryPathForSymbol(GetSVTKVersion)` macro
   * that makes the appropriate call as per the current platform.
   */
  static std::string GetLibraryPathForSymbolUnix(const char* symbolname);
  static std::string GetLibraryPathForSymbolWin32(const void* fptr);
  //@}

protected:
  svtkResourceFileLocator();
  ~svtkResourceFileLocator() override;

private:
  svtkResourceFileLocator(const svtkResourceFileLocator&) = delete;
  void operator=(const svtkResourceFileLocator&) = delete;

  int LogVerbosity;
};

#if defined(_WIN32) && !defined(__CYGWIN__)
#define svtkGetLibraryPathForSymbol(function)                                                       \
  svtkResourceFileLocator::GetLibraryPathForSymbolWin32(reinterpret_cast<const void*>(&function))
#else
#define svtkGetLibraryPathForSymbol(function)                                                       \
  svtkResourceFileLocator::GetLibraryPathForSymbolUnix(#function)
#endif

#endif
