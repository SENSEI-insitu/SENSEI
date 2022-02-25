#include "svtkLogger.h"
#include "svtkNew.h"
#include "svtkResourceFileLocator.h"
#include "svtkVersion.h"

#include <svtksys/SystemTools.hxx>

int TestResourceFileLocator(int, char*[])
{
  auto svtklib = svtkGetLibraryPathForSymbol(GetSVTKVersion);
  if (svtklib.empty())
  {
    cerr << "FAILED to locate `GetSVTKVersion`." << endl;
    return EXIT_FAILURE;
  }
  const std::string svtkdir = svtksys::SystemTools::GetFilenamePath(svtklib);

  svtkNew<svtkResourceFileLocator> locator;
  locator->SetLogVerbosity(svtkLogger::VERBOSITY_INFO);
  auto path = locator->Locate(svtkdir, "Testing/Temporary");
  if (path.empty())
  {
    cerr << "FAILED to locate 'Testing/Temporary' dir." << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
