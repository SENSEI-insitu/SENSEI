/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPythonInterpreter.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkPythonInterpreter.h"
#include "svtkPython.h" // this must be the first include.

#include "svtkCommand.h"
#include "svtkLogger.h"
#include "svtkNew.h"
#include "svtkObjectFactory.h"
#include "svtkOutputWindow.h"
#include "svtkPythonStdStreamCaptureHelper.h"
#include "svtkResourceFileLocator.h"
#include "svtkVersion.h"
#include "svtkWeakPointer.h"

#include <svtksys/SystemInformation.hxx>
#include <svtksys/SystemTools.hxx>

#include <algorithm>
#include <csignal>
#include <sstream>
#include <string>
#include <vector>

#if PY_VERSION_HEX >= 0x03000000
#if defined(__APPLE__) && PY_VERSION_HEX < 0x03050000
extern "C"
{
  extern wchar_t* _Py_DecodeUTF8_surrogateescape(const char* s, Py_ssize_t size);
}
#endif
#endif

#if defined(_WIN32) && !defined(__CYGWIN__)
#define SVTK_PATH_SEPARATOR "\\"
#else
#define SVTK_PATH_SEPARATOR "/"
#endif

#define SVTKPY_DEBUG_MESSAGE(x)                                                                     \
  svtkVLog(svtkLogger::ConvertToVerbosity(svtkPythonInterpreter::GetLogVerbosity()), x)
#define SVTKPY_DEBUG_MESSAGE_VV(x)                                                                  \
  svtkVLog(svtkLogger::ConvertToVerbosity(svtkPythonInterpreter::GetLogVerbosity() + 1), x)

#if defined(_WIN32) && !defined(__CYGWIN__) && defined(SVTK_BUILD_SHARED_LIBS) &&                   \
  PY_VERSION_HEX >= 0x03080000
#define svtkPythonInterpreter_USE_DIRECTORY_COOKIE
#endif

namespace
{

template <class T>
void strFree(T* foo)
{
  delete[] foo;
}

template <class T>
class PoolT
{
  std::vector<T*> Strings;

public:
  ~PoolT()
  {
    for (T* astring : this->Strings)
    {
      strFree(astring);
    }
  }

  T* push_back(T* val)
  {
    this->Strings.push_back(val);
    return val;
  }
};

using StringPool = PoolT<char>;
#if PY_VERSION_HEX >= 0x03000000
template <>
void strFree(wchar_t* foo)
{
#if PY_VERSION_HEX >= 0x03050000
  PyMem_RawFree(foo);
#else
  PyMem_Free(foo);
#endif
}
using WCharStringPool = PoolT<wchar_t>;
#endif

#if PY_VERSION_HEX >= 0x03000000
wchar_t* svtk_Py_DecodeLocale(const char* arg, size_t* size)
{
  (void)size;
#if PY_VERSION_HEX >= 0x03050000
  return Py_DecodeLocale(arg, size);
#elif defined(__APPLE__)
  return _Py_DecodeUTF8_surrogateescape(arg, strlen(arg));
#else
  return _Py_char2wchar(arg, size);
#endif
}
#endif

#if PY_VERSION_HEX >= 0x03000000
char* svtk_Py_EncodeLocale(const wchar_t* arg, size_t* size)
{
  (void)size;
#if PY_VERSION_HEX >= 0x03050000
  return Py_EncodeLocale(arg, size);
#else
  return _Py_wchar2char(arg, size);
#endif
}
#endif

static std::vector<svtkWeakPointer<svtkPythonInterpreter> >* GlobalInterpreters;
static std::vector<std::string> PythonPaths;

void NotifyInterpreters(unsigned long eventid, void* calldata = nullptr)
{
  std::vector<svtkWeakPointer<svtkPythonInterpreter> >::iterator iter;
  for (iter = GlobalInterpreters->begin(); iter != GlobalInterpreters->end(); ++iter)
  {
    if (iter->GetPointer())
    {
      iter->GetPointer()->InvokeEvent(eventid, calldata);
    }
  }
}

inline void svtkPrependPythonPath(const char* pathtoadd)
{
  SVTKPY_DEBUG_MESSAGE("adding module search path " << pathtoadd);
  svtkPythonScopeGilEnsurer gilEnsurer;
  PyObject* path = PySys_GetObject(const_cast<char*>("path"));
#if PY_VERSION_HEX >= 0x03000000
  PyObject* newpath = PyUnicode_FromString(pathtoadd);
#else
  PyObject* newpath = PyString_FromString(pathtoadd);
#endif

  // avoid adding duplicate paths.
  if (PySequence_Contains(path, newpath) == 0)
  {
    PyList_Insert(path, 0, newpath);
  }
  Py_DECREF(newpath);
}

}

// Schwarz counter idiom for GlobalInterpreters object
static unsigned int svtkPythonInterpretersCounter;
svtkPythonGlobalInterpreters::svtkPythonGlobalInterpreters()
{
  if (svtkPythonInterpretersCounter++ == 0)
  {
    GlobalInterpreters = new std::vector<svtkWeakPointer<svtkPythonInterpreter> >();
  };
}

svtkPythonGlobalInterpreters::~svtkPythonGlobalInterpreters()
{
  if (--svtkPythonInterpretersCounter == 0)
  {
    delete GlobalInterpreters;
    GlobalInterpreters = nullptr;
  }
}

bool svtkPythonInterpreter::InitializedOnce = false;
bool svtkPythonInterpreter::CaptureStdin = false;
bool svtkPythonInterpreter::ConsoleBuffering = false;
std::string svtkPythonInterpreter::StdErrBuffer;
std::string svtkPythonInterpreter::StdOutBuffer;
int svtkPythonInterpreter::LogVerbosity = svtkLogger::VERBOSITY_TRACE;

svtkStandardNewMacro(svtkPythonInterpreter);
//----------------------------------------------------------------------------
svtkPythonInterpreter::svtkPythonInterpreter()
{
  GlobalInterpreters->push_back(this);
}

//----------------------------------------------------------------------------
svtkPythonInterpreter::~svtkPythonInterpreter()
{
  // We need to check that GlobalInterpreters has not been deleted yet. It can be
  // deleted prior to a call to this destructor if another static object with a
  // reference to a svtkPythonInterpreter object deletes that object after
  // GlobalInterpreters has been destructed. It all depends on the destruction order
  // of the other static object and GlobalInterpreters.
  if (!GlobalInterpreters)
  {
    return;
  }
  std::vector<svtkWeakPointer<svtkPythonInterpreter> >::iterator iter;
  for (iter = GlobalInterpreters->begin(); iter != GlobalInterpreters->end(); ++iter)
  {
    if (*iter == this)
    {
      GlobalInterpreters->erase(iter);
      break;
    }
  }
}

//----------------------------------------------------------------------------
bool svtkPythonInterpreter::IsInitialized()
{
  return (Py_IsInitialized() != 0);
}

//----------------------------------------------------------------------------
void svtkPythonInterpreter::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
bool svtkPythonInterpreter::Initialize(int initsigs /*=0*/)
{
  if (Py_IsInitialized() == 0)
  {
    // guide the mechanism to locate Python standard library, if possible.
    svtkPythonInterpreter::SetupPythonPrefix();

    Py_InitializeEx(initsigs);

    // setup default argv. Without this, code snippets that check `sys.argv` may
    // fail when run in embedded SVTK Python environment.
    PySys_SetArgvEx(0, nullptr, 0);

#ifdef SVTK_PYTHON_FULL_THREADSAFE
    int threadInit = PyEval_ThreadsInitialized();
    if (!threadInit)
    {
      PyEval_InitThreads(); // initialize and acquire GIL
    }
    // Always release GIL, as it has been acquired either by PyEval_InitThreads
    // prior to Python 3.7 or by Py_InitializeEx in Python 3.7 and after
    PyEval_SaveThread();
#endif

#ifdef SIGINT
    // Put default SIGINT handler back after Py_Initialize/Py_InitializeEx.
    signal(SIGINT, SIG_DFL);
#endif
  }

  if (!svtkPythonInterpreter::InitializedOnce)
  {
    svtkPythonInterpreter::InitializedOnce = true;

    // HACK: Calling PyRun_SimpleString for the first time for some reason results in
    // a "\n" message being generated which is causing the error dialog to
    // popup. So we flush that message out of the system before setting up the
    // callbacks.
    svtkPythonInterpreter::RunSimpleString("");

    // Redirect Python's stdout and stderr and stdin - GIL protected operation
    {
      // Setup handlers for stdout/stdin/stderr.
      svtkPythonStdStreamCaptureHelper* wrapperOut = NewPythonStdStreamCaptureHelper(false);
      svtkPythonStdStreamCaptureHelper* wrapperErr = NewPythonStdStreamCaptureHelper(true);
      svtkPythonScopeGilEnsurer gilEnsurer;
      PySys_SetObject(const_cast<char*>("stdout"), reinterpret_cast<PyObject*>(wrapperOut));
      PySys_SetObject(const_cast<char*>("stderr"), reinterpret_cast<PyObject*>(wrapperErr));
      PySys_SetObject(const_cast<char*>("stdin"), reinterpret_cast<PyObject*>(wrapperOut));
      Py_DECREF(wrapperOut);
      Py_DECREF(wrapperErr);
    }

    // We call this before processing any of Python paths added by the
    // application using `PrependPythonPath`. This ensures that application
    // specified paths are preferred to the ones `svtkPythonInterpreter` adds.
    svtkPythonInterpreter::SetupSVTKPythonPaths();

    for (size_t cc = 0; cc < PythonPaths.size(); cc++)
    {
      svtkPrependPythonPath(PythonPaths[cc].c_str());
    }

    NotifyInterpreters(svtkCommand::EnterEvent);
    return true;
  }

  return false;
}

#ifdef svtkPythonInterpreter_USE_DIRECTORY_COOKIE
static PyObject* DLLDirectoryCookie = nullptr;

static void CloseDLLDirectoryCookie()
{
  if (DLLDirectoryCookie)
  {
    PyObject* close = PyObject_GetAttrString(DLLDirectoryCookie, "close");
    if (close)
    {
      PyObject* ret = PyObject_CallMethodObjArgs(DLLDirectoryCookie, close, nullptr);
      Py_XDECREF(ret);
    }

    Py_XDECREF(DLLDirectoryCookie);
    DLLDirectoryCookie = nullptr;
  }
}
#endif

//----------------------------------------------------------------------------
void svtkPythonInterpreter::Finalize()
{
  if (Py_IsInitialized() != 0)
  {
    NotifyInterpreters(svtkCommand::ExitEvent);
    svtkPythonScopeGilEnsurer gilEnsurer(false, true);
#ifdef svtkPythonInterpreter_USE_DIRECTORY_COOKIE
    CloseDLLDirectoryCookie();
#endif
    // Py_Finalize will take care of releasing gil
    Py_Finalize();
  }
}

//----------------------------------------------------------------------------
void svtkPythonInterpreter::SetProgramName(const char* programname)
{
  if (programname)
  {
// From Python Docs: The argument should point to a zero-terminated character
// string in static storage whose contents will not change for the duration of
// the program's execution. No code in the Python interpreter will change the
// contents of this storage.
#if PY_VERSION_HEX >= 0x03000000
    wchar_t* argv0 = svtk_Py_DecodeLocale(programname, nullptr);
    if (argv0 == 0)
    {
      fprintf(stderr,
        "Fatal svtkpython error: "
        "unable to decode the program name\n");
      static wchar_t empty[1] = { 0 };
      argv0 = empty;
      Py_SetProgramName(argv0);
    }
    else
    {
      static WCharStringPool wpool;
      Py_SetProgramName(wpool.push_back(argv0));
    }
#else
    static StringPool pool;
    Py_SetProgramName(pool.push_back(svtksys::SystemTools::DuplicateString(programname)));
#endif
  }
}

//----------------------------------------------------------------------------
void svtkPythonInterpreter::PrependPythonPath(const char* dir)
{
  if (!dir)
  {
    return;
  }

  std::string out_dir = dir;

#if defined(_WIN32) && !defined(__CYGWIN__)
  // Convert slashes for this platform.
  std::replace(out_dir.begin(), out_dir.end(), '/', '\\');
#endif

  if (Py_IsInitialized() == 0)
  {
    // save path for future use.
    PythonPaths.push_back(out_dir);
    return;
  }

  // Append the path to the python sys.path object.
  svtkPrependPythonPath(out_dir.c_str());
}

//----------------------------------------------------------------------------
void svtkPythonInterpreter::PrependPythonPath(
  const char* anchor, const char* landmark, bool add_landmark)
{
  const std::vector<std::string> prefixes = {
    SVTK_PYTHON_SITE_PACKAGES_SUFFIX
#if defined(__APPLE__)
    // if in an App bundle, the `sitepackages` dir is <app_root>/Contents/Python
    ,
    "Contents/Python"
#endif
    ,
    "."
  };

  svtkNew<svtkResourceFileLocator> locator;
  locator->SetLogVerbosity(svtkPythonInterpreter::GetLogVerbosity() + 1);
  std::string path = locator->Locate(anchor, prefixes, landmark);
  if (!path.empty())
  {
    if (add_landmark)
    {
      path = path + "/" + landmark;
    }
    svtkPythonInterpreter::PrependPythonPath(path.c_str());
  }
}

//----------------------------------------------------------------------------
int svtkPythonInterpreter::PyMain(int argc, char** argv)
{
  svtksys::SystemTools::EnableMSVCDebugHook();

  int count_v = 0;
  for (int cc = 0; cc < argc; ++cc)
  {
    if (argv[cc] && strcmp(argv[cc], "-v") == 0)
    {
      ++count_v;
    }
    if (argv[cc] && strcmp(argv[cc], "-vv") == 0)
    {
      count_v += 2;
    }
  }

  if (count_v > 0)
  {
    // change the svtkPythonInterpreter's log verbosity. We only touch it
    // if the command line arguments explicitly requested a certain verbosity.
    svtkPythonInterpreter::SetLogVerbosity(svtkLogger::VERBOSITY_INFO);
    svtkLogger::SetStderrVerbosity(svtkLogger::ConvertToVerbosity(count_v - 1));
  }
  else
  {
    // update log verbosity such that default is to only show errors/warnings.
    // this avoids show the standard loguru INFO messages for executable args etc.
    // unless `-v` was specified.
    svtkLogger::SetStderrVerbosity(svtkLogger::VERBOSITY_WARNING);
  }

  svtkLogger::Init(argc, argv, nullptr); // since `-v` and `-vv` are parsed as Python verbosity flags
                                        // and not log verbosity flags.

  svtkPythonInterpreter::Initialize(1);

#if PY_VERSION_HEX >= 0x03000000

#if PY_VERSION_HEX >= 0x03070000 && PY_VERSION_HEX < 0x03080000
  // Python 3.7.0 has a bug where Py_InitializeEx (called above) followed by
  // Py_Main (at the end of this block) causes a crash. Gracefully exit with
  // failure if we're using 3.7.0 and suggest getting the newest 3.7.x release.
  // See <https://gitlab.kitware.com/svtk/svtk/issues/17434> for details.
  {
    bool is_ok = true;
    svtkPythonScopeGilEnsurer gilEnsurer(false, true);
    PyObject* sys = PyImport_ImportModule("sys");
    if (sys)
    {
      // XXX: Check sys.implementation.name == 'cpython'?

      PyObject* version_info = PyObject_GetAttrString(sys, "version_info");
      if (version_info)
      {
        PyObject* major = PyObject_GetAttrString(version_info, "major");
        PyObject* minor = PyObject_GetAttrString(version_info, "minor");
        PyObject* micro = PyObject_GetAttrString(version_info, "micro");

        auto py_number_cmp = [](PyObject* obj, long expected) {
          return obj && PyLong_Check(obj) && PyLong_AsLong(obj) == expected;
        };

        // Only 3.7.0 has this issue. Any failures to get the version
        // information is OK; we'll just crash later anyways if the version is
        // bad.
        is_ok = !py_number_cmp(major, 3) || !py_number_cmp(minor, 7) || !py_number_cmp(micro, 0);

        Py_XDECREF(micro);
        Py_XDECREF(minor);
        Py_XDECREF(major);
      }

      Py_XDECREF(version_info);
    }

    Py_XDECREF(sys);

    if (!is_ok)
    {
      std::cerr << "Python 3.7.0 has a known issue that causes a crash with a "
                   "specific API usage pattern. This has been fixed in 3.7.1 and all "
                   "newer 3.7.x Python releases. Exiting now to avoid the crash."
                << std::endl;
      return 1;
    }
  }
#endif

  // Need two copies of args, because programs might modify the first
  wchar_t** argvWide = new wchar_t*[argc];
  wchar_t** argvWide2 = new wchar_t*[argc];
  int argcWide = 0;
  for (int i = 0; i < argc; i++)
  {
    if (argv[i] && strcmp(argv[i], "--enable-bt") == 0)
    {
      svtksys::SystemInformation::SetStackTraceOnError(1);
      continue;
    }
    if (argv[i] && strcmp(argv[i], "-V") == 0)
    {
      // print out SVTK version and let argument pass to Py_Main(). At which point,
      // Python will print its version and exit.
      cout << svtkVersion::GetSVTKSourceVersion() << endl;
    }

    argvWide[argcWide] = svtk_Py_DecodeLocale(argv[i], nullptr);
    argvWide2[argcWide] = argvWide[argcWide];
    if (argvWide[argcWide] == 0)
    {
      fprintf(stderr,
        "Fatal svtkpython error: "
        "unable to decode the command line argument #%i\n",
        i + 1);
      for (int k = 0; k < argcWide; k++)
      {
        PyMem_Free(argvWide2[k]);
      }
      delete[] argvWide;
      delete[] argvWide2;
      return 1;
    }
    argcWide++;
  }
  svtkPythonScopeGilEnsurer gilEnsurer(false, true);
  int res = Py_Main(argcWide, argvWide);
  for (int i = 0; i < argcWide; i++)
  {
    PyMem_Free(argvWide2[i]);
  }
  delete[] argvWide;
  delete[] argvWide2;
  return res;
#else

  // process command line arguments to remove unhandled args.
  std::vector<char*> newargv;
  for (int i = 0; i < argc; ++i)
  {
    if (argv[i] && strcmp(argv[i], "--enable-bt") == 0)
    {
      svtksys::SystemInformation::SetStackTraceOnError(1);
      continue;
    }
    if (argv[i] && strcmp(argv[i], "-V") == 0)
    {
      // print out SVTK version and let argument pass to Py_Main(). At which point,
      // Python will print its version and exit.
      cout << svtkVersion::GetSVTKSourceVersion() << endl;
    }
    newargv.push_back(argv[i]);
  }

  svtkPythonScopeGilEnsurer gilEnsurer(false, true);
  return Py_Main(static_cast<int>(newargv.size()), &newargv[0]);
#endif
}

//----------------------------------------------------------------------------
int svtkPythonInterpreter::RunSimpleString(const char* script)
{
  svtkPythonInterpreter::Initialize(1);
  svtkPythonInterpreter::ConsoleBuffering = true;

  // The embedded python interpreter cannot handle DOS line-endings, see
  // http://sourceforge.net/tracker/?group_id=5470&atid=105470&func=detail&aid=1167922
  std::string buffer = script ? script : "";
  buffer.erase(std::remove(buffer.begin(), buffer.end(), '\r'), buffer.end());

  // The cast is necessary because PyRun_SimpleString() hasn't always been const-correct
  int pyReturn;
  {
    svtkPythonScopeGilEnsurer gilEnsurer;
    pyReturn = PyRun_SimpleString(const_cast<char*>(buffer.c_str()));
  }

  svtkPythonInterpreter::ConsoleBuffering = false;
  if (!svtkPythonInterpreter::StdErrBuffer.empty())
  {
    svtkOutputWindow::GetInstance()->DisplayErrorText(svtkPythonInterpreter::StdErrBuffer.c_str());
    NotifyInterpreters(
      svtkCommand::ErrorEvent, const_cast<char*>(svtkPythonInterpreter::StdErrBuffer.c_str()));
    svtkPythonInterpreter::StdErrBuffer.clear();
  }
  if (!svtkPythonInterpreter::StdOutBuffer.empty())
  {
    svtkOutputWindow::GetInstance()->DisplayText(svtkPythonInterpreter::StdOutBuffer.c_str());
    NotifyInterpreters(
      svtkCommand::SetOutputEvent, const_cast<char*>(svtkPythonInterpreter::StdOutBuffer.c_str()));
    svtkPythonInterpreter::StdOutBuffer.clear();
  }

  return pyReturn;
}

//----------------------------------------------------------------------------
void svtkPythonInterpreter::SetCaptureStdin(bool val)
{
  svtkPythonInterpreter::CaptureStdin = val;
}

//----------------------------------------------------------------------------
bool svtkPythonInterpreter::GetCaptureStdin()
{
  return svtkPythonInterpreter::CaptureStdin;
}

//----------------------------------------------------------------------------
void svtkPythonInterpreter::WriteStdOut(const char* txt)
{
  if (svtkPythonInterpreter::ConsoleBuffering)
  {
    svtkPythonInterpreter::StdOutBuffer += std::string(txt);
  }
  else
  {
    svtkOutputWindow::GetInstance()->DisplayText(txt);
    NotifyInterpreters(svtkCommand::SetOutputEvent, const_cast<char*>(txt));
  }
}

//----------------------------------------------------------------------------
void svtkPythonInterpreter::FlushStdOut() {}

//----------------------------------------------------------------------------
void svtkPythonInterpreter::WriteStdErr(const char* txt)
{
  if (svtkPythonInterpreter::ConsoleBuffering)
  {
    svtkPythonInterpreter::StdErrBuffer += std::string(txt);
  }
  else
  {
    svtkOutputWindow::GetInstance()->DisplayErrorText(txt);
    NotifyInterpreters(svtkCommand::ErrorEvent, const_cast<char*>(txt));
  }
}

//----------------------------------------------------------------------------
void svtkPythonInterpreter::FlushStdErr() {}

//----------------------------------------------------------------------------
svtkStdString svtkPythonInterpreter::ReadStdin()
{
  if (!svtkPythonInterpreter::CaptureStdin)
  {
    svtkStdString string;
    cin >> string;
    return string;
  }
  svtkStdString string;
  NotifyInterpreters(svtkCommand::UpdateEvent, &string);
  return string;
}

//----------------------------------------------------------------------------
void svtkPythonInterpreter::SetupPythonPrefix()
{
  using systools = svtksys::SystemTools;

  // Check Py_FrozenFlag global variable defined by Python to see if we're using
  // frozen Python.
  if (Py_FrozenFlag)
  {
    SVTKPY_DEBUG_MESSAGE("`Py_FrozenFlag` is set. Skipping setting up of program path.");
    return;
  }

  std::string pythonlib = svtkGetLibraryPathForSymbol(Py_SetProgramName);
  if (pythonlib.empty())
  {
    SVTKPY_DEBUG_MESSAGE("static Python build or `Py_SetProgramName` library couldn't be found. "
                        "Set `PYTHONHOME` if Python standard library fails to load.");
    return;
  }

  const std::string newprogramname =
    systools::GetFilenamePath(pythonlib) + SVTK_PATH_SEPARATOR "svtkpython";
  SVTKPY_DEBUG_MESSAGE(
    "calling Py_SetProgramName(" << newprogramname << ") to aid in setup of Python prefix.");
#if PY_VERSION_HEX >= 0x03000000
  static WCharStringPool wpool;
  Py_SetProgramName(wpool.push_back(svtk_Py_DecodeLocale(newprogramname.c_str(), nullptr)));
#else
  static StringPool pool;
  Py_SetProgramName(pool.push_back(systools::DuplicateString(newprogramname.c_str())));
#endif
}

//----------------------------------------------------------------------------
void svtkPythonInterpreter::SetupSVTKPythonPaths()
{
  // Check Py_FrozenFlag global variable defined by Python to see if we're using
  // frozen Python.
  if (Py_FrozenFlag)
  {
    SVTKPY_DEBUG_MESSAGE("`Py_FrozenFlag` is set. Skipping locating of `svtk` package.");
    return;
  }

  using systools = svtksys::SystemTools;
  std::string svtklib = svtkGetLibraryPathForSymbol(GetSVTKVersion);
  if (svtklib.empty())
  {
    SVTKPY_DEBUG_MESSAGE(
      "`GetSVTKVersion` library couldn't be found. Will use `Py_GetProgramName` next.");
  }

  if (svtklib.empty())
  {
#if PY_VERSION_HEX >= 0x03000000
    auto tmp = svtk_Py_EncodeLocale(Py_GetProgramName(), nullptr);
    svtklib = tmp;
    PyMem_Free(tmp);
#else
    svtklib = Py_GetProgramName();
#endif
  }

  svtklib = systools::CollapseFullPath(svtklib);
  const std::string svtkdir = systools::GetFilenamePath(svtklib);

#if defined(_WIN32) && !defined(__CYGWIN__) && defined(SVTK_BUILD_SHARED_LIBS)
  // On Windows, based on how the executable is run, we end up failing to load
  // pyd files due to inability to load dependent dlls. This seems to overcome
  // the issue.
  if (!svtkdir.empty())
  {
#if PY_VERSION_HEX >= 0x03080000
    svtkPythonScopeGilEnsurer gilEnsurer(false, true);
    CloseDLLDirectoryCookie();
    PyObject* os = PyImport_ImportModule("os");
    if (os)
    {
      PyObject* add_dll_directory = PyObject_GetAttrString(os, "add_dll_directory");
      if (add_dll_directory && PyCallable_Check(add_dll_directory))
      {
        PyObject* newpath = PyUnicode_FromString(svtkdir.c_str());
        DLLDirectoryCookie = PyObject_CallFunctionObjArgs(add_dll_directory, newpath, nullptr);
        Py_XDECREF(newpath);
      }

      Py_XDECREF(add_dll_directory);
    }

    Py_XDECREF(os);
#else
    std::string env_path;
    if (systools::GetEnv("PATH", env_path))
    {
      env_path = svtkdir + ";" + env_path;
    }
    else
    {
      env_path = svtkdir;
    }
    systools::PutEnv(std::string("PATH=") + env_path);
#endif
  }
#endif

#if defined(SVTK_BUILD_SHARED_LIBS)
  svtkPythonInterpreter::PrependPythonPath(svtkdir.c_str(), "svtkmodules/__init__.py");
#else
  // since there may be other packages not zipped (e.g. mpi4py), we added path to _svtk.zip
  // to the search path as well.
  svtkPythonInterpreter::PrependPythonPath(svtkdir.c_str(), "_svtk.zip", /*add_landmark*/ false);
  svtkPythonInterpreter::PrependPythonPath(svtkdir.c_str(), "_svtk.zip", /*add_landmark*/ true);
#endif
}

//----------------------------------------------------------------------------
void svtkPythonInterpreter::SetLogVerbosity(int val)
{
  svtkPythonInterpreter::LogVerbosity = svtkLogger::ConvertToVerbosity(val);
}

//----------------------------------------------------------------------------
int svtkPythonInterpreter::GetLogVerbosity()
{
  return svtkPythonInterpreter::LogVerbosity;
}

#if !defined(SVTK_LEGACY_REMOVE)
//----------------------------------------------------------------------------
int svtkPythonInterpreter::GetPythonVerboseFlag()
{
  SVTK_LEGACY_REPLACED_BODY(
    svtkPythonInterpreter::GetPythonVerboseFlag, "SVTK 9.0", svtkPythonInterpreter::GetLogVerbosity);
  return svtkPythonInterpreter::LogVerbosity == svtkLogger::VERBOSITY_INFO ? 1 : 0;
}
#endif
