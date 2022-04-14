#include "PythonAnalysis.h"
#include "DataAdaptor.h"
#include "Error.h"

#include <svtkObjectFactory.h>
#include <mpi4py/mpi4py.MPI_api.h>
#include <string>
#include <cstdio>
#include <cstring>
#include <errno.h>
#include <Python.h>

#include "senseiPyString.h"

// Macro to report error through sensei's normal mechanism
// and include Python exception info and stack
#define SENSEI_PYTHON_ERROR(_msg)                       \
{                                                       \
SENSEI_ERROR(_msg)                                      \
cerr << "====================================="         \
  "==========================================" << endl; \
PyErr_Print();                                          \
cerr << "====================================="         \
  "==========================================" << endl; \
}

// get a handle to the named function in the given module
static
int getFunction(PyObject *module, const std::string &modName,
  const std::string &funcName, bool required, PyObject *&func)
{
  func = nullptr;

  PyObject *f = PyObject_GetAttrString(module, funcName.c_str());

  if (!required && !f)
    {
    PyErr_Clear();
    return 0;
    }

  if (!f)
    {
    SENSEI_ERROR("Module \"" << modName
      << "\" has no function named \"" << funcName << "\"")
    return -1;
    }

  if (!PyCallable_Check(f))
    {
    SENSEI_ERROR("\"" << funcName << "\" in module \""
      << modName << "\" is not a callable")
    return -1;
    }

  func = f;

  return 0;
}

// call the function with the given arguments
static
int callFunction(const std::string &funcName, PyObject *func, PyObject *args)
{
  PyObject *pyRet = PyObject_CallObject(func, args);
  if (!pyRet || PyErr_Occurred())
    {
    SENSEI_PYTHON_ERROR("An error ocurred in call to \"" << funcName << "\"")
    return -1;
    }

  Py_XDECREF(pyRet);

  return 0;
}

// call the function with the given arguments
static
int callFunction(const std::string &funcName,
  PyObject *func, PyObject *args, PyObject *ret)
{
  ret = PyObject_CallObject(func, args);
  if (!ret || PyErr_Occurred())
    {
    SENSEI_PYTHON_ERROR("An error ocurred in call to \"" << funcName << "\"")
    return -1;
    }
  return 0;
}

// run the python code in the given string, in the given
// module's scope.
static
int runString(PyObject *module, const std::string &code)
{
  PyObject *globals = PyModule_GetDict(module);
  PyObject *locals = globals;

  PyObject *pyRet = PyRun_String(code.c_str(), Py_file_input, globals, locals);

  if (!pyRet || PyErr_Occurred())
    {
    SENSEI_PYTHON_ERROR("An error ocurred when running the string \"" << code  << "\"")
    return -1;
    }

  Py_XDECREF(pyRet);

  return 0;
}

static
int loadScript(MPI_Comm comm, const std::string &scriptFile, PyObject *&module)
{
  // read and broadcast the script
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  char *script = nullptr;
  long scriptLen = 0;

  if (rank == 0)
    {
    FILE *f = fopen(scriptFile.c_str(), "r");
    if (!f)
      {
      const char *estr = strerror(errno);
      SENSEI_ERROR("Failed to open \"" << scriptFile << "\""
        << std::endl << estr)
      scriptLen = -1;
      MPI_Bcast(&scriptLen, 1, MPI_LONG, 0, comm);
      return -1;
      }

    fseek(f, 0, SEEK_END);
    scriptLen = ftell(f);
    fseek(f, 0, SEEK_SET);

    script = static_cast<char*>(malloc(scriptLen+1));
    script[scriptLen] = '\0';

    long nrd = fread(script, 1, scriptLen, f);

    fclose(f);

    if (nrd != scriptLen)
      {
      const char *estr = strerror(errno);
      SENSEI_ERROR("Failed to read \"" << scriptFile << "\""
        << std::endl << estr)
      scriptLen = -1;
      MPI_Bcast(&scriptLen, 1, MPI_LONG, 0, comm);
      free(script);
      return -1;
      }

    MPI_Bcast(&scriptLen, 1, MPI_LONG, 0, comm);
    MPI_Bcast(script, scriptLen, MPI_CHAR, 0, comm);
    }
  else
    {
    MPI_Bcast(&scriptLen, 1, MPI_LONG, 0, comm);

    if (scriptLen < 1)
      return -1;

    script = static_cast<char*>(malloc(scriptLen+1));
    script[scriptLen] = '\0';

    MPI_Bcast(script, scriptLen, MPI_CHAR, 0, comm);
    }

  // this does some internal initialization
  module = PyImport_AddModule("__main__");
  Py_INCREF(module);

  PyModule_AddStringConstant(module, "__file__", scriptFile.c_str());

  if (runString(module, script))
    {
    SENSEI_ERROR("Failed to import the script \"" << scriptFile << "\"")
    free(script);
    return -1;
    }

  free(script);
  return 0;
}

namespace sensei
{

struct PythonAnalysis::InternalsType
{
  InternalsType() : Module(nullptr), Initialize(nullptr),
    Execute(nullptr), Finalize(nullptr) {}

  ~InternalsType();

  std::string ScriptModule;
  std::string ScriptFile;
  std::string InitializeSource;

  PyObject *Module;
  PyObject *Initialize;
  PyObject *Execute;
  PyObject *Finalize;
};

//-----------------------------------------------------------------------------
PythonAnalysis::InternalsType::~InternalsType()
{
  if (this->Initialize || this->Execute || this->Finalize || this->Module)
    SENSEI_ERROR("PythonAnalysis::Finalize not called")
}


//-----------------------------------------------------------------------------
senseiNewMacro(PythonAnalysis);

//-----------------------------------------------------------------------------
PythonAnalysis::PythonAnalysis() : Internals(nullptr)
{
  this->Internals = new InternalsType;
}

//-----------------------------------------------------------------------------
PythonAnalysis::~PythonAnalysis()
{
  delete this->Internals;
}

//-----------------------------------------------------------------------------
void PythonAnalysis::SetInitializeSource(const std::string &source)
{
  this->Internals->InitializeSource = source;
}

//-----------------------------------------------------------------------------
void PythonAnalysis::SetScriptModule(const std::string &moduleName)
{
  this->Internals->ScriptModule = moduleName;
}

//-----------------------------------------------------------------------------
void PythonAnalysis::SetScriptFile(const std::string &scriptName)
{
  this->Internals->ScriptFile = scriptName;
}

//-----------------------------------------------------------------------------
int PythonAnalysis::Finalize()
{
  if (this->Internals->Finalize)
    callFunction("Finalize", this->Internals->Finalize, nullptr);

  Py_XDECREF(this->Internals->Initialize);
  Py_XDECREF(this->Internals->Execute);
  Py_XDECREF(this->Internals->Finalize);
  Py_XDECREF(this->Internals->Module);

  this->Internals->Initialize = nullptr;
  this->Internals->Execute = nullptr;
  this->Internals->Finalize = nullptr;
  this->Internals->Module = nullptr;

  Py_Finalize();

  return 0;
}

//-----------------------------------------------------------------------------
int PythonAnalysis::Initialize()
{
  // initialize the interpreter
  Py_SetProgramName(C_STRING_LITERAL("PythonAnalysis"));
  Py_Initialize();

  if (!this->Internals->ScriptFile.empty() && !this->Internals->ScriptModule.empty())
    {
    SENSEI_ERROR("Both a script file and script module were provided. "
      "You must provide either a script module or a script file not both")
    return -1;
    }

  if (this->Internals->ScriptFile.empty() && this->Internals->ScriptModule.empty())
    {
    SENSEI_ERROR("Neither a script file nor script module were provided. "
      "You must provide either a script file or script module")
    return -1;
    }

  if (!this->Internals->ScriptFile.empty())
    {
    // read, boradcast, and run the script
    if (loadScript(this->GetCommunicator(), this->Internals->ScriptFile,
      this->Internals->Module))
      return -1;
    }
  else
    {
    // import the script
    PyObject *module = PyImport_ImportModule(this->Internals->ScriptModule.c_str());

    if (!module || PyErr_Occurred())
      {
      SENSEI_PYTHON_ERROR("Failed to import module \""
        << this->Internals->ScriptModule  << "\"")
      return -1;
      }

    this->Internals->Module = module;
    }

  // look for AnalysisAdaptor API
  int ierr = getFunction(this->Internals->Module,
    this->Internals->ScriptModule, "Initialize", false,
    this->Internals->Initialize);

  ierr += getFunction(this->Internals->Module,
    this->Internals->ScriptModule, "Execute", true,
    this->Internals->Execute);

  ierr += getFunction(this->Internals->Module,
    this->Internals->ScriptModule, "Finalize", false,
    this->Internals->Finalize);

  if (ierr)
    {
    SENSEI_ERROR("Module \"" << this->Internals->ScriptModule <<
      "\" does not provide the required API. The API consists of the "
      "following functions defined at global scope:\n\n    Initialize() -> int\n"
      "    Execute(dataAdaptor) -> int\n    Finalize() -> int\n\nOnly Execute is "
      "required, Initialize and Finalize are optional.")
    return -1;
    }

  // import the sensei wrapper and mpi4py
  if (runString(this->Internals->Module,
    "from mpi4py import *\n"
    "from sensei.PythonAnalysis import *\n"))
    {
    SENSEI_ERROR("Failed to import baseline modules")
    return -1;
    }

  // set the communicator
  PyModule_AddObject(this->Internals->Module,
    "comm", PyMPIComm_New(this->GetCommunicator()));

  // set provided globals
  if (!this->Internals->InitializeSource.empty())
    {
    if (runString(this->Internals->Module, this->Internals->InitializeSource))
      {
      SENSEI_ERROR("Failed to run initialize source")
      return -1;
      }
    }

  // call the provided initialize function
  if (this->Internals->Initialize)
    return callFunction("Initialize", this->Internals->Initialize, nullptr);

  return 0;
}

//-----------------------------------------------------------------------------
bool PythonAnalysis::Execute(DataAdaptor *daIn, DataAdaptor **daOut)
{
  // start off by indicating no return. if we have one, then correct this
  if (daOut)
    {
    *daOut = nullptr;
    }

  if (!this->Internals->Execute)
    {
    SENSEI_ERROR("Missing an Execute function")
    return false;
    }

  // wrap the data adaptor instance
  PyObject *pyDataAdaptor = SWIG_NewPointerObj(
    SWIG_as_voidptr(daIn), SWIGTYPE_p_sensei__DataAdaptor, 0);

  // the tuple takes owner ship with N
  PyObject *args = Py_BuildValue("(N)", pyDataAdaptor);
  PyObject *ret = nullptr;

  // invoke the provided execute function
  if (callFunction("Execute", this->Internals->Execute, args, ret))
    {
    Py_DECREF(args);
    return false;
    }

  // clean up function arguments
  Py_DECREF(args);

  // the user provided function could return one of four possible things:
  // 1. None
  // 2. an integer status code
  // 3. a data adaptor instance
  // 4. a tuple continaing an integer status code and a data adaptor instance
  int status = 1;
  if (ret && (ret != Py_None))
    {
    PyObject *pyDaOut = ret;
    if (PyLong_Check(ret))
      {
      // status code
      status = PyLong_AsLong(ret);
      Py_DECREF(ret);
      return status;
      }
    else if (PyTuple_Check(ret) && (PyTuple_Size(ret) == 2))
      {
      if ((PyTuple_Size(ret) != 2) && !PyLong_Check(PyTuple_GetItem(ret, 0)))
        {
        SENSEI_PYTHON_ERROR("Bad tuple returned from Execute")
        PyObject_Print(ret, stderr, Py_PRINT_RAW);
        Py_DECREF(ret);
        return false;
        }
      // status code
      status = PyLong_AsLong(PyTuple_GetItem(ret, 0));
      // data adaptor
      pyDaOut = PyTuple_GetItem(ret, 1);
      }

    // a data adaptor was returned
    int newmem = 0;
    void *tmpvp = nullptr;
    int ierr = SWIG_ConvertPtrAndOwn(pyDaOut,
      &tmpvp, SWIGTYPE_p_sensei__DataAdaptor, 0, &newmem);

    if (ierr == SWIG_ERROR)
      {
      SENSEI_PYTHON_ERROR("Execute returned an invalid DataAdaptor")
      PyObject_Print(ret, stderr, Py_PRINT_RAW);
      Py_DECREF(ret);
      return false;
      }

    /* newmem = 1 SWIG_CAST_NEW_MEMORY = 2 tmpvp = 0x557e358ea080
    std::cerr << "newmem = " << newmem << " SWIG_CAST_NEW_MEMORY = "
       << SWIG_CAST_NEW_MEMORY  << " tmpvp = " << tmpvp << std::endl;*/

    // capture the return and take a reference
    if (tmpvp)
      {
      *daOut = reinterpret_cast<sensei::DataAdaptor*>(tmpvp);
      (*daOut)->Register(nullptr);
      }

    // clean up memory allocated by SWIG
    if (newmem & SWIG_CAST_NEW_MEMORY)
      {
      (*daOut)->Delete();
      }
    }

  // clean up
  Py_XDECREF(ret);

  return true;
}

}
