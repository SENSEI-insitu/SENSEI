set(classes
  svtkPythonInteractiveInterpreter
  svtkPythonInterpreter)

set(headers
  svtkPythonStdStreamCaptureHelper.h)

svtk_module_add_module(SVTK::PythonInterpreter
  CLASSES ${classes}
  HEADERS ${headers})
