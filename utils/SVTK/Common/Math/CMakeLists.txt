set(classes
  svtkAmoebaMinimizer
  svtkFunctionSet
  svtkInitialValueProblemSolver
  svtkMatrix3x3
  svtkMatrix4x4
  svtkPolynomialSolversUnivariate
  svtkQuaternionInterpolator
  svtkRungeKutta2
  svtkRungeKutta4
  svtkRungeKutta45)

set(template_classes
  svtkQuaternion)

set(headers
  svtkTuple.h)

if(CMAKE_COMPILER_IS_GNUCXX)
  set_source_files_properties(svtkPolynomialSolversUnivariate.cxx
    PROPERTIES
      COMPILE_FLAGS -O0)
endif()

svtk_module_add_module(SVTK::CommonMath
  CLASSES           ${classes}
  TEMPLATE_CLASSES  ${template_classes}
  HEADERS           ${headers})
