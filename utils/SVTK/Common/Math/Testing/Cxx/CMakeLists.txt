svtk_add_test_cxx(svtkCommonMathCxxTests tests
  NO_DATA NO_VALID NO_OUTPUT
  TestAmoebaMinimizer.cxx
  TestMatrix3x3.cxx
  TestPolynomialSolversUnivariate.cxx
  TestQuaternion.cxx
  )
svtk_test_cxx_executable(svtkCommonMathCxxTests tests)
