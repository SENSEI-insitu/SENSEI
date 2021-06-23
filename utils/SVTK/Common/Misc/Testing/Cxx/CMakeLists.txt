svtk_add_test_cxx(svtkCommonMiscCxxTests tests
  NO_DATA NO_VALID NO_OUTPUT
  TestPolygonBuilder.cxx
  TestPolygonBuilder2.cxx
  TestPolygonBuilder3.cxx
  TestPolygonBuilder4.cxx
  TestResourceFileLocator.cxx
  TestPolygonBuilder5.cxx
  UnitTestFunctionParser.cxx
  )
svtk_test_cxx_executable(svtkCommonMiscCxxTests tests)
