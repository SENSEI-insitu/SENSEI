svtk_add_test_cxx(svtkCommonTransformsCxxTests tests
  NO_DATA NO_VALID NO_OUTPUT
  TestTransform.cxx
  TestLandmarkTransform.cxx
  TestThinPlateSplineTransform.cxx
  )
svtk_test_cxx_executable(svtkCommonTransformsCxxTests tests)
