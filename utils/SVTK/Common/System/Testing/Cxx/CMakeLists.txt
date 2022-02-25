svtk_add_test_cxx(svtkCommonSystemCxxTests tests
  NO_DATA NO_VALID NO_OUTPUT
  TestDirectory.cxx
  otherTimerLog.cxx
  )
svtk_test_cxx_executable(svtkCommonSystemCxxTests tests)
