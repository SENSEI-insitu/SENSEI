set(classes
  svtkRenderTimings)

svtk_module_add_module(SVTK::UtilitiesBenchmarks
  CLASSES ${classes})

# Add our test executables.
svtk_module_add_executable(TimingTests
  NO_INSTALL
  TimingTests.cxx)
target_link_libraries(TimingTests
  PRIVATE
    SVTK::UtilitiesBenchmarks)

svtk_module_autoinit(
  TARGETS TimingTests
  MODULES SVTK::UtilitiesBenchmarks)

svtk_module_add_executable(GLBenchmarking
  NO_INSTALL
  GLBenchmarking.cxx)
target_link_libraries(GLBenchmarking
  PRIVATE
    SVTK::ChartsCore
    SVTK::UtilitiesBenchmarks
    SVTK::ViewsContext2D)

svtk_module_autoinit(
  TARGETS GLBenchmarking
  MODULES SVTK::ChartsCore
          SVTK::UtilitiesBenchmarks
          SVTK::ViewsContext2D)
