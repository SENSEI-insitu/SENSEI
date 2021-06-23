svtk_module_test_data(
  Data/Disc_BiQuadraticQuads_0_0.vtu
  Data/Disc_QuadraticQuads_0_0.vtu
  Data/blowGeom.svtk
  Data/headsq/,REGEX:.*
  Data/nonWatertightPolyhedron.vtu
  Data/quadraticTetra01.vtu)

add_subdirectory(Cxx)

if (SVTK_WRAP_PYTHON)
  add_subdirectory(Python)
endif ()
