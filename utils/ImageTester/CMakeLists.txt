#------------------------------------------------------------------------------
# Generate the image compare tool.

include(vtk)
add_executable(CompareImages CompareImages.cxx)
target_link_libraries(CompareImages vtk vtkCommonDataModel)
