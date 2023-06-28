set(SENSEI_USE_EXTERNAL_pugixml ON CACHE BOOL "")
include(${CMAKE_CURRENT_LIST_DIR}/configure_adios2.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/configure_ascent.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/configure_catalyst.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/configure_hdf5.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/configure_vtkio.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/configure_python3.cmake)

include(${CMAKE_CURRENT_LIST_DIR}/configure_common.cmake)
