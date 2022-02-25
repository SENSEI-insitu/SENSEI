# check for gcc/clang atomic builtins like __atomic_add_fetch
if(NOT WIN32)
  if(NOT DEFINED SVTK_HAVE_ATOMIC_BUILTINS)
    message(STATUS "Checking for builtin __atomic_add_fetch")
    try_compile(SVTK_TEST_ATOMIC_BUILTINS_COMPILED
      ${CMAKE_CURRENT_BINARY_DIR}/CMakeTmp/Sync
      ${CMAKE_CURRENT_SOURCE_DIR}/svtkTestSyncBuiltins.cxx
      OUTPUT_VARIABLE OUTPUT)
    if(SVTK_TEST_ATOMIC_BUILTINS_COMPILED)
      set(svtk_atomic_add_fetch_detection "success")
      set(SVTK_HAVE_ATOMIC_BUILTINS 1)
    else()
      set(svtk_atomic_add_fetch_detection "failed")
      set(SVTK_HAVE_ATOMIC_BUILTINS 0)
    endif()
    message(STATUS "Checking for builtin __atomic_add_fetch -- ${svtk_atomic_add_fetch_detection}")
    set(SVTK_HAVE_ATOMIC_BUILTINS ${SVTK_HAVE_ATOMIC_BUILTINS}
      CACHE INTERNAL "For __atomic_ builtins.")
    file(APPEND "${CMAKE_BINARY_DIR}/CMakeFiles/CMakeOutput.log"
      "Determining if the C++ compiler supports __atomic_add_fetch builtin "
      "completed with the following output:\n"
      "${OUTPUT}\n")
  endif()
endif()
