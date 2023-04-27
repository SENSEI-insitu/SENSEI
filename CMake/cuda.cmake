if(SENSEI_ENABLE_CUDA)

  set(SENSEI_CUDA_ARCHITECTURES "60;70;75" CACHE
    STRING "Compile for these CUDA virtual and real architectures")

  include(CheckLanguage)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${SENSEI_CUDA_ARCHITECTURES}")
  endif()
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "CUDA is required by SENSEI_ENABLE_CUDA but CUDA was not found")
  endif()

  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11")
    message(FATAL_ERROR "CUDA >= 11 is required by SENSEI_ENABLE_CUDA")
  endif ()

  # Use this function to make sure targets and sources compile on CUDA.
  # arguments:

  # TARGET - the name of the target to use CUDA with
  # SOURCES - an optional list of source files that need to be compiled with the
  #           CUDA compiler
  function(sensei_cuda_target)
    set(OPTS "")
    set(NVPO TARGET)
    set(MVO SOURCES)
    cmake_parse_arguments(PARSE_ARGV 0 CUDA_TGT "${OPTS}" "${NVPO}" "${MVO}")

    message(STATUS "SENSEI: Created CUDA target ${CUDA_TGT_TARGET}")

    target_compile_features(${CUDA_TGT_TARGET} PUBLIC cxx_std_17)

    set_target_properties(${CUDA_TGT_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${CUDA_TGT_TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    #set_target_properties(${CUDA_TGT_TARGET} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    #set_target_properties(${CUDA_TGT_TARGET} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_target_properties(${CUDA_TGT_TARGET} PROPERTIES CUDA_ARCHITECTURES "${SENSEI_CUDA_ARCHITECTURES}")

    if (CUDA_TGT_SOURCES)
       message(STATUS "SENSEI: Compiling ${CUDA_TGT_SOURCES} with the CUDA compiler for ${SENSEI_CUDA_ARCHITECTURES}")
       #set_source_files_properties(${CUDA_TGT_SOURCES} PROPERTIES LANGUAGE CUDA)
    endif()
  endfunction()

endif()
