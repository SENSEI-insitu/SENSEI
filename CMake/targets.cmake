#
# a function to configure targets consistently and report in the terminal=
#
function(add_target)
  set(OPTS LIB SHARED_LIB STATIC_LIB EXEC CUDA)
  set(NVPO NAME CU_ARCH)
  set(MVO CXX_SOURCES CU_SOURCES LINK_LIBS)
  cmake_parse_arguments(PARSE_ARGV 0 TGT "${OPTS}" "${NVPO}" "${MVO}")

  if (TGT_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unparsed arguments detected ${TGT_UNPARSED_ARGUMENTS}")
  endif()

  message(STATUS "SENSEI:  + add_target ")
  message(STATUS "SENSEI:  ++++++++++++++++++++++++++++++++++++++++++++++ ")

  if (TGT_CU_SOURCES)
    message(STATUS "SENSEI:  + Compiling ${TGT_CU_SOURCES} with the CUDA compiler")
    set_source_files_properties(${TGT_CU_SOURCES} PROPERTIES LANGUAGE CUDA)
  endif()

  if (TGT_CXX_SOURCES)
    message(STATUS "SENSEI:  + Compiling ${TGT_CXX_SOURCES} with the C++ compiler")
  endif()

  if (TGT_SHARED_LIB)
    add_library(${TGT_NAME} SHARED ${TGT_CU_SOURCES} ${TGT_CXX_SOURCES})
    message(STATUS "SENSEI:  + Created shared library ${TGT_NAME}")
  elseif (TGT_STATIC_LIB)
    add_library(${TGT_NAME} STATIC ${TGT_CU_SOURCES} ${TGT_CXX_SOURCES})
    message(STATUS "SENSEI:  + Created static library ${TGT_NAME}")
  elseif (TGT_LIB)
    add_library(${TGT_NAME} ${TGT_CU_SOURCES} ${TGT_CXX_SOURCES})
    message(STATUS "SENSEI:  + Created library ${TGT_NAME}")
  elseif (TGT_EXEC)
    add_executable(${TGT_NAME} ${TGT_CU_SOURCES} ${TGT_CXX_SOURCES})
    message(STATUS "SENSEI:  + Created executable ${TGT_NAME}")
  elseif (NOT NAME ${TGT_NAME})
    message(FATAL_ERROR "Target ${TGT_NAME} does not exist and was not created (missing LIB or EXEC).")
  endif()

  target_compile_features(${TGT_NAME} PUBLIC cxx_std_17)
  set_target_properties(${TGT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  #set_target_properties(${TGT_NAME} PROPERTIES COMPILE_OPTIONS -fvisibility=hidden)
  target_compile_options(${TGT_NAME} PUBLIC -fvisibility=hidden)
  target_compile_options(${TGT_NAME} INTERFACE -fvisibility=hidden)

  if (TGT_CUDA OR TGT_CU_SOURCES)
    if (NOT TGT_CU_ARCH)
      set(TGT_CU_ARCH "60;70;75")
    endif()
    set_target_properties(${TGT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    #set_target_properties(${TGT_NAME} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    #set_target_properties(${TGT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_target_properties(${TGT_NAME} PROPERTIES CUDA_ARCHITECTURES "${TGT_CU_ARCH}")
    #target_link_options(${TGT_NAME} PUBLIC $<DEVICE_LINK:-fvisibility=hidden>)
    #target_link_options(${TGT_NAME} INTERFACE $<DEVICE_LINK:-fvisibility=hidden>)

    message(STATUS "SENSEI:  + Configured ${TGT_NAME} for CUDA ${TGT_CU_ARCH}")
 endif()

  if (TGT_LINK_LIBS)
    target_link_libraries(${TGT_NAME} PUBLIC ${TGT_LINK_LIBS})
    message(STATUS "SENSEI:  + ${TGT_NAME} links to ${TGT_LINK_LIBS}")
  endif()

  message(STATUS "SENSEI:  + ")
endfunction()

