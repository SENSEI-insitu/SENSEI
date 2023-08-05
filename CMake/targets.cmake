#
# a function to configure targets consistently
#
# EXEC          makes an executable rather than a library
# LIB           makes a library rather than an executable
# SHARED_LIB    makes a shared library
# STATIC_LIB    makes a static library
# VERBOSE       prints debug messages
# CUDA          set CUDA related options on the target
# CU_ARCH       the CUDA architectures to compile for
# CXX_SOURCES   list of C++ sources to compile
# CU_SOURCES    list of CUDA sources to compile
# LINK_LIBS     list of libraries to link to
# LINK_OPTS     list of flags for the linker
#
function(add_target)
  set(OPTS LIB SHARED_LIB STATIC_LIB EXEC CUDA VERBOSE)
  set(NVPO NAME CU_ARCH)
  set(MVO CXX_SOURCES CU_SOURCES LINK_LIBS LINK_OPTS)
  cmake_parse_arguments(PARSE_ARGV 0 TGT "${OPTS}" "${NVPO}" "${MVO}")

  set(TGT_VERBOSE ON)

  if (TGT_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unparsed arguments detected ${TGT_UNPARSED_ARGUMENTS}")
  endif()

  if (TGT_VERBOSE)
    message(STATUS "SENSEI:  + add_target : ${TGT_NAME}")
    message(STATUS "SENSEI:  ++++++++++++++++++++++++++++++++++++++++++++++ ")
  endif()

  if (TGT_CU_SOURCES)
    if (SENSEI_NVHPC_CUDA)
      set(TGT_CXX_SOURCES ${TGT_CXX_SOURCES} ${TGT_CU_SOURCES})
    else()
      set_source_files_properties(${TGT_CU_SOURCES} PROPERTIES LANGUAGE CUDA)
      if (TGT_VERBOSE)
        message(STATUS "SENSEI:  + Compiling ${TGT_CU_SOURCES} with the CUDA compiler")
      endif()
    endif()
  endif()

  if (TGT_CXX_SOURCES)
    if (TGT_VERBOSE)
      message(STATUS "SENSEI:  + Compiling ${TGT_CXX_SOURCES} with the C++ compiler")
    endif()
  endif()

  if (TGT_SHARED_LIB)
    add_library(${TGT_NAME} SHARED ${TGT_CU_SOURCES} ${TGT_CXX_SOURCES})
    if (TGT_VERBOSE)
      message(STATUS "SENSEI:  + Created shared library ${TGT_NAME}")
    endif()
  elseif (TGT_STATIC_LIB)
    add_library(${TGT_NAME} STATIC ${TGT_CU_SOURCES} ${TGT_CXX_SOURCES})
    if (TGT_VERBOSE)
      message(STATUS "SENSEI:  + Created static library ${TGT_NAME}")
    endif()
  elseif (TGT_LIB)
    add_library(${TGT_NAME} ${TGT_CU_SOURCES} ${TGT_CXX_SOURCES})
    if (TGT_VERBOSE)
      message(STATUS "SENSEI:  + Created library ${TGT_NAME}")
    endif()
  elseif (TGT_EXEC)
    add_executable(${TGT_NAME} ${TGT_CU_SOURCES} ${TGT_CXX_SOURCES})
    if (TGT_VERBOSE)
      message(STATUS "SENSEI:  + Created executable ${TGT_NAME}")
    endif()
  elseif (NOT NAME ${TGT_NAME})
    message(FATAL_ERROR "Target ${TGT_NAME} does not exist and was not created (missing LIB or EXEC).")
  endif()

  target_compile_features(${TGT_NAME} PUBLIC cxx_std_17)
  set_target_properties(${TGT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  #set_target_properties(${TGT_NAME} PROPERTIES COMPILE_OPTIONS -fvisibility=hidden)
  target_compile_options(${TGT_NAME} PUBLIC -fvisibility=hidden)
  target_compile_options(${TGT_NAME} INTERFACE -fvisibility=hidden)

  if (NOT SENSEI_NVHPC_CUDA)
    if ((TGT_CUDA OR TGT_CU_SOURCES) OR (SENSEI_ENABLE_CUDA AND SENSEI_ENABLE_OPENMP))
      if (NOT TGT_CU_ARCH)
        set(TGT_CU_ARCH "${SENSEI_CUDA_ARCHITECTURES}")
      endif()
      set_target_properties(${TGT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
      #set_target_properties(${TGT_NAME} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
      set_target_properties(${TGT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
      set_target_properties(${TGT_NAME} PROPERTIES CUDA_ARCHITECTURES "${TGT_CU_ARCH}")
      #target_link_options(${TGT_NAME} PUBLIC $<DEVICE_LINK:-fvisibility=hidden>)
      #target_link_options(${TGT_NAME} INTERFACE $<DEVICE_LINK:-fvisibility=hidden>)
      if (TGT_VERBOSE)
        message(STATUS "SENSEI:  + Configured ${TGT_NAME} for CUDA ${TGT_CU_ARCH}")
      endif()
      if (SENSEI_ENABLE_OPENMP)
        target_link_options(${TGT_NAME} PUBLIC ${SENSEI_OPENMP_LINK_FLAGS})
      endif()
   endif()
 endif()

  if (TGT_LINK_LIBS)
    target_link_libraries(${TGT_NAME} PUBLIC ${TGT_LINK_LIBS})
    if (TGT_VERBOSE)
      message(STATUS "SENSEI:  + ${TGT_NAME} links to ${TGT_LINK_LIBS}")
    endif()
  endif()

  #if (TGT_LINK_OPTS)
  #  target_link_options(${TGT_NAME} PUBLIC ${TGT_LINK_OPTS})
  #  if (TGT_VERBOSE)
  #    message(STATUS "SENSEI:  + ${TGT_NAME} link options ${TGT_LINK_OPTS}")
  #  endif()
  #endif()

  if (TGT_VERBOSE)
    message(STATUS "SENSEI:  + ")
  endif()
endfunction()
