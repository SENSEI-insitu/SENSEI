if(SENSEI_ENABLE_CUDA)

  set(tmp_nvcc FALSE)
  set(tmp_clang FALSE)
  set(tmp_nvhpc FALSE)
  set(tmp_cuda_arch "60;70;75")
  if (SENSEI_ENABLE_CUDA)
    set(tmp_have_cuda FALSE)
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC")
      set(tmp_have_cuda TRUE)
      set(tmp_cuda_arch "cc75")
      set(tmp_nvhpc TRUE)
    else()
      include(CheckLanguage)
      check_language(CUDA)
      if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(tmp_have_cuda TRUE)
        if ("${CMAKE_CUDA_COMPILER_ID}" MATCHES "Clang")
          set(tmp_cuda_arch "75")
          set(tmp_clang TRUE)
        elseif ("${CMAKE_CUDA_COMPILER_ID}" MATCHES "NVIDIA")
          set(tmp_nvcc TRUE)
        endif()
      endif()
    endif()
    find_package(CUDAToolkit REQUIRED)
  endif()

  set(SENSEI_NVCC_CUDA ${tmp_nvcc} CACHE
    STRING "Internal: set if the CUDA compiler is nvcc")

  set(SENSEI_NVHPC_CUDA ${tmp_nvhpc} CACHE
    STRING "Internal: set if the CUDA compiler is nvc++")

  set(SENSEI_CLANG_CUDA ${tmp_clang} CACHE
    STRING "Internal: set if the CUDA compiler is clang++")

  set(SENSEI_CUDA_ARCHITECTURES ${tmp_cuda_arch} CACHE
    STRING "Compile for these CUDA virtual and real architectures")

  if (SENSEI_ENABLE_CUDA)
    if (tmp_have_cuda)
      message(STATUS "SENSEI: CUDA features -- enabled (${CMAKE_CUDA_COMPILER_ID}:${SENSEI_CUDA_ARCHITECTURES})")

      set(CMAKE_CUDA_STANDARD 17)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
      set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
      set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
      set(CMAKE_CUDA_ARCHITECTURES ${SENSEI_CUDA_ARCHITECTURES})
      #set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    else()
      message(FATAL_ERROR "CUDA is required for SENSEI but was not found")
    endif()
  else()
    message(STATUS "SENSEI: CUDA features -- disabled")
  endif()

  # separate implementations
  set(tmp OFF)
  if (SENSEI_ENABLE_OPENMP AND SENSEI_ENABLE_CUDA AND NOT SENSEI_CUDA_NVHPC)
    set(tmp ON)
  endif()
  set(SENSEI_SEPARATE_IMPL ${tmp} CACHE BOOL
    "Compile to a library with explicit instantiatons for POD types")
  if (SENSEI_SEPARATE_IMPL)
    message(STATUS "SENSEI: Separate implementations -- enabled")
  else()
    message(STATUS "SENSEI: Separate implementations -- disabled")
  endif()

endif()
