project(oscillators)

set(OSCILLATOR_CUDA OFF CACHE BOOL "Enable CUDA acceleration in the miniapp")
if (OSCILLATOR_CUDA)
  include(CheckLanguage)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "oscillators miniapp will use CUDA")
  else()
    message(FATAL_ERROR "CUDA is required but was not found")
  endif()
endif()

set(sources main.cpp analysis.cpp)
set(libs sMPI sDIY sOPTS thread)

if(ENABLE_SENSEI)
  list(APPEND sources bridge.cpp DataAdaptor.cpp
    Oscillator.cpp Particles.cpp Block.cpp BlockInternals.cpp)

  list(APPEND libs sensei)
endif()

if (OSCILLATOR_CUDA)
  set_source_files_properties(main.cpp
    Block.cpp BlockInternals.cpp Oscillator.cpp
    PROPERTIES LANGUAGE CUDA)
endif()

add_executable(oscillator ${sources})
target_link_libraries(oscillator ${libs})

if (OSCILLATOR_CUDA)
  target_compile_definitions(oscillator PUBLIC OSCILLATOR_CUDA)
  target_compile_features(oscillator PUBLIC cxx_std_17)
  set_target_properties(oscillator PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(oscillator PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

target_include_directories(oscillator SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(TARGETS oscillator RUNTIME DESTINATION bin)

add_subdirectory(testing)
