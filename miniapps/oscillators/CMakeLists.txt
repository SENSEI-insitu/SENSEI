set(sources oscillator.cpp analysis.cpp)

set(libs mpi diy grid opts thread timer util)

if(ENABLE_SENSEI)
  list(APPEND sources dataadaptor.cpp bridge.cpp)
  list(APPEND libs sensei)
endif()

add_library(util STATIC src/format.cc)
target_include_directories(util SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(oscillator ${sources})
target_link_libraries(oscillator ${libs})
