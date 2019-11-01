set(sources main.cpp analysis.cpp)

set(libs sMPI sDIY opts thread util)

if(ENABLE_SENSEI)
  list(APPEND sources bridge.cpp DataAdaptor.cpp Oscillator.cpp Particles.cpp Block.cpp)
  list(APPEND libs sensei)
endif()

add_library(util STATIC src/format.cc)
target_include_directories(util SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(oscillator ${sources})
target_link_libraries(oscillator ${libs})

install(TARGETS oscillator
  RUNTIME DESTINATION bin)

add_subdirectory(testing)
