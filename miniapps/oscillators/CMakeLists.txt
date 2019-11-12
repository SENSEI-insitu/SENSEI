set(sources main.cpp analysis.cpp)

set(libs sMPI sDIY sOPTS thread)

if(ENABLE_SENSEI)
  list(APPEND sources bridge.cpp DataAdaptor.cpp Oscillator.cpp Particles.cpp Block.cpp)
  list(APPEND libs sensei)
endif()

add_executable(oscillator ${sources})
target_link_libraries(oscillator ${libs})

target_include_directories(oscillator SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(TARGETS oscillator
  RUNTIME DESTINATION bin)

add_subdirectory(testing)
