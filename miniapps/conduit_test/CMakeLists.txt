set(sources conduitTest.cpp bridge.cpp)
set(libs sOPTS thread util sensei sConduit sMPI)

add_executable(conduitTest ${sources})
target_link_libraries(conduitTest ${libs})

install(TARGETS conduitTest RUNTIME DESTINATION bin)
