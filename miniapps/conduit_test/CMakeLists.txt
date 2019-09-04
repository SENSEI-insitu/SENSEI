set(sources conduitTest.cpp bridge.cpp)
#set(libs opts thread timer util sensei sConduit sDIY sMPI)
set(libs sOPTS thread timer util sensei sMPI)

add_executable(conduitTest ${sources})
target_link_libraries(conduitTest ${libs})
