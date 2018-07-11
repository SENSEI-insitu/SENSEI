set(sources conduitTest.cpp bridge.cpp)
set(libs mpi diy grid opts thread timer util sconduit sensei)

add_executable(conduitTest ${sources})
target_link_libraries(conduitTest ${libs})
