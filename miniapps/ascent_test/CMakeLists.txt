set(libs sMPI sAscent sensei)

# create our example 
add_executable(ascent_render_example ascent_render_example.cpp)

# link to ascent
target_link_libraries(ascent_render_example ${libs})

# install to bin
install(TARGETS ascent_render_example RUNTIME DESTINATION bin)
