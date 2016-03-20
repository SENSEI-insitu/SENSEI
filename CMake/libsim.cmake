find_package(LIBSIM REQUIRED)

add_library(libsim INTERFACE)
target_link_libraries(libsim INTERFACE ${LIBSIM_STATIC_PAR_LIBRARIES})
target_include_directories(libsim SYSTEM INTERFACE ${LIBSIM_INCLUDE_DIR})
