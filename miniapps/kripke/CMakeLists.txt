set(KRIPKE_SOURCES
    Kripke/Directions.cpp
    Kripke/Grid.cpp
    Kripke/Kernel.cpp
    Kripke/Layout.cpp
    Kripke/Subdomain.cpp
    Kripke/Sweep_Solver.cpp
    Kripke/ParallelComm.cpp
    Kripke/Timing.cpp

    Kripke/Kernel/Kernel_3d_GDZ.cpp
    Kripke/Kernel/Kernel_3d_DGZ.cpp
    Kripke/Kernel/Kernel_3d_ZDG.cpp
    Kripke/Kernel/Kernel_3d_DZG.cpp
    Kripke/Kernel/Kernel_3d_ZGD.cpp
    Kripke/Kernel/Kernel_3d_GZD.cpp

    Kripke/ParallelComm/BlockJacobiComm.cpp
    Kripke/ParallelComm/SweepComm.cpp
    tools/kripke.cpp
    tools/testKernels.cpp

    bridge.cpp
)

# copy over example command line
configure_file(run_kripke_simple_example.sh
               ${CMAKE_CURRENT_BINARY_DIR}/run_kripke_simple_example.sh
               COPYONLY)

if(SENSEI_ENABLE_OPENMP)
   set(kripke_openmp_flags "-DKRIPKE_USE_OPENMP")
else()
   set(kripke_openmp_flags "")
endif()

include_directories(.)
include_directories("tools")

#set(libs sMPI sConduit sensei)
set(libs sMPI sensei)
if(SENSEI_ENABLE_OPENMP)
    list(APPEND libs openmp)
endif()

add_executable(kripke_p ${KRIPKE_SOURCES})
target_link_libraries(kripke_p ${libs})


# install target for kripke mpi
install(TARGETS kripke_p
    RUNTIME DESTINATION bin)

#    install(FILES ASCENT_README.md
#                  ascent_actions.json
#                  ascent_options.json
#            DESTINATION examples/proxies/kripke)
