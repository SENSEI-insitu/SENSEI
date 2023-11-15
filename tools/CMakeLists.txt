install(CODE "message(STATUS \"Configure and install sensei_config ... \")")
install(CODE "execute_process(WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tools/sensei_config COMMAND ./make_install ${CMAKE_INSTALL_PREFIX})")
install(CODE "message(STATUS \"Configure and install sensei_config ... done!\")")
