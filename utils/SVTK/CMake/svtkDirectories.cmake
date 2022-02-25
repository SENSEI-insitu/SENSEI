include(GNUInstallDirs)

# SVTK installation structure
set(svtk_subdir "svtk-${SVTK_MAJOR_VERSION}.${SVTK_MINOR_VERSION}")
svtk_set_with_default(SVTK_INSTALL_RUNTIME_DIR "${CMAKE_INSTALL_BINDIR}")
svtk_set_with_default(SVTK_INSTALL_LIBRARY_DIR "${CMAKE_INSTALL_LIBDIR}")
svtk_set_with_default(SVTK_INSTALL_ARCHIVE_DIR "${CMAKE_INSTALL_LIBDIR}")
svtk_set_with_default(SVTK_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/${svtk_subdir}")
svtk_set_with_default(SVTK_INSTALL_DATA_DIR "${CMAKE_INSTALL_DATADIR}/${svtk_subdir}")
# CMAKE_INSTALL_DOCDIR already includes PROJECT_NAME, which is not what we want
svtk_set_with_default(SVTK_INSTALL_DOC_DIR "${CMAKE_INSTALL_DATAROOTDIR}/doc/${svtk_subdir}")
svtk_set_with_default(SVTK_INSTALL_PACKAGE_DIR "${SVTK_INSTALL_LIBRARY_DIR}/cmake/${svtk_subdir}")
svtk_set_with_default(SVTK_INSTALL_DOXYGEN_DIR "${SVTK_INSTALL_DOC_DIR}/doxygen")
svtk_set_with_default(SVTK_INSTALL_NDK_MODULES_DIR "${SVTK_INSTALL_DATA_DIR}/ndk-modules")
