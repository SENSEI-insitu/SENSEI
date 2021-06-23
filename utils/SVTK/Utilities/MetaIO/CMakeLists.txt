set(METAIO_FOR_SVTK 1)
set(SVTK_ZLIB_LIBRARIES svtkzlib)
if(SVTK_REQUIRE_LARGE_FILE_SUPPORT)
  add_definitions(
    -D_LARGEFILE_SOURCE
    -D_LARGEFILE64_SOURCE
    -D_LARGE_FILES
    -D_FILE_OFFSET_BITS=64)
endif()
# We never want MetaIO's tests.
set(BUILD_TESTING OFF)

# FIXME: MetaIO only exports symbols properly on Windows.
unset(CMAKE_CXX_VISIBILITY_PRESET)
unset(CMAKE_VISIBILITY_INLINES_HIDDEN)

svtk_module_third_party_internal(
  LICENSE_FILES "svtkmetaio/License.txt"
  VERSION       "master"
  STANDARD_INCLUDE_DIRS)
