## mangle_vtk.sh
A script to mangle a copy of the officially released VTK library.
Mangles files, folders and classes from `vtk` to `svtk` and from `VTK` to `SVTK`.

### Compiling Mangled Sources
Modify your `-DVTK_X` CMake options to be `-DSVTK_X`. Things should compile as normal.

### Known issues
Wrapping should be disabled when compiling the mangled code due to crashes in
VTK wrpping codes.



