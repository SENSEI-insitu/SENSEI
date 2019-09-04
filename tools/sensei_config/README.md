# `sensei_config` #
A command line tool for exporting SENSEI's installed configuration
to non-cmake based projects.

## Install ##
Point CMake to the SENSEI install, the CMake interface libraries are parsed and
the script is generated. CMake has no notion of a post install command so this
must be run a secondary step after the SENSEI install

```bash
$ cmake -DSENSEI_DIR=~/sc17/software/sensei/2.0.0-catalyst/lib/cmake/ \
    -DCMAKE_INSTALL_PREFIX=~/sc17/software/sensei/2.0.0-catalyst/
     ..
$ make install
```

## Use ##
Print the flags and link libraries needed to build against a SENSEI install
```bash
$ sensei_config --cflags --libs
```
Bring `SENSEI_INCLUDES` and `SENSEI_LIBRARIES` into the environment
```bash
$ source sensei_config
```
