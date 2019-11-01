# CTest Regression Dashboard

A number of systems have been configured for nightly testing and continuous integration.
To view the results of these runs navigate your web browser to [http://cdash.hpcvis.com/index.php?project=SENSEI]().

# Running the tests

To enable the regression tests one must configure the build with testing enabled.
```bash
cmake -DBUILD_TESTING=ON ...
```
To run the regression tests locally, from the build directory issue ctest command.
```bash
cd build_dir
ctest
```
To run the tests and submit the results to the web dashboard add the ctest track option. Note, that it is case sensitive.
```bash
ctest -DExperimental
```

# Adding regression tests using CTest

### senseiAddTest

Tests are added by calling the CMake function *senseiAddTest*. This function
encapsulates the common scenarios needed to compile, link, and run tests in
serial, and parallel; and configures CTest to flag absolute test failures
(those retrurning non-zero exit code) as well as tests that print the word
`ERROR` (case insensitive). Given that not all tests require a compile/link
step(eg Python based tests) arguments related to these steps are optional.
```CMake
function senseiAddTest(<test name>
  COMMAND <command line w/ args>
  [EXEC_NAME <exec name>]
  [SOURCES <source file 1> .. <source file n>]
  [LIBS <lib 1> .. <lib n>]
  [FEATURES <condition 1> .. < condition n>]
  [REQ_SENSEI_DATA])
```
Arguments to *senseiAddTest* are:
1. test name - a unique test name (required)
2. **COMMAND** - a command line including all arguments needed to run the test (required)
3. **EXEC_NAME** - name to compile the test executable to (optional). If not given then the test name is used. Compilation of a test executable is triggered by the presence of a list of SOURCES.
3. **SOURCES** - a list of source code files needed to compile the test executable (optional).
4. **LIBS** - a list of additional libraries needed to link the test executable (optional)
5. **FEATURES** - a list of values used in logical AND operation to determine if the test should be enabled or not. For example calling *senseiAddTest* with `FEATURES ${ENABLE_PYTHON} ${ENABLE_ADIOS1}` will enable the test only when both Python and ADIOS 1 features are enabled. Note that this argument expects *values* hence use of the CMake dereference operator $.
6. **REQ_SENSEI_DATA** - a flag that indicates SENSEI's test data repo is needed to run the test. If the data repo is not found then the test will be disabled.

### examples
#### C++, serial and parallel
Here is an example of a serial test that needs to be compiled in C++ and depends on the SENSEI test data repository for a baseline.
```CMake
senseiAddTest(testHistogramSerial
  COMMAND testHistogram EXEC_NAME testHistogram
  SOURCES testHistogram.cpp LIBS sensei)
```
Now the parallel version of the test. Note that it makes use of the above test's executable and therefore compiles nothing.
```CMake
senseiAddTest(testHistogramParallel
  COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
    ${TEST_NP} testHistogram)
```

#### Python

Here is an example of a Python test.
```CMake
senseiAddTest(testADIOSFlexpath
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS1.sh
    ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
    ${TEST_NP} ${CMAKE_CURRENT_SOURCE_DIR}
    testADIOSFlexpath.bp FLEXPATH FLEXPATH 2
  FEATURES ${ENABLE_PYTHON} ${ENABLE_ADIOS1})
```

# Setting up a new test system

Examples of setting up nightly and continuous test sites is beyond the scope of this document.
However, a few examples runs can be found at [https://github.com/burlen/SENSEI_ctest]().
