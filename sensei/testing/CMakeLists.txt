if (BUILD_TESTING)

  ##############################################################################
  senseiAddTest(testHistogramSerial
    SOURCES testHistogram.cpp LIBS sensei EXEC_NAME testHistogram
    COMMAND $<TARGET_FILE:testHistogram>
    CUDA_TARGET
    LABELS HISTO)

  senseiAddTest(testHistogramParallel
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:testHistogram>
    PROPERTIES
      LABELS HISTO)

  ##############################################################################
  senseiAddTest(testParticleDensitySerial
    SOURCES testParticleDensity.cpp
    LIBS sensei EXEC_NAME testParticleDensity
    COMMAND $<TARGET_FILE:testParticleDensity> 20000 xy 128 ./
    CUDA_TARGET
    )

  senseiAddTest(testParticleDensityParallel
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:testParticleDensity> 20000 xy 128 ./
    CUDA_TARGET
    )

  ##############################################################################
  senseiAddTest(testDataBinningHostSync
    SOURCES testDataBinning.cpp
    LIBS sensei EXEC_NAME testDataBinning
    COMMAND $<TARGET_FILE:testDataBinning> 32 100000 128 ./ -1 0 xpos ypos sum min max avg
    CUDA_TARGET
    )

  senseiAddTest(testDataBinningHostAsync
    COMMAND $<TARGET_FILE:testDataBinning> 32 100000 128 ./ -1 1 xpos ypos sum min max avg
    )

  senseiAddTest(testDataBinningHostSyncParallel
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:testDataBinning> 32 100000 128 ./ -1 0 xpos ypos sum min max avg
    )

  senseiAddTest(testDataBinningHostAsyncParallel
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:testDataBinning> 32 100000 128 ./ -1 1 xpos ypos sum min max avg
    )

  if (SENSEI_ENABLE_CUDA)
    senseiAddTest(testDataBinningCUDAAsync
      COMMAND $<TARGET_FILE:testDataBinning> 32 100000 128 ./ 0 1 xpos ypos sum min max avg
      )
    senseiAddTest(testDataBinningCUDASync
      COMMAND $<TARGET_FILE:testDataBinning> 32 100000 128 ./ 0 0 xpos ypos sum min max avg
      )
    senseiAddTest(testDataBinningCUDAAsyncParallel
      PARALLEL ${TEST_NP}
      COMMAND $<TARGET_FILE:testDataBinning> 32 100000 128 ./ 0 1 xpos ypos sum min max avg
      )
    senseiAddTest(testDataBinningCUDASyncParallel
      PARALLEL ${TEST_NP}
      COMMAND $<TARGET_FILE:testDataBinning> 32 100000 128 ./ 0 0 xpos ypos sum min max avg
      )
  endif()


  ##############################################################################
  senseiAddTest(testHDF5Write
    SOURCES testHDF5.cpp LIBS sensei EXEC_NAME testHDF5
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:testHDF5> w 4 n h5test
    FEATURES HDF5
    PROPERTIES
      FIXTURES_SETUP HDF5_IO)

  senseiAddTest(testHDF5Read
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:testHDF5> r h5test.n${TEST_NP}
    FEATURES HDF5
    PROPERTIES
      FIXTURES_REQUIRED HDF5_IO)

  ##############################################################################
  senseiAddTest(testHDF5WriteStreaming
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:testHDF5> w 4 s h5stream
    FEATURES HDF5
    PROPERTIES
      LABELS STREAMING
      FIXTURES_SETUP HDF5_STREAMING)

  senseiAddTest(testHDF5ReadStreaming
    PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:testHDF5> r h5stream.n${TEST_NP} s
    FEATURES HDF5
    PROPERTIES
      FIXTURES_REQUIRED HDF5_STREAMING
      LABELS STREAMING)

  ##############################################################################
#   senseiAddTest(testProgrammableDataAdaptor
#     PARALLEL 1
#     COMMAND $<TARGET_FILE:testProgrammableDataAdaptor>
#     SOURCES testProgrammableDataAdaptor.cpp
#     LIBS sensei)
# 
#   senseiAddTest(testProgrammableDataAdaptorPy
#     PARALLEL 1
#     COMMAND ${PYTHON_EXECUTABLE}
#       ${CMAKE_CURRENT_SOURCE_DIR}/testProgrammableDataAdaptor.py
#     FEATURES PYTHON)

  ##############################################################################
  senseiAddTest(testPythonAnalysis
    SOURCES simpleTestDriver.cpp LIBS sensei EXEC_NAME simpleTestDriver
    COMMAND $<TARGET_FILE:simpleTestDriver>
      ${CMAKE_CURRENT_SOURCE_DIR}/testPythonAnalysis.xml)

  senseiAddTest(testPythonAnalysisParallel PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:simpleTestDriver>
      ${CMAKE_CURRENT_SOURCE_DIR}/testPythonAnalysis.xml)

  ##############################################################################
  senseiAddTest(testVTKPosthocIO
    COMMAND $<TARGET_FILE:simpleTestDriver>
      ${CMAKE_CURRENT_SOURCE_DIR}/testVTKPosthocIO.xml
    FEATURES PYTHON VTK_IO)

  senseiAddTest(testVTKPosthocIOParallel PARALLEL ${TEST_NP}
    COMMAND $<TARGET_FILE:simpleTestDriver>
      ${CMAKE_CURRENT_SOURCE_DIR}/testVTKPosthocIO.xml
    FEATURES PYTHON VTK_IO)

  ##############################################################################
  senseiAddTest(testPartitionerPy
    COMMAND
      ${PYTHON_EXECUTABLE} -m mpi4py ${CMAKE_CURRENT_SOURCE_DIR}/testPartitioner.py 13
    FEATURES PYTHON)

  ##############################################################################
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/catalyst_render_partition.xml.in
      ${CMAKE_CURRENT_BINARY_DIR}/catalyst_render_partition.xml  @ONLY)

  senseiAddTest(testPartitionersADIOS1BP
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitionersDriver.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
      ${TEST_NP_HALF} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} adios1_bp 3 3 3 0
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON ADIOS1 CATALYST)

  # in adios 1.13.1 flexpath has a limitation of 1 sender block per
  # MPI rank. there's also a deadlock if some blocks are not sent.
  # hence this test current has to run with M=9 N=2
  senseiAddTest(testPartitionersADIOS1FLEXPATH
    PARALLEL_SHELL 9
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitionersDriver.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 9 2
      ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} adios1_flexpath 3 3 3 1
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON ADIOS1 CATALYST)

  senseiAddTest(testADIOS1Flexpath
    PARALLEL_SHELL ${TEST_NP}
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS1.sh
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
      ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
      testADIOS1Flexpath.bp FLEXPATH FLEXPATH 3
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON ADIOS1)

  senseiAddTest(testADIOS1MPIBP
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS1.sh
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
      ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
      testADIOS1MPIBP.bp MPI BP 3
    FEATURES
      PYTHON ADIOS1)

  ##############################################################################
  senseiAddTest(testADIOS2BP4
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS2.sh
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
      ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
      testADIOS2BP4_%05d.bp BP4 BP4 3
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON ADIOS2)

  senseiAddTest(testADIOS2SST
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testADIOS2.sh
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
      ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}
      testADIOS2SST.bp SST SST 3
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON ADIOS2)

  senseiAddTest(testPartitionersADIOS2BP4
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitionersDriver.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
      ${TEST_NP_HALF} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} adios2_bp4 3 3 3 0
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON ADIOS2 CATALYST
    PROPERTIES
      TIMEOUT 400)

  senseiAddTest(testPartitionersADIOS2SST
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitionersDriver.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
      ${TEST_NP_HALF} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} adios2_sst 3 3 3 1
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON ADIOS2 CATALYST
    PROPERTIES
      TIMEOUT 400)

  ##############################################################################
  senseiAddTest(testPartitionersHDF5A
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitionersDriver.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
      ${TEST_NP} ${TEST_NP_HALF} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} h5 3 3 3 0
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON HDF5 CATALYST
    PROPERTIES
      TIMEOUT 400)

  senseiAddTest(testPartitionersHDF5B
    PARALLEL_SHELL ${TEST_NP}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitionersDriver.sh
      ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
      ${TEST_NP} ${TEST_NP_HALF} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} h6 3 3 3 0
      -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
    FEATURES
      PYTHON HDF5 CATALYST)

   ##############################################################################
   senseiAddTest(testADIOS1FLEXPATHHistogram
     PARALLEL_SHELL 4
     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitioners.sh
       ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 2 2 2
       ${CMAKE_CURRENT_SOURCE_DIR} write_adios1_flexpath.xml
       histogram.xml read_adios1_flexpath_default.xml 10 1
       -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
     FEATURES
       PYTHON ADIOS1)

   ##############################################################################
   senseiAddTest(testADIOS2SSTHistogram
     PARALLEL_SHELL 4
     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testPartitioners.sh
       ${PYTHON_EXECUTABLE} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 4 4 2
       ${CMAKE_CURRENT_SOURCE_DIR} write_adios2_sst.xml
       histogram.xml read_adios2_sst_block.xml 10 1
       -- ${MPIEXEC_PREFLAGS} ${MPIEXEC_POSTFLAGS}
     FEATURES
       PYTHON ADIOS2)

  ##############################################################################
  senseiAddTest(testMeshMetadata
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testMeshMetadata.py
    FEATURES PYTHON)

  ##############################################################################
  if (SENSEI_ENABLE_CUDA)
    senseiAddTest(testHAMRDataArrayCUDA
      SOURCES testHAMRDataArray.cpp LIBS sensei EXEC_NAME testHAMRDataArray
      COMMAND $<TARGET_FILE:testHAMRDataArray> 0
      CUDA_TARGET
      )

    senseiAddTest(testHAMRDataArrayHost
      COMMAND $<TARGET_FILE:testHAMRDataArray> -1)

    senseiAddTest(testHAMRDataArrayMultiGPU
      SOURCES testHAMRDataArrayMultiGPU.cpp LIBS sensei EXEC_NAME testHAMRDataArrayMultiGPU
      COMMAND $<TARGET_FILE:testHAMRDataArrayMultiGPU> 0
      CUDA_TARGET
      )

  else()

    senseiAddTest(testHAMRDataArrayHost
      SOURCES testHAMRDataArray.cpp LIBS sensei EXEC_NAME testHAMRDataArray
      COMMAND $<TARGET_FILE:testHAMRDataArray> -1
      )

  endif()

  ##############################################################################
  if (SENSEI_ENABLE_OPENMP AND SENSEI_ENABLE_CUDA)
    if (NOT SENSEI_NVHPC_CUDA OR
     (SENSEI_NVHPC_CUDA AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 23.2)))

    senseiAddTest(testHAMRDataArrayOpenMP
      SOURCES testHAMRDataArrayOpenMP.cpp LIBS sensei EXEC_NAME testHAMRDataArrayOpenMP
      COMMAND $<TARGET_FILE:testHAMRDataArrayOpenMP> 0
      CUDA_TARGET
      )
    endif()
  endif()

endif()
