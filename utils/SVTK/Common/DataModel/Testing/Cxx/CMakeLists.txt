  if(SVTK_LEGACY_REMOVE)
  set(legacy_srcs)
  else()
  set(legacy_srcs
    TestDataArrayDispatcher.cxx
    TestDispatchers.cxx
    )
endif()

svtk_add_test_cxx(svtkCommonDataModelCxxTests tests
  NO_DATA NO_VALID NO_OUTPUT
  ${legacy_srcs}
  LagrangeHexahedron.cxx
  BezierInterpolation.cxx
  TestBezier.cxx
  TestAngularPeriodicDataArray.cxx
  TestColor.cxx
  TestVector.cxx
  TestVectorOperators.cxx
  TestAMRBox.cxx
  TestBiQuadraticQuad.cxx
  TestCellArray.cxx
  TestCellArrayTraversal.cxx
  TestCompositeDataSets.cxx
  TestCompositeDataSetRange.cxx
  TestComputeBoundingSphere.cxx
  TestDataObject.cxx
  TestDataObjectTreeRange.cxx
  TestFieldList.cxx
  TestGenericCell.cxx
  TestGraph.cxx
  TestGraph2.cxx
  TestGraphAttributes.cxx
  TestHigherOrderCell.cxx
  TestImageDataFindCell.cxx
  TestImageDataInterpolation.cxx
  TestImageDataOrientation.cxx
  TestImageIterator.cxx
  TestInterpolationDerivs.cxx
  TestInterpolationFunctions.cxx
  TestMappedGridDeepCopy.cxx
  TestPath.cxx
  TestPentagonalPrism.cxx
  TestPiecewiseFunctionLogScale.cxx
  TestPixelExtent.cxx
  TestPointLocators.cxx
  TestPolyDataRemoveCell.cxx
  TestPolygon.cxx
  TestPolygonBoundedTriangulate.cxx
  TestPolyhedron0.cxx
  TestPolyhedron1.cxx
  TestPolyhedron3.cxx
  TestPolyhedronCombinatorialContouring.cxx
  TestPolyhedronConvexity.cxx
  TestPolyhedronConvexityMultipleCells.cxx
  TestQuadraticPolygon.cxx
  TestRect.cxx
  TestSelectionExpression.cxx
  TestSelectionSubtract.cxx
  TestSortFieldData.cxx
  TestStaticCellLocator.cxx
  TestTable.cxx
  TestTreeBFSIterator.cxx
  TestTreeDFSIterator.cxx
  TestTriangle.cxx
  TimePointLocators.cxx
  otherCellBoundaries.cxx
  otherCellPosition.cxx
  otherCellTypes.cxx
  otherColorTransferFunction.cxx
  otherEmptyCell.cxx
  otherFieldData.cxx
  otherImageData.cxx
  otherRectilinearGrid.cxx
  otherStructuredGrid.cxx
  otherUniformGrid.cxx
  quadCellConsistency.cxx
  quadraticEvaluation.cxx
  TestBoundingBox.cxx
  TestPlane.cxx
  TestStaticCellLinks.cxx
  TestStructuredData.cxx
  TestDataObjectTypes.cxx
  TestPolyDataRemoveDeletedCells.cxx
  UnitTestCells.cxx
  UnitTestImplicitDataSet.cxx
  UnitTestImplicitVolume.cxx
  UnitTestLine.cxx
  UnitTestPlanesIntersection.cxx
  UnitTestTriangleIntersection.cxx
  )
svtk_add_test_cxx(svtkCommonDataModelCxxTests data_tests
  TestImageDataOrientation2.cxx,NO_DATA
  TestLagrangeTriangle.cxx
  TestLagrangeTetra.cxx
  TestLagrangeWedge.cxx
  quadraticIntersection.cxx
  TestCellIterators.cxx,NO_VALID,NO_OUTPUT
  TestCellLocator.cxx,NO_DATA
  TestMeanValueCoordinatesInterpolation1.cxx
  TestMeanValueCoordinatesInterpolation2.cxx
  TestPolyhedron4.cxx,NO_VALID,NO_OUTPUT
  TestSmoothErrorMetric.cxx
  TestQuadraticPolygonFilters.cxx
  )
# add to the list but don't define a test
list(APPEND data_tests
  TestPolyhedron2.cxx
  TestPolyhedronContouring.cxx
  TestPolyhedronCutter.cxx
  TestStaticCellLocatorEdgeCases.cxx
  )

svtk_add_test_cxx(svtkCommonDataModelCxxTests output_tests
  TestKdTreeRepresentation.cxx,NO_DATA
  )
ExternalData_add_test(${_svtk_build_TEST_DATA_TARGET}
  NAME SVTK::CommonDataModelCxxTests-TestPolyhedron2
  COMMAND svtkCommonDataModelCxxTests TestPolyhedron2
  -D DATA{../Data/polyhedron_mesh.vtu}
  )
ExternalData_add_test(${_svtk_build_TEST_DATA_TARGET}
  NAME SVTK::CommonDataModelCxxTests-TestPolyhedronContouring
  COMMAND svtkCommonDataModelCxxTests TestPolyhedronContouring
   DATA{../Data/cell_850113.vtu} DATA{../Data/cell_12851_26.vtu}
  )
ExternalData_add_test(${_svtk_build_TEST_DATA_TARGET}
  NAME SVTK::CommonDataModelCxxTests-TestPolyhedronCutter
  COMMAND svtkCommonDataModelCxxTests TestPolyhedronCutter
  DATA{../Data/onePolyhedron.vtu} DATA{../Data/sliceOfPolyhedron.vtu}
  )
ExternalData_add_test(${_svtk_build_TEST_DATA_TARGET}
  NAME SVTK::CommonDataModelCxxTests-TestStaticCellLocatorEdgeCases
  COMMAND svtkCommonDataModelCxxTests TestStaticCellLocatorEdgeCases
  DATA{../Data/test_surface.vtp}
  )

set(all_tests
  ${tests}
  ${data_tests}
  ${output_tests}
  ${custom_tests}
  )
svtk_test_cxx_executable(svtkCommonDataModelCxxTests all_tests)
