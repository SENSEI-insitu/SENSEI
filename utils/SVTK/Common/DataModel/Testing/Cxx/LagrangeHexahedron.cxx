#include "svtkLagrangeCurve.h"
#include "svtkLagrangeHexahedron.h"
#include "svtkLagrangeInterpolation.h"
#include "svtkLagrangeQuadrilateral.h"

#include "svtkMultiBaselineRegressionTest.h"

#include "svtkDoubleArray.h"
#include "svtkNew.h"
#include "svtkSmartPointer.h"
#include "svtkTable.h"

#include "svtkCell.h"
#include "svtkCellArray.h"
#include "svtkCellData.h"
#include "svtkIncrementalOctreePointLocator.h"
#include "svtkPointData.h"
#include "svtkPoints.h"
#include "svtkPolyData.h"
#include "svtkXMLPolyDataWriter.h"

#include "svtkVector.h"
#include "svtkVectorOperators.h"

#include "svtkAxis.h"
#include "svtkChartXY.h"
#include "svtkColor.h"
#include "svtkColorSeries.h"
#include "svtkContextScene.h"
#include "svtkContextView.h"
#include "svtkFloatArray.h"
#include "svtkPlot.h"
#include "svtkRegressionTestImage.h"
#include "svtkRenderWindow.h"
#include "svtkRenderWindowInteractor.h"
#include "svtkTable.h"

#include <sstream>
#include <vector>

#include "svtkTestConditionals.txx"

using namespace svtk;

static int expectedDOFIndices1[] = {
  0, 1, //
  3, 2, //
  4, 5, //
  7, 6  //
};

static int expectedDOFIndices2[] = {
  0, 8, 1,   //
  11, 24, 9, //
  3, 10, 2,  //

  16, 22, 17, //
  20, 26, 21, //
  19, 23, 18, //

  4, 12, 5,   //
  15, 25, 13, //
  7, 14, 6    //
};

static int expectedDOFIndices3[] = {
  0, 8, 9, 1,     //
  14, 48, 49, 10, //
  15, 50, 51, 11, //
  3, 12, 13, 2,   //

  24, 40, 41, 26, //
  32, 56, 57, 36, //
  33, 58, 59, 37, //
  30, 44, 45, 28, //

  25, 42, 43, 27, //
  34, 60, 61, 38, //
  35, 62, 63, 39, //
  31, 46, 47, 29, //

  4, 16, 17, 5,   //
  22, 52, 53, 18, //
  23, 54, 55, 19, //
  7, 20, 21, 6    //
};

static const double expectedFacePoints333[96][3] = {
  { 0, 1, 0 },
  { 0, 0, 0 },
  { 0, 0, 1 },
  { 0, 1, 1 },
  { 0, 0.666667, 0 },
  { 0, 0.333333, 0 },
  { 0, 0, 0.333333 },
  { 0, 0, 0.666667 },
  { 0, 0.666667, 1 },
  { 0, 0.333333, 1 },
  { 0, 1, 0.333333 },
  { 0, 1, 0.666667 },
  { 0, 0.666667, 0.333333 },
  { 0, 0.333333, 0.333333 },
  { 0, 0.666667, 0.666667 },
  { 0, 0.333333, 0.666667 },

  { 1, 0, 0 },
  { 1, 1, 0 },
  { 1, 1, 1 },
  { 1, 0, 1 },
  { 1, 0.333333, 0 },
  { 1, 0.666667, 0 },
  { 1, 1, 0.333333 },
  { 1, 1, 0.666667 },
  { 1, 0.333333, 1 },
  { 1, 0.666667, 1 },
  { 1, 0, 0.333333 },
  { 1, 0, 0.666667 },
  { 1, 0.333333, 0.333333 },
  { 1, 0.666667, 0.333333 },
  { 1, 0.333333, 0.666667 },
  { 1, 0.666667, 0.666667 },

  { 0, 0, 0 },
  { 1, 0, 0 },
  { 1, 0, 1 },
  { 0, 0, 1 },
  { 0.333333, 0, 0 },
  { 0.666667, 0, 0 },
  { 1, 0, 0.333333 },
  { 1, 0, 0.666667 },
  { 0.333333, 0, 1 },
  { 0.666667, 0, 1 },
  { 0, 0, 0.333333 },
  { 0, 0, 0.666667 },
  { 0.333333, 0, 0.333333 },
  { 0.666667, 0, 0.333333 },
  { 0.333333, 0, 0.666667 },
  { 0.666667, 0, 0.666667 },

  { 1, 1, 0 },
  { 0, 1, 0 },
  { 0, 1, 1 },
  { 1, 1, 1 },
  { 0.666667, 1, 0 },
  { 0.333333, 1, 0 },
  { 0, 1, 0.333333 },
  { 0, 1, 0.666667 },
  { 0.666667, 1, 1 },
  { 0.333333, 1, 1 },
  { 1, 1, 0.333333 },
  { 1, 1, 0.666667 },
  { 0.666667, 1, 0.333333 },
  { 0.333333, 1, 0.333333 },
  { 0.666667, 1, 0.666667 },
  { 0.333333, 1, 0.666667 },

  { 1, 0, 0 },
  { 0, 0, 0 },
  { 0, 1, 0 },
  { 1, 1, 0 },
  { 0.666667, 0, 0 },
  { 0.333333, 0, 0 },
  { 0, 0.333333, 0 },
  { 0, 0.666667, 0 },
  { 0.666667, 1, 0 },
  { 0.333333, 1, 0 },
  { 1, 0.333333, 0 },
  { 1, 0.666667, 0 },
  { 0.666667, 0.333333, 0 },
  { 0.333333, 0.333333, 0 },
  { 0.666667, 0.666667, 0 },
  { 0.333333, 0.666667, 0 },

  { 0, 0, 1 },
  { 1, 0, 1 },
  { 1, 1, 1 },
  { 0, 1, 1 },
  { 0.333333, 0, 1 },
  { 0.666667, 0, 1 },
  { 1, 0.333333, 1 },
  { 1, 0.666667, 1 },
  { 0.333333, 1, 1 },
  { 0.666667, 1, 1 },
  { 0, 0.333333, 1 },
  { 0, 0.666667, 1 },
  { 0.333333, 0.333333, 1 },
  { 0.666667, 0.333333, 1 },
  { 0.333333, 0.666667, 1 },
  { 0.666667, 0.666667, 1 },
};

static const double expectedEdgePoints333[48][3] = {
  { 0, 0, 0 },
  { 1, 0, 0 },
  { 0.333333, 0, 0 },
  { 0.666667, 0, 0 },

  { 1, 0, 0 },
  { 1, 1, 0 },
  { 1, 0.333333, 0 },
  { 1, 0.666667, 0 },

  { 0, 1, 0 },
  { 1, 1, 0 },
  { 0.333333, 1, 0 },
  { 0.666667, 1, 0 },

  { 0, 0, 0 },
  { 0, 1, 0 },
  { 0, 0.333333, 0 },
  { 0, 0.666667, 0 },

  { 0, 0, 1 },
  { 1, 0, 1 },
  { 0.333333, 0, 1 },
  { 0.666667, 0, 1 },

  { 1, 0, 1 },
  { 1, 1, 1 },
  { 1, 0.333333, 1 },
  { 1, 0.666667, 1 },

  { 0, 1, 1 },
  { 1, 1, 1 },
  { 0.333333, 1, 1 },
  { 0.666667, 1, 1 },

  { 0, 0, 1 },
  { 0, 1, 1 },
  { 0, 0.333333, 1 },
  { 0, 0.666667, 1 },

  { 0, 0, 0 },
  { 0, 0, 1 },
  { 0, 0, 0.333333 },
  { 0, 0, 0.666667 },

  { 1, 0, 0 },
  { 1, 0, 1 },
  { 1, 0, 0.333333 },
  { 1, 0, 0.666667 },

  { 1, 1, 0 },
  { 1, 1, 1 },
  { 1, 1, 0.333333 },
  { 1, 1, 0.666667 },

  { 0, 1, 0 },
  { 0, 1, 1 },
  { 0, 1, 0.333333 },
  { 0, 1, 0.666667 },
};

static bool SnapFace(svtkPoints* pts)
{
  int face1PtIds[] = { 1, 2, 5, 6, 9, 13, 17, 18, 21 };
  double face1PtDeltaX[] = { -0.10, -0.10, -0.10, -0.10, -0.05, -0.05, -0.05, -0.05, 0.0 };
  double face1PtDeltaY[] = { -0.10, +0.10, -0.10, +0.10, 0.00, 0.00, -0.05, +0.05, 0.0 };
  double face1PtDeltaZ[] = { -0.10, -0.10, +0.10, +0.10, -0.05, +0.05, 0.00, 0.00, 0.0 };
  svtkVector3d xx;
  for (unsigned ii = 0; ii < sizeof(face1PtIds) / sizeof(face1PtIds[0]); ++ii)
  {
    pts->GetPoint(face1PtIds[ii], xx.GetData());
    xx[0] += face1PtDeltaX[ii];
    xx[1] += face1PtDeltaY[ii];
    xx[2] += face1PtDeltaZ[ii];
    pts->SetPoint(face1PtIds[ii], xx.GetData());
  }
  return true;
}

static svtkSmartPointer<svtkLagrangeHexahedron> CreateCell(const svtkVector3i& testOrder)
{
  // Create a hex cell:
  svtkSmartPointer<svtkLagrangeHexahedron> hex = svtkSmartPointer<svtkLagrangeHexahedron>::New();
  hex->SetOrder(testOrder[0], testOrder[1], testOrder[2]);
  svtkSmartPointer<svtkPoints> pts = svtkSmartPointer<svtkPoints>::New();
  svtkLagrangeInterpolation::AppendHexahedronCollocationPoints(pts, testOrder.GetData());
  if (testOrder[0] == 2)
  {
    SnapFace(pts);
  }
  svtkIdType npts = pts->GetNumberOfPoints();
  std::cout << "Creating hex order " << testOrder << " with " << npts << " vertices\n";
  std::vector<svtkIdType> conn(npts);
  for (int c = 0; c < npts; ++c)
  {
    conn[c] = c;

    /*
    svtkVector3d pt;
    pts->GetPoint(c, pt.GetData());
    std::cout << "  " << c << "   " << pt << "\n";
    */
  }
  svtkCell* hexc = hex.GetPointer();
  hexc->Initialize(npts, &conn[0], pts);

  return hex;
}

template <typename T>
bool TestDOFIndices(T hex, const int* expectedDOFIndices)
{
  // A. Test DOF index lookup
  const int* testOrder = hex->GetOrder();

  std::cout << "Test index conversion for order (" << testOrder[0] << " " << testOrder[1] << " "
            << testOrder[2] << "):\n";
  int ee = 0;
  bool ok = true;
  for (int kk = 0; kk <= testOrder[2]; ++kk)
  {
    for (int jj = 0; jj <= testOrder[1]; ++jj)
    {
      for (int ii = 0; ii <= testOrder[0]; ++ii)
      {
        std::ostringstream tname;
        tname << "  PointIndexFromIJK(" << ii << ", " << jj << ", " << kk
              << ") == " << expectedDOFIndices[ee];
        // std::cout << tname.str() << " " << hex->PointIndexFromIJK(ii, jj, kk) << "\n";
        ok &= testEqual(hex->PointIndexFromIJK(ii, jj, kk), expectedDOFIndices[ee++], tname.str());
      }
    }
  }
  std::cout << "\n";
  return ok;
}

template <typename T>
bool TestGetFace(T hex, const double expected[][3])
{
  bool ok = true;
  int nn = 0;
  for (int faceId = 0; faceId < hex->GetNumberOfFaces(); ++faceId)
  {
    svtkLagrangeQuadrilateral* qq = svtkLagrangeQuadrilateral::SafeDownCast(hex->GetFace(faceId));
    testNotNull(qq, "GetFace: returns a non-NULL Lagrange quadrilateral");
    svtkIdType npts = qq->GetPointIds()->GetNumberOfIds();
    for (int pp = 0; pp < npts; ++pp)
    {
      svtkVector3d pt;
      // qq->GetPoints()->GetPoint(qq->GetPointIds()->GetId(pp), pt.GetData());
      qq->GetPoints()->GetPoint(pp, pt.GetData());
      std::ostringstream tname;
      tname << "  GetFace(" << faceId << ") point " << pp << " = " << pt;
      ok &= testNearlyEqual(pt, svtkVector3d(expected[nn++]), tname.str(), 1e-5);
    }
  }
  return ok;
}

template <typename T>
bool TestGetEdge(T hex, const double expected[][3])
{
  bool ok = true;
  int nn = 0;
  for (int edgeId = 0; edgeId < hex->GetNumberOfEdges(); ++edgeId)
  {
    svtkLagrangeCurve* cc = svtkLagrangeCurve::SafeDownCast(hex->GetEdge(edgeId));
    testNotNull(cc, "GetEdge: returns a non-NULL Lagrange curve");
    svtkIdType npts = cc->GetPointIds()->GetNumberOfIds();
    for (int pp = 0; pp < npts; ++pp)
    {
      svtkVector3d pt;
      // cc->GetPoints()->GetPoint(cc->GetPointIds()->GetId(pp), pt.GetData());
      cc->GetPoints()->GetPoint(pp, pt.GetData());
      std::ostringstream tname;
      tname << "  GetEdge(" << edgeId << ") point " << pp << " = " << pt;
      testNearlyEqual(pt, svtkVector3d(expected[nn++]), tname.str(), 1e-5);
    }
  }
  return ok;
}

template <typename T>
bool TestEvaluation(T& hex)
{
  bool ok = true;

  // A. EvaluateLocation: Convert parametric to world coordinates.
  int subId = -100;
  svtkVector3d param(1., 1., 1.);
  svtkVector3d posn;
  std::vector<double> shape(hex->GetPoints()->GetNumberOfPoints());
  hex->EvaluateLocation(subId, param.GetData(), posn.GetData(), &shape[0]);
  std::cout << "\nEvaluateLocation" << param << " -> " << posn << "\n";
  ok &= testEqual(subId, 0, "EvaluateLocation: subId should be 0");
  svtkVector3d p6;
  hex->GetPoints()->GetPoint(6, p6.GetData());
  ok &= testNearlyEqual(posn, p6, "EvaluateLocation: interpolate point coordinates");

  // B. EvaluatePosition: Convert world to parametric coordinates.
  svtkVector3d closest;
  double minDist2 = -1.0; // invalid
  int result = hex->EvaluatePosition(
    posn.GetData(), closest.GetData(), subId, param.GetData(), minDist2, &shape[0]);
  std::cout << "\nEvaluatePosition" << posn << " -> " << param << " dist " << minDist2 << " subid "
            << subId << " status " << result << "\n";
  ok &= testEqual(result, 1, "EvaluatePosition: proper return code for interior point");
  ok &= testNearlyEqual(
    param, svtkVector3d(1., 1., 1.), "EvaluatePosition: returned parametric coordinates");
  ok &= testNearlyEqual(closest, posn, "EvaluatePosition: test point interior to hex");
  ok &= testEqual(minDist2, 0.0, "EvaluatePosition: squared minimum distance should be 0");
  ok &= testEqual(subId, 7, "EvaluatePosition: point should be in last sub-hex");

  return ok;
}

template <typename T>
bool TestIntersection(T hex)
{
  bool ok = true;
  double testLines[][3] = {
    { +2., +2., +2. },
    { 0., 0., 0. },
    { +1.5, 0., +1. },
    { 0., 0., 0. },
    { -2., -2., -2. },
    { -3., -2., -1. },
  };
  int testLineStatus[] = {
    1,
    1,
    0,
  };
  /*
  double testLineParam[] = {
    0.571429,
    0.0714286,
    SVTK_DOUBLE_MAX
  };
  */
  for (unsigned tl = 0; tl < sizeof(testLines) / sizeof(testLines[0]) / 2; ++tl)
  {
    svtkVector3d p0(testLines[2 * tl]);
    svtkVector3d p1(testLines[2 * tl + 1]);
    double tol = 1e-7;
    double t;
    svtkVector3d x;
    svtkVector3d p;
    int subId = -1;
    int stat =
      hex->IntersectWithLine(p0.GetData(), p1.GetData(), tol, t, x.GetData(), p.GetData(), subId);
    std::cout << "\nIntersectWithLine " << p0 << " -- " << p1 << " stat " << stat << " t " << t
              << "\n       "
              << " subId " << subId << " x " << x << " p " << p << "\n";
    std::ostringstream tname;
    tname << "IntersectWithLine: status should be " << testLineStatus[tl];
    ok &= testEqual(stat, testLineStatus[tl], tname.str());
    // Commented out until we can validate:
    // ok &= testNearlyEqual(t, testLineParam[tl], "IntersectWithLine: line parameter",
    // testLineParam[tl]*1e-5);
  }
  return ok;
}

template <typename T>
bool TestContour(T hex)
{
  bool ok = true;
  double testPlanes[][6] = {
    { 0., 0., 0., 1., 0., 0. },
    { 0., 0., 0., 0., 1., 0. },
    { 0., 0., 0., 0., 0., 1. },
  };
  int testContourPointCount[] = {
    (hex->GetOrder()[0] + 1) * (hex->GetOrder()[1] + 1),
    (hex->GetOrder()[1] + 1) * (hex->GetOrder()[2] + 1),
    (hex->GetOrder()[2] + 1) * (hex->GetOrder()[0] + 1),
  };
  for (unsigned tp = 0; tp < sizeof(testPlanes) / sizeof(testPlanes[0]); ++tp)
  {
    svtkVector3d origin(testPlanes[tp]);
    svtkVector3d normal(testPlanes[tp] + 3);
    int np = hex->GetNumberOfPoints();
    svtkNew<svtkDoubleArray> contourScalars;
    svtkNew<svtkPoints> contourPoints;
    svtkNew<svtkIncrementalOctreePointLocator> locator;
    svtkNew<svtkCellArray> verts;
    svtkNew<svtkCellArray> lines;
    svtkNew<svtkCellArray> polys;
    svtkNew<svtkPointData> inPd;
    svtkNew<svtkPointData> outPd;
    svtkNew<svtkCellData> inCd;
    svtkNew<svtkCellData> outCd;
    contourScalars->SetNumberOfTuples(np);
    locator->InitPointInsertion(contourPoints, hex->GetBounds());
    for (svtkIdType ii = 0; ii < np; ++ii)
    {
      svtkVector3d pt(hex->GetPoints()->GetPoint(ii));
      double distance = normal.Dot(origin - pt);
      contourScalars->SetTuple1(ii, distance);
    }
    hex->Contour(
      0.0, contourScalars, locator, verts, lines, polys, inPd, outPd, inCd, /* cellId */ 0, outCd);

    int stat = static_cast<int>(contourPoints->GetNumberOfPoints());
    std::cout << "\nContour planar function: orig " << origin << " norm " << normal << "\n";
    std::ostringstream tname;
    tname << "Contour: num points out should be " << testContourPointCount[tp];
    ok &= testEqual(stat, testContourPointCount[tp], tname.str());
    for (int pp = 0; pp < stat; ++pp)
    {
      svtkVector3d pt(contourPoints->GetPoint(pp));
      double distance = normal.Dot(origin - pt);
      std::ostringstream testName;
      testName << "  Contour point " << pp << ": distance ";
      ok &= testNearlyEqual(distance, 0.0, testName.str(), 1e-5);
    }
  }
  return ok;
}

int LagrangeHexahedron(int argc, char* argv[])
{
  (void)argc;
  (void)argv;
  bool ok = true;

  svtkVector3i testOrder1(1, 1, 1);
  svtkSmartPointer<svtkLagrangeHexahedron> hex1 = CreateCell(testOrder1);

  svtkVector3i testOrder2(2, 2, 2);
  svtkSmartPointer<svtkLagrangeHexahedron> hex2 = CreateCell(testOrder2);

  svtkVector3i testOrder3(3, 3, 3);
  svtkSmartPointer<svtkLagrangeHexahedron> hex3 = CreateCell(testOrder3);

  // I. Low-level methods
  ok &= TestDOFIndices(hex1, expectedDOFIndices1);
  ok &= TestDOFIndices(hex2, expectedDOFIndices2);
  ok &= TestDOFIndices(hex3, expectedDOFIndices3);
  ok &= TestGetFace(hex3, expectedFacePoints333);
  ok &= TestGetEdge(hex3, expectedEdgePoints333);

  // II. High-level methods
  ok &= TestEvaluation(hex2);
  ok &= TestIntersection(hex1);
  ok &= TestIntersection(hex2);
  ok &= TestIntersection(hex3);
  ok &= TestContour(hex1);
  ok &= TestContour(hex2);
  ok &= TestContour(hex3);

  return ok ? 0 : 1;
}
