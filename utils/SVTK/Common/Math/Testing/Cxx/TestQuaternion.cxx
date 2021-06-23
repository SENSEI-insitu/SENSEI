/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestQuaternion.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkMathUtilities.h"
#include "svtkQuaternion.h"
#include "svtkSetGet.h"

// Pre-declarations of the test functions
static int TestQuaternionSetGet();
static int TestQuaternionNormalization();
static int TestQuaternionConjugationAndInversion();
static int TestQuaternionRotation();
static int TestQuaternionMatrixConversions();
static int TestQuaternionConversions();
static int TestQuaternionSlerp();

//----------------------------------------------------------------------------
int TestQuaternion(int, char*[])
{
  // Store up any errors, return non-zero if something fails.
  int retVal = 0;

  retVal += TestQuaternionSetGet();
  retVal += TestQuaternionNormalization();
  retVal += TestQuaternionConjugationAndInversion();
  retVal += TestQuaternionRotation();
  retVal += TestQuaternionMatrixConversions();
  retVal += TestQuaternionConversions();
  retVal += TestQuaternionSlerp();

  return retVal;
}

// Test if the access and set methods are valids
//----------------------------------------------------------------------------
int TestQuaternionSetGet() // use of svtkQuaternionf for this test
{
  int retVal = 0;
  //
  // Test out the general vector data types, give nice API and great memory use
  svtkQuaternionf qf(1.0f);
  float zeroArrayf[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  qf.Set(zeroArrayf[0], zeroArrayf[1], zeroArrayf[2], zeroArrayf[3]);

  if (sizeof(qf) != sizeof(zeroArrayf))
  {
    // The two should be the same size and memory layout - error out if not
    std::cerr << "svtkQuaternionf should be the same size as float[4]." << std::endl
              << "sizeof(svtkQuaternionf) = " << sizeof(qf) << std::endl
              << "sizeof(float[4]) = " << sizeof(zeroArrayf) << std::endl;
    ++retVal;
  }
  if (qf.GetSize() != 4)
  {
    std::cerr << "Incorrect size of svtkQuaternionf, should be 4, but is " << qf.GetSize()
              << std::endl;
    ++retVal;
  }

  //
  // Test out svtkQuaternionf and ensure the various access methods are the same
  qf.Set(0.0f, 6.0f, 9.0f, 15.0f);
  if (qf.GetW() != qf[0] || !svtkMathUtilities::FuzzyCompare<float>(qf.GetW(), 0.0f))
  {
    std::cerr << "qf.GetW() should equal qf.GetData()[0] which should equal 0."
              << "\nqf.W() = " << qf.GetW() << std::endl
              << "qf[0] = " << qf[0] << std::endl;
    ++retVal;
  }
  if (qf.GetX() != qf[1] || !svtkMathUtilities::FuzzyCompare<float>(qf.GetX(), 6.0f))
  {
    std::cerr << "qf.GetX() should equal qf.GetData()[1] "
              << "which should equal 6.0. \nqf.GetX() = " << qf.GetX() << std::endl
              << "qf[1] = " << qf[1] << std::endl;
    ++retVal;
  }
  if (qf.GetY() != qf[2] || !svtkMathUtilities::FuzzyCompare<float>(qf.GetY(), 9.0f))
  {
    std::cerr << "qf.GetY() should equal qf.GetData()[2]"
              << " which should equal 9.0.\nqf.GetY() = " << qf.GetY() << std::endl
              << "qf[2] = " << qf[2] << std::endl;
    ++retVal;
  }
  if (qf.GetZ() != qf[3] || !svtkMathUtilities::FuzzyCompare<float>(qf.GetZ(), 15.0f))
  {
    std::cerr << "qf.GetZ() should equal qf.GetData()[3] "
              << "which should equal 15.0.\nqf.Z() = " << qf.GetZ() << std::endl
              << "qf[3] = " << qf[3] << std::endl;
    ++retVal;
  }

  //
  // Assign the data to an float array and ensure the two ways of
  // referencing are the same.
  float* floatPtr = qf.GetData();
  for (int i = 0; i < 3; ++i)
  {
    if (qf[i] != floatPtr[i] || qf(i) != qf[i])
    {
      std::cerr << "Error: qf[i] != floatPtr[i]" << std::endl
                << "qf[i] = " << qf[i] << std::endl
                << "floatPtr[i] = " << floatPtr[i] << std::endl;
      ++retVal;
    }
  }

  // To and from float[4]
  float setArray[4] = { 1.0f, -38.0f, 42.0f, 0.0001f };
  qf.Set(setArray);
  if (!qf.Compare(svtkQuaternionf(1.0, -38.0, 42.0, 0.0001), 0.0001))
  {
    std::cerr << "Error svtkQuaterniond::Set(float[4]) failed: " << qf << std::endl;
    ++retVal;
  }

  float arrayToCompare[4];
  qf.Get(arrayToCompare);
  for (int i = 0; i < 4; ++i)
  {
    if (!svtkMathUtilities::FuzzyCompare(setArray[i], arrayToCompare[i]))
    {
      std::cerr << "Error svtkQuaterniond::Get(float[4]) failed: " << setArray[i]
                << "!= " << arrayToCompare[i] << std::endl;
      ++retVal;
    }
  }

  return retVal;
}

// Test the normalize and normalized functions.
//----------------------------------------------------------------------------
int TestQuaternionNormalization() // This test use svtkQuaterniond
{
  int retVal = 0;

  svtkQuaterniond normy(1.0, 2.0, 3.0, 4.0);
  svtkQuaterniond normed = normy.Normalized();
  if (!normed.Compare(svtkQuaterniond(0.182574, 0.365148, 0.547723, 0.730297), 0.0001))
  {
    std::cerr << "Error svtkQuaterniond::Normalized() failed: " << normed << std::endl;
    ++retVal;
  }
  normy.Normalize();
  if (!normy.Compare(normed, 0.0001))
  {
    std::cerr << "Error svtkQuaterniond::Normalize() failed: " << normy << std::endl;
  }
  if (!svtkMathUtilities::FuzzyCompare(normy.Norm(), 1.0, 0.0001))
  {
    std::cerr << "Normalized length should always be ~= 1.0, value is " << normy.Norm()
              << std::endl;
    ++retVal;
  }

  return retVal;
}

// This tests the conjugation and inversion at the same time.
// Since inversion depends on normalization, this will probably fail
// if TestQuaternionNormalisation() fails.
//----------------------------------------------------------------------------
int TestQuaternionConjugationAndInversion() // this test uses svtkQuaternionf
{
  int retVal = 0;

  //
  // Test conjugate and inverse at the same time.
  // [inv(q) = conj(q)/norm2(q)]
  svtkQuaternionf toConjugate(2.0f);
  svtkQuaternionf conjugate = toConjugate.Conjugated();
  if (!conjugate.Compare(svtkQuaternionf(2.0f, -2.0f, -2.0f, -2.0f), 0.0001))
  {
    std::cerr << "Error svtkQuaternionf::Conjugated() failed: " << conjugate << std::endl;
    ++retVal;
  }
  float squaredNorm = conjugate.SquaredNorm();
  svtkQuaternionf invToConjugate = conjugate / squaredNorm;
  if (!invToConjugate.Compare(svtkQuaternionf(0.125f, -0.125f, -0.125f, -0.125f), 0.0001))
  {
    std::cerr << "Error svtkQuaternionf Divide by Scalar() failed: " << invToConjugate << std::endl;
    ++retVal;
  }

  svtkQuaternionf shouldBeIdentity = invToConjugate * toConjugate;
  svtkQuaternionf identity;
  identity.ToIdentity();
  if (!shouldBeIdentity.Compare(identity, 0.0001))
  {
    std::cerr << "Error svtkQuaternionf multiplication failed: " << shouldBeIdentity << std::endl;
    ++retVal;
  }
  toConjugate.Invert();
  if (!invToConjugate.Compare(toConjugate, 0.0001))
  {
    std::cerr << "Error svtkQuaternionf::Inverse failed: " << toConjugate << std::endl;
    ++retVal;
  }
  shouldBeIdentity.Invert();
  if (!shouldBeIdentity.Compare(identity, 0.0001))
  {
    std::cerr << "Error svtkQuaternionf::Inverse failed: " << shouldBeIdentity << std::endl;
    ++retVal;
  }

  return retVal;
}

// Test the rotations
//----------------------------------------------------------------------------
int TestQuaternionRotation() // this test uses svtkQuaterniond
{
  int retVal = 0;

  //
  // Test rotations
  svtkQuaterniond rotation;
  rotation.SetRotationAngleAndAxis(svtkMath::RadiansFromDegrees(10.0), 1.0, 1.0, 1.0);

  if (!rotation.Compare(svtkQuaterniond(0.996195, 0.0290519, 0.0290519, 0.0290519), 0.0001))
  {
    std::cerr << "Error svtkQuaterniond::SetRotation Angle()"
              << " and Axis() failed: " << rotation << std::endl;
    ++retVal;
  }

  svtkQuaterniond secondRotation;
  secondRotation.SetRotationAngleAndAxis(svtkMath::RadiansFromDegrees(-20.0), 1.0, -1.0, 1.0);
  if (!secondRotation.Compare(svtkQuaterniond(0.984808, -0.0578827, 0.0578827, -0.0578827), 0.0001))
  {
    std::cerr << "Error svtkQuaterniond::SetRotation Angle()"
              << " and Axis() failed: " << secondRotation << std::endl;
    ++retVal;
  }

  svtkQuaterniond resultRotation = rotation * secondRotation;
  double axis[3];
  double supposedAxis[3] = { -0.338805, 0.901731, -0.2685 };
  double angle = resultRotation.GetRotationAngleAndAxis(axis);

  if (!svtkMathUtilities::FuzzyCompare(axis[0], supposedAxis[0], 0.0001) ||
    !svtkMathUtilities::FuzzyCompare(axis[1], supposedAxis[1], 0.0001) ||
    !svtkMathUtilities::FuzzyCompare(axis[2], supposedAxis[2], 0.0001))
  {
    std::cerr << "Error svtkQuaterniond::GetRotationAxis() failed: " << axis[0] << "  " << axis[1]
              << "  " << axis[2] << std::endl;
    ++retVal;
  }
  if (!svtkMathUtilities::FuzzyCompare(svtkMath::DegreesFromRadians(angle), 11.121, 0.0001))
  {
    std::cerr << "Error svtkQuaterniond::GetRotationAngle() failed: "
              << svtkMath::DegreesFromRadians(angle) << std::endl;
    ++retVal;
  }

  return retVal;
}

// Test the matrix conversions
//----------------------------------------------------------------------------
int TestQuaternionMatrixConversions() // this test uses svtkQuaternionf
{
  int retVal = 0;

  svtkQuaternionf quat;
  float M[3][3];
  M[0][0] = 0.98420;
  M[0][1] = 0.17354;
  M[0][2] = 0.03489;
  M[1][0] = -0.17327;
  M[1][1] = 0.90415;
  M[1][2] = 0.39049;
  M[2][0] = 0.03621;
  M[2][1] = -0.39037;
  M[2][2] = 0.91994;
  quat.FromMatrix3x3(M);

  if (!(quat.Compare(svtkQuaternionf(-0.975744, 0.200069, 0.000338168, 0.0888578), 0.001)))
  {
    std::cerr << "Error svtkQuaternionf FromMatrix3x3 failed: " << quat << std::endl;
    ++retVal;
  }

  // an easy one, just to make sure !
  float newM[3][3];
  quat.ToMatrix3x3(newM);
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      if (!svtkMathUtilities::FuzzyCompare(M[i][j], newM[i][j], 0.001f))
      {
        std::cerr << "Error svtkQuaternionf ToMatrix3x3 failed: " << M[i][j] << " != " << newM[i][j]
                  << std::endl;
        ++retVal;
      }
    }
  }

  // Rotate -23 degrees around X
  M[0][0] = 1.0;
  M[0][1] = 0.0;
  M[0][2] = 0.0;
  M[1][0] = 0.0;
  M[1][1] = 0.92050;
  M[1][2] = 0.39073;
  M[2][0] = 0.0;
  M[2][1] = -0.39073;
  M[2][2] = 0.92050;
  // Let's also make the quaternion
  quat.SetRotationAngleAndAxis(svtkMath::RadiansFromDegrees(-23.0), 1.0, 0.0, 0.0);

  // just in case, it makes another test
  svtkQuaternionf newQuat;
  newQuat.FromMatrix3x3(M);
  if (!(newQuat.Compare(quat, 0.00001)))
  {
    std::cerr << "Error svtkQuaternionf FromMatrix3x3 failed: " << newQuat << " != " << quat
              << std::endl;
    ++retVal;
  }

  // And compare again !
  quat.ToMatrix3x3(newM);
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      if (!svtkMathUtilities::FuzzyCompare(M[i][j], newM[i][j], 0.001f))
      {
        std::cerr << "Error svtkQuaternionf ToMatrix3x3 failed: " << M[i][j] << " != " << newM[i][j]
                  << std::endl;
        ++retVal;
      }
    }
  }

  return retVal;
}

// Test the quaternion's conversions
//----------------------------------------------------------------------------
int TestQuaternionConversions() // this test uses svtkQuaterniond
{
  int retVal = 0;
  svtkQuaterniond quat(15.0, -3.0, 2.0, 0.001);

  // Logarithm
  svtkQuaterniond logQuat;
  logQuat = quat.UnitLog();
  if (!(logQuat.Compare(svtkQuaterniond(0, -0.19628, 0.13085, 0.00007), 0.00001)))
  {
    std::cerr << "Error svtkQuaterniond UnitLogQuaternion() failed: " << logQuat << std::endl;
    ++retVal;
  }

  // Exponential
  svtkQuaterniond expQuat = quat.UnitExp();
  if (!(expQuat.Compare(svtkQuaterniond(-0.89429, 0.37234, -0.24822, -0.00012), 0.00001)))
  {
    std::cerr << "Error svtkQuaterniond UnitExpQuaternion() failed: " << expQuat << std::endl;
    ++retVal;
  }

  // UnitExp(UnitLog(q)) on a normalized quaternion is an identity operation
  svtkQuaterniond normQuat = quat.Normalized();
  if (!(normQuat.Compare(logQuat.UnitExp(), 0.00001)))
  {
    std::cerr << "Error svtkQuaterniond UnitExp(UnitLog(q)) is not identity: " << logQuat.UnitExp()
              << " vs. " << normQuat << std::endl;
    ++retVal;
  }

  // To SVTK
  svtkQuaterniond svtkQuat = quat.NormalizedWithAngleInDegrees();
  if (!(svtkQuat.Compare(svtkQuaterniond(55.709, -0.194461, 0.129641, 6.48204e-005), 0.00001)))
  {
    std::cerr << "Error svtkQuaterniond UnitForSVTKQuaternion() failed: " << svtkQuat << std::endl;
    ++retVal;
  }

  return retVal;
}

// Test the quaternion's slerp
//----------------------------------------------------------------------------
int TestQuaternionSlerp() // this test uses svtkQuaterniond
{
  // return value
  int retVal = 0;

  // first quaternion
  svtkQuaternion<double> q1;
  // quaternion which represents a small rotation
  svtkQuaternion<double> dq;
  // q2 is obtained by doing dq*q1
  svtkQuaternion<double> q2;
  // dqt is the rotation to multiply with q1
  // to obtained the SLERP interpolation of q1 and q2
  svtkQuaternion<double> dqt;
  // qTruth is the result of dqt*q1
  svtkQuaternion<double> qTruth;
  // qSlerp is the result of the SLERP interpolation
  // it should be equal to qTruth
  svtkQuaternion<double> qSlerp;

  // exhaustive test : 250000 operations
  // Control the sampling of rotation's axis
  const int M = 5;
  // Control the sampling of the rotation's angle
  const int L = 10;
  // Control the sampling of the interpolation
  const int N = 20;

  // axis coordinates step
  double dAxis = 1.0 / static_cast<double>(M);
  // angle step
  double dAngle = 360.0 / static_cast<double>(L);
  // interpolation step
  double dt = 1.0 / static_cast<double>(N);

  double x, y, z, angle, t, distance, angleShort;
  double axis[3];
  double axisNorme;

  // loop over x-coordinates
  for (int i = 1; i <= M; ++i)
  {
    x = static_cast<double>(i) * dAxis;
    // loop over y-coordinates
    for (int j = 1; j <= M; ++j)
    {
      y = static_cast<double>(j) * dAxis;
      // loop over z-coordinates
      for (int k = 1; k <= M; ++k)
      {
        z = static_cast<double>(k) * dAxis;
        axisNorme = sqrt(x * x + y * y + z * z);
        axis[0] = x / axisNorme;
        axis[1] = y / axisNorme;
        axis[2] = z / axisNorme;
        // loop over the angle of q1
        for (int u = 1; u <= L; ++u)
        {
          angle = static_cast<double>(u) * dAngle;
          q1.SetRotationAngleAndAxis(svtkMath::RadiansFromDegrees(angle), axis[0], axis[1], axis[2]);
          // loop over the angle of dq
          for (int v = 1; v < L; ++v)
          {
            angleShort = (static_cast<double>(v) * dAngle) / 2;
            dq.SetRotationAngleAndAxis(
              svtkMath::RadiansFromDegrees(angleShort), axis[0], axis[1], axis[2]);
            q2 = dq * q1;
            // loop over the interpolation step
            for (int w = 0; w <= N; ++w)
            {
              t = static_cast<double>(w) * dt;
              dqt.SetRotationAngleAndAxis(
                svtkMath::RadiansFromDegrees(t * angleShort), axis[0], axis[1], axis[2]);
              qTruth = dqt * q1;
              qSlerp = q1.Slerp(t, q2);
              distance = (qSlerp - qTruth).Norm();
              if (distance > 1e-12)
              {
                ++retVal;
              }
            }
          }
        }
      }
    }
  }

  // Particular test : we test that the SLERP take the
  // short path

  double u[3] = { -0.54, -0.0321, 1 };
  double normU = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
  u[0] /= normU;
  u[1] /= normU;
  u[2] /= normU;

  // interpolation step
  const int N2 = 1000;
  double dtheta = 3.0;
  // Set q1 close to the angle boundary
  q1.SetRotationAngleAndAxis(svtkMath::RadiansFromDegrees(359.5), u[0], u[1], u[2]);
  // dq represents a small rotation
  dq.SetRotationAngleAndAxis(svtkMath::RadiansFromDegrees(dtheta), u[0], u[1], u[2]);
  // q2 is a rotation close to q1 but the quaternion representant is far
  q2 = dq * q1;

  dt = 1.0 / static_cast<double>(N2);

  for (int i = 0; i <= N2; ++i)
  {
    t = static_cast<double>(i) * dt;
    dqt.SetRotationAngleAndAxis(svtkMath::RadiansFromDegrees(t * dtheta), u[0], u[1], u[2]);
    qTruth = dqt * q1;
    qSlerp = q1.Slerp(t, q2);
    distance = (qSlerp - qTruth).Norm();
    if (distance > 1e-12)
    {
      ++retVal;
    }
  }

  if (retVal != 0)
  {
    std::cerr << "Error TestQuaternionSlerp() failed" << std::endl;
  }

  return retVal;
}
