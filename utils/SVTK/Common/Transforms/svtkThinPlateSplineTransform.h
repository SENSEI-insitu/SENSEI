/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkThinPlateSplineTransform.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkThinPlateSplineTransform
 * @brief   a nonlinear warp transformation
 *
 * svtkThinPlateSplineTransform describes a nonlinear warp transform defined
 * by a set of source and target landmarks. Any point on the mesh close to a
 * source landmark will be moved to a place close to the corresponding target
 * landmark. The points in between are interpolated smoothly using
 * Bookstein's Thin Plate Spline algorithm.
 *
 * To obtain a correct TPS warp, use the R2LogR kernel if your data is 2D, and
 * the R kernel if your data is 3D. Or you can specify your own RBF. (Hence this
 * class is more general than a pure TPS transform.)
 * @warning
 * 1) The inverse transform is calculated using an iterative method,
 * and is several times more expensive than the forward transform.
 * 2) Whenever you add, subtract, or set points you must call Modified()
 * on the svtkPoints object, or the transformation might not update.
 * 3) Collinear point configurations (except those that lie in the XY plane)
 * result in an unstable transformation. Forward transform can be computed
 * for any configuration by disabling bulk transform regularization.
 * @sa
 * svtkGridTransform svtkGeneralTransform
 */

#ifndef svtkThinPlateSplineTransform_h
#define svtkThinPlateSplineTransform_h

#include "svtkCommonTransformsModule.h" // For export macro
#include "svtkWarpTransform.h"

#define SVTK_RBF_CUSTOM 0
#define SVTK_RBF_R 1
#define SVTK_RBF_R2LOGR 2

class SVTKCOMMONTRANSFORMS_EXPORT svtkThinPlateSplineTransform : public svtkWarpTransform
{
public:
  svtkTypeMacro(svtkThinPlateSplineTransform, svtkWarpTransform);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  static svtkThinPlateSplineTransform* New();

  //@{
  /**
   * Specify the 'stiffness' of the spline. The default is 1.0.
   */
  svtkGetMacro(Sigma, double);
  svtkSetMacro(Sigma, double);
  //@}

  //@{
  /**
   * Specify the radial basis function to use.  The default is
   * R2LogR which is appropriate for 2D. Use |R| (SetBasisToR)
   * if your data is 3D. Alternatively specify your own basis function,
   * however this will mean that the transform will no longer be a true
   * thin-plate spline.
   */
  void SetBasis(int basis);
  svtkGetMacro(Basis, int);
  void SetBasisToR() { this->SetBasis(SVTK_RBF_R); }
  void SetBasisToR2LogR() { this->SetBasis(SVTK_RBF_R2LOGR); }
  const char* GetBasisAsString();
  //@}

  //@{
  /**
   * Set the radial basis function to a custom function.  You must
   * supply both the function and its derivative with respect to r.
   */
  void SetBasisFunction(double (*U)(double r))
  {
    if (this->BasisFunction == U)
    {
      return;
    }
    this->SetBasis(SVTK_RBF_CUSTOM);
    this->BasisFunction = U;
    this->Modified();
  }
  void SetBasisDerivative(double (*dUdr)(double r, double& dU))
  {
    this->BasisDerivative = dUdr;
    this->Modified();
  }
  //@}

  //@{
  /**
   * Set the source landmarks for the warp.  If you add or change the
   * svtkPoints object, you must call Modified() on it or the transformation
   * might not update.
   */
  void SetSourceLandmarks(svtkPoints* source);
  svtkGetObjectMacro(SourceLandmarks, svtkPoints);
  //@}

  //@{
  /**
   * Set the target landmarks for the warp.  If you add or change the
   * svtkPoints object, you must call Modified() on it or the transformation
   * might not update.
   */
  void SetTargetLandmarks(svtkPoints* target);
  svtkGetObjectMacro(TargetLandmarks, svtkPoints);
  //@}

  /**
   * Get the MTime.
   */
  svtkMTimeType GetMTime() override;

  /**
   * Make another transform of the same type.
   */
  svtkAbstractTransform* MakeTransform() override;

  //@{
  /**
   * Get/set whether the bulk linear transformation matrix is regularized.
   *
   * If regularization is enabled: If all landmark points are on the
   * XY plane then forward and inverse transforms are computed correctly.
   * For other coplanar configurations, both forward an inverse transform
   * computation is unstable.
   *
   * If regularization is disabled: Forward transform is computed correctly
   * for all point configurations. Inverse transform computation is unstable
   * if source and/or target points are coplanar.
   *
   * If landmarks points are not coplanar then this setting has no effect.
   *
   * The default is true.
   */
  svtkGetMacro(RegularizeBulkTransform, bool);
  svtkSetMacro(RegularizeBulkTransform, bool);
  svtkBooleanMacro(RegularizeBulkTransform, bool);
  //@}

protected:
  svtkThinPlateSplineTransform();
  ~svtkThinPlateSplineTransform() override;

  /**
   * Prepare the transformation for application.
   */
  void InternalUpdate() override;

  /**
   * This method does no type checking, use DeepCopy instead.
   */
  void InternalDeepCopy(svtkAbstractTransform* transform) override;

  void ForwardTransformPoint(const float in[3], float out[3]) override;
  void ForwardTransformPoint(const double in[3], double out[3]) override;

  void ForwardTransformDerivative(const float in[3], float out[3], float derivative[3][3]) override;
  void ForwardTransformDerivative(
    const double in[3], double out[3], double derivative[3][3]) override;

  double Sigma;
  svtkPoints* SourceLandmarks;
  svtkPoints* TargetLandmarks;

  // the radial basis function to use
  double (*BasisFunction)(double r);
  double (*BasisDerivative)(double r, double& dUdr);

  int Basis;

  int NumberOfPoints;
  double** MatrixW;

  bool RegularizeBulkTransform;

private:
  svtkThinPlateSplineTransform(const svtkThinPlateSplineTransform&) = delete;
  void operator=(const svtkThinPlateSplineTransform&) = delete;
};

#endif
