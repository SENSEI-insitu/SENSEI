set(classes
  svtkAbstractTransform
  svtkCylindricalTransform
  svtkGeneralTransform
  svtkHomogeneousTransform
  svtkIdentityTransform
  svtkLandmarkTransform
  svtkLinearTransform
  svtkMatrixToHomogeneousTransform
  svtkMatrixToLinearTransform
  svtkPerspectiveTransform
  svtkSphericalTransform
  svtkThinPlateSplineTransform
  svtkTransform
  svtkTransform2D
  svtkTransformCollection
  svtkWarpTransform)

svtk_module_add_module(SVTK::CommonTransforms
  CLASSES ${classes})
