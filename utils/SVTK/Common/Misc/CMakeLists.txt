set(classes
  svtkContourValues
  svtkErrorCode
  svtkFunctionParser
  svtkHeap
  svtkPolygonBuilder
  svtkResourceFileLocator)

svtk_module_add_module(SVTK::CommonMisc
  CLASSES ${classes})
