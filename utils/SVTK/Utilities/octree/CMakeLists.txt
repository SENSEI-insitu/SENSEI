set(template_classes
  octree/octree
  octree/octree_cursor
  octree/octree_iterator
  octree/octree_node
  octree/octree_path)

set(headers
  octree/octree)

svtk_module_add_module(SVTK::octree
  TEMPLATE_CLASSES ${template_classes}
  HEADERS ${headers}
  HEADERS_SUBDIR "octree"
  HEADER_ONLY)
