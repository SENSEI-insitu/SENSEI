"""
This file is obsolete.
All the constants are part of the base svtk module.
"""

# Some constants used throughout code

SVTK_FLOAT_MAX = 1.0e+38
SVTK_INT_MAX = 2147483647 # 2^31 - 1

# These types are returned by GetDataType to indicate pixel type.
SVTK_VOID            = 0
SVTK_BIT             = 1
SVTK_CHAR            = 2
SVTK_SIGNED_CHAR     =15
SVTK_UNSIGNED_CHAR   = 3
SVTK_SHORT           = 4
SVTK_UNSIGNED_SHORT  = 5
SVTK_INT             = 6
SVTK_UNSIGNED_INT    = 7
SVTK_LONG            = 8
SVTK_UNSIGNED_LONG   = 9
SVTK_FLOAT           =10
SVTK_DOUBLE          =11
SVTK_ID_TYPE         =12

# These types are not currently supported by GetDataType, but are
# for completeness.
SVTK_STRING          =13
SVTK_OPAQUE          =14

SVTK_LONG_LONG          =16
SVTK_UNSIGNED_LONG_LONG =17

# These types are required by svtkVariant and svtkVariantArray
SVTK_VARIANT =20
SVTK_OBJECT  =21

# Storage for Unicode strings
SVTK_UNICODE_STRING  =22


# Some constant required for correct template performance
SVTK_BIT_MIN = 0
SVTK_BIT_MAX = 1
SVTK_CHAR_MIN = -128
SVTK_CHAR_MAX = 127
SVTK_UNSIGNED_CHAR_MIN = 0
SVTK_UNSIGNED_CHAR_MAX = 255
SVTK_SHORT_MIN = -32768
SVTK_SHORT_MAX = 32767
SVTK_UNSIGNED_SHORT_MIN = 0
SVTK_UNSIGNED_SHORT_MAX = 65535
SVTK_INT_MIN = (-SVTK_INT_MAX-1)
SVTK_INT_MAX = SVTK_INT_MAX
#SVTK_UNSIGNED_INT_MIN = 0
#SVTK_UNSIGNED_INT_MAX = 4294967295
SVTK_LONG_MIN = (-SVTK_INT_MAX-1)
SVTK_LONG_MAX = SVTK_INT_MAX
#SVTK_UNSIGNED_LONG_MIN = 0
#SVTK_UNSIGNED_LONG_MAX = 4294967295
SVTK_FLOAT_MIN = -SVTK_FLOAT_MAX
SVTK_FLOAT_MAX = SVTK_FLOAT_MAX
SVTK_DOUBLE_MIN = -1.0e+99
SVTK_DOUBLE_MAX  = 1.0e+99

# These types are returned to distinguish dataset types
SVTK_POLY_DATA          = 0
SVTK_STRUCTURED_POINTS  = 1
SVTK_STRUCTURED_GRID    = 2
SVTK_RECTILINEAR_GRID   = 3
SVTK_UNSTRUCTURED_GRID  = 4
SVTK_PIECEWISE_FUNCTION = 5
SVTK_IMAGE_DATA         = 6
SVTK_DATA_OBJECT        = 7
SVTK_DATA_SET           = 8
SVTK_POINT_SET          = 9
SVTK_UNIFORM_GRID                  = 10
SVTK_COMPOSITE_DATA_SET            = 11
SVTK_MULTIGROUP_DATA_SET           = 12 # OBSOLETE
SVTK_MULTIBLOCK_DATA_SET           = 13
SVTK_HIERARCHICAL_DATA_SET         = 14 # OBSOLETE
SVTK_HIERARCHICAL_BOX_DATA_SET     = 15
SVTK_GENERIC_DATA_SET              = 16
SVTK_HYPER_OCTREE                  = 17
SVTK_TEMPORAL_DATA_SET             = 18
SVTK_TABLE                         = 19
SVTK_GRAPH                         = 20
SVTK_TREE                          = 21
SVTK_SELECTION                     = 22

# These types define error codes for svtk functions
SVTK_OK                 = 1
SVTK_ERROR              = 2

# These types define different text properties
SVTK_ARIAL        = 0
SVTK_COURIER      = 1
SVTK_TIMES        = 2
SVTK_UNKNOWN_FONT = 3

SVTK_TEXT_LEFT     = 0
SVTK_TEXT_CENTERED = 1
SVTK_TEXT_RIGHT    = 2

SVTK_TEXT_BOTTOM   = 0
SVTK_TEXT_TOP      = 2

SVTK_TEXT_GLOBAL_ANTIALIASING_SOME  = 0
SVTK_TEXT_GLOBAL_ANTIALIASING_NONE  = 1
SVTK_TEXT_GLOBAL_ANTIALIASING_ALL   = 2

SVTK_LUMINANCE        = 1
SVTK_LUMINANCE_ALPHA  = 2
SVTK_RGB              = 3
SVTK_RGBA             = 4

SVTK_COLOR_MODE_DEFAULT     = 0
SVTK_COLOR_MODE_MAP_SCALARS = 1

# Constants for InterpolationType
SVTK_NEAREST_INTERPOLATION      = 0
SVTK_LINEAR_INTERPOLATION       = 1

# For volume rendering
SVTK_MAX_VRCOMP    = 4

# These types define the 17 linear SVTK Cell Types
# See Filtering/svtkCellType.h

# Linear cells
SVTK_EMPTY_CELL       = 0
SVTK_VERTEX           = 1
SVTK_POLY_VERTEX      = 2
SVTK_LINE             = 3
SVTK_POLY_LINE        = 4
SVTK_TRIANGLE         = 5
SVTK_TRIANGLE_STRIP   = 6
SVTK_POLYGON          = 7
SVTK_PIXEL            = 8
SVTK_QUAD             = 9
SVTK_TETRA            = 10
SVTK_VOXEL            = 11
SVTK_HEXAHEDRON       = 12
SVTK_WEDGE            = 13
SVTK_PYRAMID          = 14
SVTK_PENTAGONAL_PRISM = 15
SVTK_HEXAGONAL_PRISM  = 16

# Quadratic, isoparametric cells
SVTK_QUADRATIC_EDGE                   = 21
SVTK_QUADRATIC_TRIANGLE               = 22
SVTK_QUADRATIC_QUAD                   = 23
SVTK_QUADRATIC_TETRA                  = 24
SVTK_QUADRATIC_HEXAHEDRON             = 25
SVTK_QUADRATIC_WEDGE                  = 26
SVTK_QUADRATIC_PYRAMID                = 27
SVTK_BIQUADRATIC_QUAD                 = 28
SVTK_TRIQUADRATIC_HEXAHEDRON          = 29
SVTK_QUADRATIC_LINEAR_QUAD            = 30
SVTK_QUADRATIC_LINEAR_WEDGE           = 31
SVTK_BIQUADRATIC_QUADRATIC_WEDGE      = 32
SVTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33

# Special class of cells formed by convex group of points
SVTK_CONVEX_POINT_SET = 41

# Higher order cells in parametric form
SVTK_PARAMETRIC_CURVE        = 51
SVTK_PARAMETRIC_SURFACE      = 52
SVTK_PARAMETRIC_TRI_SURFACE  = 53
SVTK_PARAMETRIC_QUAD_SURFACE = 54
SVTK_PARAMETRIC_TETRA_REGION = 55
SVTK_PARAMETRIC_HEX_REGION   = 56

# Higher order cells
SVTK_HIGHER_ORDER_EDGE        = 60
SVTK_HIGHER_ORDER_TRIANGLE    = 61
SVTK_HIGHER_ORDER_QUAD        = 62
SVTK_HIGHER_ORDER_POLYGON     = 63
SVTK_HIGHER_ORDER_TETRAHEDRON = 64
SVTK_HIGHER_ORDER_WEDGE       = 65
SVTK_HIGHER_ORDER_PYRAMID     = 66
SVTK_HIGHER_ORDER_HEXAHEDRON  = 67

# A macro to get the name of a type
__svtkTypeNameDict = {SVTK_VOID:"void",
                     SVTK_DOUBLE:"double",
                     SVTK_FLOAT:"float",
                     SVTK_LONG:"long",
                     SVTK_UNSIGNED_LONG:"unsigned long",
                     SVTK_INT:"int",
                     SVTK_UNSIGNED_INT:"unsigned int",
                     SVTK_SHORT:"short",
                     SVTK_UNSIGNED_SHORT:"unsigned short",
                     SVTK_CHAR:"char",
                     SVTK_UNSIGNED_CHAR:"unsigned char",
                     SVTK_SIGNED_CHAR:"signed char",
                     SVTK_LONG_LONG:"long long",
                     SVTK_UNSIGNED_LONG_LONG:"unsigned long long",
                     SVTK_ID_TYPE:"svtkIdType",
                     SVTK_BIT:"bit"}

def svtkImageScalarTypeNameMacro(type):
  return __svtkTypeNameDict[type]
