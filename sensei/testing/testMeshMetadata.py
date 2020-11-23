from sensei import *
from vtk import VTK_MULTIBLOCK_DATA_SET, vtkDataObject, VTK_IMAGE_DATA, VTK_DOUBLE, VTK_FLOAT

# test setting and getting the members
md = MeshMetadata.New()
md.GlobalView = True
md.MeshName = "foo"
md.MeshType = VTK_MULTIBLOCK_DATA_SET
md.BlockType = VTK_IMAGE_DATA
md.NumBlocks = 2
md.NumBlocksLocal = [2]
md.Extent = [0,2,0,2,0,2]
md.Bounds = [0.,1.,0.,1.,0.,1.]
md.CoordinateType = VTK_DOUBLE
md.NumPoints = 3**3
md.NumCells = 2**3
md.CellArraySize = -1
md.NumArrays = 2
md.NumGhostCells = 1
md.NumGhostNodes = 0
md.NumLevels = 1
md.StaticMesh = 1
md.ArrayName = ['a1', 'a2']
md.ArrayCentering = [vtkDataObject.POINT, vtkDataObject.CELL]
md.ArrayComponents = [1, 3]
md.ArrayType = [VTK_DOUBLE, VTK_FLOAT]
md.ArrayRange = [[0.,1.],[0.,1.]]
md.BlockOwner = [0, 0]
md.BlockIds = [0,1]
md.BlockNumPoints = [2*4**2, 2*4**2]
md.BlockNumCells = [1*3**2, 1*3**2]
md.BlockCellArraySize = [-1,-1]
md.BlockExtents = [[0,1,0,2,0,2], [1,2,0,2,0,2]]
md.BlockBounds = [[0.,.5,0.,1.,0,1.], [0.5,1.,0.,1.,0.,1.]]
md.BlockArrayRange = [[[-1.,1.],[0.,1.]], [[1.,2.],[-1.,0.]]]
md.Flags = MeshMetadataFlags()
md.Flags.SetBlockDecomp()
md.Flags.SetBlockExtents()
md.Flags.SetBlockSize()
md.Flags.SetBlockBounds()
md.Flags.SetBlockArrayRange()


print(str(md))
