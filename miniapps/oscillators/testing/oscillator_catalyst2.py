# script-version: 2.0
# Catalyst state generated using paraview version 5.9.0-489-g30de93d505

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [2674, 1566]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [32.0, 32.0, 32.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [172.21574697572467, 95.74147107235041, -65.9154527015831]
renderView1.CameraFocalPoint = [32.000000000000014, 32.00000000000009, 32.000000000000014]
renderView1.CameraViewUp = [-0.31526110210101105, 0.9358040708864755, 0.15773768863006818]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 57.15767664977295

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(2674, 1566)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVD Reader'
meshpvd = PVDReader(registrationName='mesh')
meshpvd.CellArrays = ['vtkGhostType', 'data']

# create a new 'Cell Data to Point Data'
cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=meshpvd)
cellDatatoPointData1.CellDataArraytoprocess = ['data', 'vtkGhostType']

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=cellDatatoPointData1)
contour1.ContourBy = ['POINTS', 'data']
contour1.Isosurfaces = [0.05115328840111033, 1.0, 0.5, 1.5, 0.1]
contour1.PointMergeMethod = 'Uniform Binning'

# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=contour1)
clip1.ClipType = 'Plane'
clip1.HyperTreeGridClipper = 'Plane'
clip1.Scalars = ['POINTS', 'data']
clip1.Value = 0.5255766436457634

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [32.0, 32.0, 29.73122787475586]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip1.HyperTreeGridClipper.Origin = [32.0, 32.0, 29.73122787475586]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from clip1
clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'data'
dataLUT = GetColorTransferFunction('data')
dataLUT.AutomaticRescaleRangeMode = 'Never'
dataLUT.RGBPoints = [-0.9384190440177917, 0.231373, 0.298039, 0.752941, 0.369352787733078, 0.865003, 0.865003, 0.865003, 1.6771246194839478, 0.705882, 0.0156863, 0.14902]
dataLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'data'
dataPWF = GetOpacityTransferFunction('data')
dataPWF.Points = [-0.9384190440177917, 0.0, 0.5, 0.0, 1.6771246194839478, 1.0, 0.5, 0.0]
dataPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display.Representation = 'Surface'
clip1Display.ColorArrayName = ['POINTS', 'data']
clip1Display.LookupTable = dataLUT
clip1Display.SelectTCoordArray = 'None'
clip1Display.SelectNormalArray = 'Normals'
clip1Display.SelectTangentArray = 'None'
clip1Display.OSPRayScaleArray = 'data'
clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display.SelectOrientationVectors = 'None'
clip1Display.ScaleFactor = 6.519329071044922
clip1Display.SelectScaleArray = 'data'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'data'
clip1Display.GaussianRadius = 0.3259664535522461
clip1Display.SetScaleArray = ['POINTS', 'data']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = ['POINTS', 'data']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = dataPWF
clip1Display.ScalarOpacityUnitDistance = 2.6204150037873086
clip1Display.OpacityArrayName = ['POINTS', 'data']
clip1Display.ExtractedBlockIndex = 2

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip1Display.ScaleTransferFunction.Points = [0.051153287291526794, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip1Display.OpacityTransferFunction.Points = [0.051153287291526794, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for dataLUT in view renderView1
dataLUTColorBar = GetScalarBar(dataLUT, renderView1)
dataLUTColorBar.Title = 'data'
dataLUTColorBar.ComponentTitle = ''

# set color bar visibility
dataLUTColorBar.Visibility = 1

# show color legend
clip1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
# trace defaults for the extractor.
pNG1.Trigger = 'TimeStep'

# init the 'PNG' selected for 'Writer'
pNG1.Writer.FileName = 'RenderView1_%.6ts%cm.png'
pNG1.Writer.ImageResolution = [2674, 1566]
pNG1.Writer.Format = 'PNG'

# ----------------------------------------------------------------
# restore active source
SetActiveSource(pNG1)
# ----------------------------------------------------------------

# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst
options = catalyst.Options()
options.GlobalTrigger = 'TimeStep'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'TimeStep'

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
