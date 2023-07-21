# script-version: 2.0
# Catalyst state generated using paraview version 5.11.0

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.ViewSize = [787, 796]
renderView2.AxesGrid = 'GridAxes3DActor'
renderView2.CenterOfRotation = [32.0, 32.0, 32.0]
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraPosition = [146.56668679922967, 174.4861712671072, -53.23173300581486]
renderView2.CameraFocalPoint = [31.9999999999999, 31.999999999999822, 32.0]
renderView2.CameraViewUp = [0.754371100608003, -0.6520311704631508, -0.07602364968826322]
renderView2.CameraFocalDisk = 1.0
renderView2.CameraParallelScale = 77.64080309423083
renderView2.BackEnd = 'OSPRay raycaster'
renderView2.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView2)
layout1.SetSize(787, 796)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView2)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVTrivialProducer'
particles = PVTrivialProducer(registrationName='particles')

# create a new 'PVTrivialProducer'
oscillators = PVTrivialProducer(registrationName='oscillators')

# create a new 'PVTrivialProducer'
mesh = PVTrivialProducer(registrationName='mesh')

ucdmesh = PVTrivialProducer(registrationName='ucdmesh')

# ----------------------------------------------------------------
# setup the visualization in view 'renderView2'
# ----------------------------------------------------------------

# show data from ucdmesh
ucdmeshDisplay = Show(ucdmesh, renderView2, 'UnstructuredGridRepresentation')
ucdmeshDisplay.Representation = 'Outline'


# show data from mesh
meshDisplay = Show(mesh, renderView2, 'UniformGridRepresentation')

# get 2D transfer function for 'data'
dataTF2D = GetTransferFunction2D('data')
dataTF2D.ScalarRangeInitialized = 1
dataTF2D.Range = [0.00020615989342331886, 1.4246151447296143, 0.0, 1.0]

# get color transfer function/color map for 'data'
dataLUT = GetColorTransferFunction('data')
dataLUT.TransferFunction2D = dataTF2D
dataLUT.RGBPoints = [0.00020615989342331886, 0.231373, 0.298039, 0.752941, 0.7124106523115188, 0.865003, 0.865003, 0.865003, 1.4246151447296143, 0.705882, 0.0156863, 0.14902]
dataLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'data'
dataPWF = GetOpacityTransferFunction('data')
dataPWF.Points = [0.00020615989342331886, 0.0, 0.5, 0.0, 1.4246151447296143, 0.1304347813129425, 0.5, 0.0]
dataPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
meshDisplay.Representation = 'Volume'
meshDisplay.ColorArrayName = ['CELLS', 'data']
meshDisplay.LookupTable = dataLUT
meshDisplay.SelectTCoordArray = 'None'
meshDisplay.SelectNormalArray = 'None'
meshDisplay.SelectTangentArray = 'None'
meshDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
meshDisplay.SelectOrientationVectors = 'None'
meshDisplay.ScaleFactor = 6.6000000000000005
meshDisplay.SelectScaleArray = 'None'
meshDisplay.GlyphType = 'Arrow'
meshDisplay.GlyphTableIndexArray = 'None'
meshDisplay.GaussianRadius = 0.33
meshDisplay.SetScaleArray = [None, '']
meshDisplay.ScaleTransferFunction = 'PiecewiseFunction'
meshDisplay.OpacityArray = [None, '']
meshDisplay.OpacityTransferFunction = 'PiecewiseFunction'
meshDisplay.DataAxesGrid = 'GridAxesRepresentation'
meshDisplay.PolarAxes = 'PolarAxesRepresentation'
meshDisplay.ScalarOpacityUnitDistance = 1.7320508075688774
meshDisplay.ScalarOpacityFunction = dataPWF
meshDisplay.TransferFunction2D = dataTF2D
meshDisplay.OpacityArrayName = ['CELLS', 'data']
meshDisplay.ColorArray2Name = ['CELLS', 'data']
meshDisplay.SliceFunction = 'Plane'
meshDisplay.Slice = 33
meshDisplay.SelectInputVectors = [None, '']
meshDisplay.WriteLog = ''
# meshDisplay.custom_kernel = ''

# init the 'Plane' selected for 'SliceFunction'
meshDisplay.SliceFunction.Origin = [32.0, 32.0, 32.0]

# show data from oscillators
oscillatorsDisplay = Show(oscillators, renderView2, 'GeometryRepresentation')

# trace defaults for the display properties.
oscillatorsDisplay.Representation = 'Point Gaussian'
oscillatorsDisplay.ColorArrayName = [None, '']
oscillatorsDisplay.SelectTCoordArray = 'None'
oscillatorsDisplay.SelectNormalArray = 'None'
oscillatorsDisplay.SelectTangentArray = 'None'
oscillatorsDisplay.OSPRayScaleArray = 'omega0'
oscillatorsDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
oscillatorsDisplay.SelectOrientationVectors = 'None'
oscillatorsDisplay.ScaleFactor = 3.2
oscillatorsDisplay.SelectScaleArray = 'None'
oscillatorsDisplay.GlyphType = 'Arrow'
oscillatorsDisplay.GlyphTableIndexArray = 'None'
oscillatorsDisplay.GaussianRadius = 1.28
oscillatorsDisplay.SetScaleArray = ['POINTS', 'omega0']
oscillatorsDisplay.ScaleTransferFunction = 'PiecewiseFunction'
oscillatorsDisplay.OpacityArray = ['POINTS', 'omega0']
oscillatorsDisplay.OpacityTransferFunction = 'PiecewiseFunction'
oscillatorsDisplay.DataAxesGrid = 'GridAxesRepresentation'
oscillatorsDisplay.PolarAxes = 'PolarAxesRepresentation'
oscillatorsDisplay.SelectInputVectors = [None, '']
oscillatorsDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
oscillatorsDisplay.ScaleTransferFunction.Points = [3.140000104904175, 0.0, 0.5, 0.0, 9.5, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
oscillatorsDisplay.OpacityTransferFunction.Points = [3.140000104904175, 0.0, 0.5, 0.0, 9.5, 1.0, 0.5, 0.0]

# show data from particles
particlesDisplay = Show(particles, renderView2, 'GeometryRepresentation')

# get 2D transfer function for 'velocity'
velocityTF2D = GetTransferFunction2D('velocity')

# get color transfer function/color map for 'velocity'
velocityLUT = GetColorTransferFunction('velocity')
velocityLUT.TransferFunction2D = velocityTF2D
velocityLUT.RGBPoints = [0.008004465007019556, 0.231373, 0.298039, 0.752941, 6.081689554456711, 0.865003, 0.865003, 0.865003, 12.155374643906402, 0.705882, 0.0156863, 0.14902]
velocityLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
particlesDisplay.Representation = '3D Glyphs'
particlesDisplay.ColorArrayName = ['POINTS', 'velocity']
particlesDisplay.LookupTable = velocityLUT
particlesDisplay.SelectTCoordArray = 'None'
particlesDisplay.SelectNormalArray = 'None'
particlesDisplay.SelectTangentArray = 'None'
particlesDisplay.OSPRayScaleArray = 'pid'
particlesDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
particlesDisplay.Orient = 1
particlesDisplay.SelectOrientationVectors = 'velocity'
particlesDisplay.Scaling = 1
particlesDisplay.ScaleMode = 'Magnitude'
particlesDisplay.ScaleFactor = 6.241652202606201
particlesDisplay.SelectScaleArray = 'None'
particlesDisplay.GlyphType = 'Arrow'
particlesDisplay.GlyphTableIndexArray = 'None'
particlesDisplay.GaussianRadius = 0.3120826101303101
particlesDisplay.SetScaleArray = ['POINTS', 'pid']
particlesDisplay.ScaleTransferFunction = 'PiecewiseFunction'
particlesDisplay.OpacityArray = ['POINTS', 'pid']
particlesDisplay.OpacityTransferFunction = 'PiecewiseFunction'
particlesDisplay.DataAxesGrid = 'GridAxesRepresentation'
particlesDisplay.PolarAxes = 'PolarAxesRepresentation'
particlesDisplay.SelectInputVectors = ['POINTS', 'velocity']
particlesDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
particlesDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 99.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
particlesDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 99.0, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for dataLUT in view renderView2
dataLUTColorBar = GetScalarBar(dataLUT, renderView2)
dataLUTColorBar.Title = 'data'
dataLUTColorBar.ComponentTitle = ''

# set color bar visibility
dataLUTColorBar.Visibility = 1

# get color legend/bar for velocityLUT in view renderView2
velocityLUTColorBar = GetScalarBar(velocityLUT, renderView2)
velocityLUTColorBar.WindowLocation = 'Upper Right Corner'
velocityLUTColorBar.Title = 'velocity'
velocityLUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
velocityLUTColorBar.Visibility = 1

# show color legend
meshDisplay.SetScalarBarVisibility(renderView2, True)

# show color legend
particlesDisplay.SetScalarBarVisibility(renderView2, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'velocity'
velocityPWF = GetOpacityTransferFunction('velocity')
velocityPWF.Points = [0.008004465007019556, 0.0, 0.5, 0.0, 12.155374643906402, 1.0, 0.5, 0.0]
velocityPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
pNG1 = CreateExtractor('PNG', renderView2, registrationName='PNG1')
# trace defaults for the extractor.
pNG1.Trigger = 'TimeStep'

# init the 'TimeStep' selected for 'Trigger'
pNG1.Trigger.Frequency = 10

# init the 'PNG' selected for 'Writer'
pNG1.Writer.FileName = 'RenderView2_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [787, 796]
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
