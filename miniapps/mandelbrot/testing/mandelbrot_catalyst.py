
#--------------------------------------------------------------

# Global timestep output options
timeStepToStartOutputAt=0
forceOutputAtFirstCall=False

# Global screenshot output options
imageFileNamePadding=5
rescale_lookuptable=False

# Whether or not to request specific arrays from the adaptor.
requestSpecificArrays=False

# a root directory under which all Catalyst output goes
rootDirectory=''

# makes a cinema D index table
make_cinema_table=False

#--------------------------------------------------------------
# Code generated from cpstate.py to create the CoProcessor.
# paraview version 5.6.0
#--------------------------------------------------------------

from paraview.simple import *
from paraview import coprocessing

# ----------------------- CoProcessor definition -----------------------

def CreateCoProcessor():
  def _CreatePipeline(coprocessor, datadescription):
    class Pipeline:
      # state file generated using paraview version 5.6.0

      # ----------------------------------------------------------------
      # setup views used in the visualization
      # ----------------------------------------------------------------

      # trace generated using paraview version 5.6.0
      #
      # To ensure correct image size when batch processing, please search 
      # for and uncomment the line `# renderView*.ViewSize = [*,*]`

      #### disable automatic camera reset on 'Show'
      paraview.simple._DisableFirstRenderCameraReset()

      # Create a new 'Render View'
      renderView1 = CreateView('RenderView')
      renderView1.ViewSize = [800, 800]
      renderView1.AxesGrid = 'GridAxes3DActor'
      renderView1.CenterOfRotation = [-1.3812499642372131, -0.6312499940395355, 0.0005000000237487257]
      renderView1.StereoType = 0
      renderView1.CameraPosition = [-1.3798994527052102, -0.6312816469711255, 0.6461113880908059]
      renderView1.CameraFocalPoint = [-1.3798994527052102, -0.6312816469711255, -0.27595779948725985]
      renderView1.CameraParallelScale = 0.23864906664741217

      # init the 'GridAxes3DActor' selected for 'AxesGrid'
      renderView1.AxesGrid.XTitleFontFile = ''
      renderView1.AxesGrid.YTitleFontFile = ''
      renderView1.AxesGrid.ZTitleFontFile = ''
      renderView1.AxesGrid.XLabelFontFile = ''
      renderView1.AxesGrid.YLabelFontFile = ''
      renderView1.AxesGrid.ZLabelFontFile = ''

      # register the view with coprocessor
      # and provide it with information such as the filename to use,
      # how frequently to write the images, etc.
      coprocessor.RegisterView(renderView1,
          filename='mandelbrot_catalyst_%t.png', freq=1, fittoscreen=0, magnification=1, width=800, height=800, cinema={})
      renderView1.ViewTime = datadescription.GetTime()

      # ----------------------------------------------------------------
      # restore active view
      SetActiveView(renderView1)
      # ----------------------------------------------------------------

      # ----------------------------------------------------------------
      # setup the data processing pipelines
      # ----------------------------------------------------------------

      # create a new 'PVD Reader'
      # create a producer from a simulation input
      meshpvd = coprocessor.CreateProducer(datadescription, 'mesh')

      # ----------------------------------------------------------------
      # setup the visualization in view 'renderView1'
      # ----------------------------------------------------------------

      # show data from meshpvd
      meshpvdDisplay = Show(meshpvd, renderView1)

      # get color transfer function/color map for 'mandelbrot'
      mandelbrotLUT = GetColorTransferFunction('mandelbrot')
      mandelbrotLUT.RGBPoints = [0.0, 0.0, 0.0, 0.5625, 3.3333300000000006, 0.0, 0.0, 1.0, 10.952385, 0.0, 1.0, 1.0, 14.761904999999999, 0.5, 1.0, 0.5, 18.571424999999998, 1.0, 1.0, 0.0, 26.19048, 1.0, 0.0, 0.0, 30.0, 0.5, 0.0, 0.0]
      mandelbrotLUT.ColorSpace = 'RGB'
      mandelbrotLUT.ScalarRangeInitialized = 1.0

      # trace defaults for the display properties.
      meshpvdDisplay.Representation = 'Surface With Edges'
      meshpvdDisplay.ColorArrayName = ['CELLS', 'mandelbrot']
      meshpvdDisplay.LookupTable = mandelbrotLUT
      meshpvdDisplay.EdgeColor = [0.0, 0.0, 0.0]
      meshpvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
      meshpvdDisplay.SelectOrientationVectors = 'None'
      meshpvdDisplay.ScaleFactor = 0.03375000208616257
      meshpvdDisplay.SelectScaleArray = 'mandelbrot'
      meshpvdDisplay.GlyphType = 'Arrow'
      meshpvdDisplay.GlyphTableIndexArray = 'mandelbrot'
      meshpvdDisplay.GaussianRadius = 0.0016875001043081283
      meshpvdDisplay.SetScaleArray = [None, '']
      meshpvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
      meshpvdDisplay.OpacityArray = [None, '']
      meshpvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
      meshpvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
      meshpvdDisplay.SelectionCellLabelFontFile = ''
      meshpvdDisplay.SelectionPointLabelFontFile = ''
      meshpvdDisplay.PolarAxes = 'PolarAxesRepresentation'

      # init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
      meshpvdDisplay.DataAxesGrid.XTitleFontFile = ''
      meshpvdDisplay.DataAxesGrid.YTitleFontFile = ''
      meshpvdDisplay.DataAxesGrid.ZTitleFontFile = ''
      meshpvdDisplay.DataAxesGrid.XLabelFontFile = ''
      meshpvdDisplay.DataAxesGrid.YLabelFontFile = ''
      meshpvdDisplay.DataAxesGrid.ZLabelFontFile = ''

      # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
      meshpvdDisplay.PolarAxes.PolarAxisTitleFontFile = ''
      meshpvdDisplay.PolarAxes.PolarAxisLabelFontFile = ''
      meshpvdDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
      meshpvdDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''

      # setup the color legend parameters for each legend in this view

      # get color legend/bar for mandelbrotLUT in view renderView1
      mandelbrotLUTColorBar = GetScalarBar(mandelbrotLUT, renderView1)
      mandelbrotLUTColorBar.Orientation = 'Horizontal'
      mandelbrotLUTColorBar.WindowLocation = 'AnyLocation'
      mandelbrotLUTColorBar.Position = [0.6225, 0.02875000000000004]
      mandelbrotLUTColorBar.Title = 'mandelbrot'
      mandelbrotLUTColorBar.ComponentTitle = ''
      mandelbrotLUTColorBar.TitleFontFile = ''
      mandelbrotLUTColorBar.TitleBold = 1
      mandelbrotLUTColorBar.TitleFontSize = 24
      mandelbrotLUTColorBar.LabelFontFile = ''
      mandelbrotLUTColorBar.LabelBold = 1
      mandelbrotLUTColorBar.LabelFontSize = 18
      mandelbrotLUTColorBar.ScalarBarLength = 0.3300000000000003

      # set color bar visibility
      mandelbrotLUTColorBar.Visibility = 1

      # show color legend
      meshpvdDisplay.SetScalarBarVisibility(renderView1, True)

      # ----------------------------------------------------------------
      # setup color maps and opacity mapes used in the visualization
      # note: the Get..() functions create a new object, if needed
      # ----------------------------------------------------------------

      # get opacity transfer function/opacity map for 'mandelbrot'
      mandelbrotPWF = GetOpacityTransferFunction('mandelbrot')
      mandelbrotPWF.Points = [0.0, 0.0, 0.5, 0.0, 30.0, 1.0, 0.5, 0.0]
      mandelbrotPWF.ScalarRangeInitialized = 1

      # ----------------------------------------------------------------
      # finally, restore active source
      SetActiveSource(meshpvd)
      # ----------------------------------------------------------------
    return Pipeline()

  class CoProcessor(coprocessing.CoProcessor):
    def CreatePipeline(self, datadescription):
      self.Pipeline = _CreatePipeline(self, datadescription)

  coprocessor = CoProcessor()
  # these are the frequencies at which the coprocessor updates.
  freqs = {'mesh': [1]}
  coprocessor.SetUpdateFrequencies(freqs)
  if requestSpecificArrays:
    arrays = [['mandelbrot', 1]]
    coprocessor.SetRequestedArrays('mesh', arrays)
  coprocessor.SetInitialOutputOptions(timeStepToStartOutputAt,forceOutputAtFirstCall)

  if rootDirectory:
      coprocessor.SetRootDirectory(rootDirectory)

  if make_cinema_table:
      coprocessor.EnableCinemaDTable()

  return coprocessor


#--------------------------------------------------------------
# Global variable that will hold the pipeline for each timestep
# Creating the CoProcessor object, doesn't actually create the ParaView pipeline.
# It will be automatically setup when coprocessor.UpdateProducers() is called the
# first time.
coprocessor = CreateCoProcessor()

#--------------------------------------------------------------
# Enable Live-Visualizaton with ParaView and the update frequency
coprocessor.EnableLiveVisualization(False, 1)

# ---------------------- Data Selection method ----------------------

def RequestDataDescription(datadescription):
    "Callback to populate the request for current timestep"
    global coprocessor

    # setup requests for all inputs based on the requirements of the
    # pipeline.
    coprocessor.LoadRequestedData(datadescription)

# ------------------------ Processing method ------------------------

def DoCoProcessing(datadescription):
    "Callback to do co-processing for current timestep"
    global coprocessor

    # Update the coprocessor by providing it the newly generated simulation data.
    # If the pipeline hasn't been setup yet, this will setup the pipeline.
    coprocessor.UpdateProducers(datadescription)

    # Write output data, if appropriate.
    coprocessor.WriteData(datadescription);

    # Write image capture (Last arg: rescale lookup table), if appropriate.
    coprocessor.WriteImages(datadescription, rescale_lookuptable=rescale_lookuptable,
        image_quality=0, padding_amount=imageFileNamePadding)

    # Live Visualization, if enabled.
    coprocessor.DoLiveVisualization(datadescription, "localhost", 22222)
