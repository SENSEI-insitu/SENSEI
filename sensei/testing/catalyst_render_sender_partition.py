
#--------------------------------------------------------------

# Global timestep output options
timeStepToStartOutputAt=0
forceOutputAtFirstCall=False

# Global screenshot output options
imageFileNamePadding=5
rescale_lookuptable=True

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
      renderView1.ViewSize = [800, 600]
      renderView1.AxesGrid = 'GridAxes3DActor'
      renderView1.CenterOfRotation = [0.0, 0.0, 4.495000000111759]
      renderView1.StereoType = 0
      renderView1.CameraPosition = [21.36058439968015, -14.840635549491012, 15.796343846677791]
      renderView1.CameraFocalPoint = [-6.088433777305516, 4.554345933565786, -2.9664775629433247]
      renderView1.CameraViewUp = [-0.38455457009543387, 0.3002198736328352, 0.8729179858924896]
      renderView1.CameraParallelScale = 9.962541047806159
      renderView1.Background = [0.0, 0.0, 0.0]

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
          filename='sender_decomp_%t.png', freq=1, fittoscreen=0, magnification=1, width=800, height=600, cinema={})
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

      extractEdges1 = ExtractEdges(Input=meshpvd)

      # create a new 'Slice'
      slice2 = Slice(Input=meshpvd)
      slice2.SliceType = 'Plane'
      slice2.SliceOffsetValues = [0.0]

      # init the 'Plane' selected for 'SliceType'
      slice2.SliceType.Origin = [-4.440892098500626e-16, -4.440892098500626e-16, 0.0]
      slice2.SliceType.Normal = [0.0, 0.0, 1.0]

      # create a new 'Slice'
      slice1 = Slice(Input=meshpvd)
      slice1.SliceType = 'Plane'
      slice1.SliceOffsetValues = [0.0]

      # init the 'Plane' selected for 'SliceType'
      slice1.SliceType.Origin = [-4.440892098500626e-16, -4.440892098500626e-16, 0.0]
      slice1.SliceType.Normal = [0.0, 0.0, 1.0]

      # create a new 'Warp By Scalar'
      warpByScalar1 = WarpByScalar(Input=slice1)
      warpByScalar1.Scalars = ['POINTS', 'f_xyt']

      # create a new 'Outline'
      outline1 = Outline(Input=slice2)

      # ----------------------------------------------------------------
      # setup the visualization in view 'renderView1'
      # ----------------------------------------------------------------

      # show data from mesh
      extractEdges1Display = Show(extractEdges1, renderView1)
      extractEdges1Display.Representation = 'Surface'
      extractEdges1Display.ColorArrayName = [None, '']
      extractEdges1Display.Opacity = 0.5

      # show data from warpByScalar1
      warpByScalar1Display = Show(warpByScalar1, renderView1)

      # get color transfer function/color map for 'f_xyt'
      f_xytLUT = GetColorTransferFunction('f_xyt')
      f_xytLUT.RGBPoints = [-1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
      f_xytLUT.ColorSpace = 'RGB'
      f_xytLUT.NanColor = [1.0, 0.0, 0.0]
      f_xytLUT.ScalarRangeInitialized = 1.0

      # trace defaults for the display properties.
      warpByScalar1Display.Representation = 'Surface'
      warpByScalar1Display.ColorArrayName = ['POINTS', 'f_xyt']
      warpByScalar1Display.LookupTable = f_xytLUT
      warpByScalar1Display.Opacity = 0.73
      warpByScalar1Display.Position = [0.0, 0.0, 4.0]
      warpByScalar1Display.OSPRayScaleArray = 'f_xyt'
      warpByScalar1Display.OSPRayScaleFunction = 'PiecewiseFunction'
      warpByScalar1Display.SelectOrientationVectors = 'None'
      warpByScalar1Display.ScaleFactor = 1.2566399574279785
      warpByScalar1Display.SelectScaleArray = 'None'
      warpByScalar1Display.GlyphType = 'Arrow'
      warpByScalar1Display.GlyphTableIndexArray = 'None'
      warpByScalar1Display.GaussianRadius = 0.06283199787139893
      warpByScalar1Display.SetScaleArray = ['POINTS', 'f_xyt']
      warpByScalar1Display.ScaleTransferFunction = 'PiecewiseFunction'
      warpByScalar1Display.OpacityArray = ['POINTS', 'f_xyt']
      warpByScalar1Display.OpacityTransferFunction = 'PiecewiseFunction'
      warpByScalar1Display.DataAxesGrid = 'GridAxesRepresentation'
      warpByScalar1Display.SelectionCellLabelFontFile = ''
      warpByScalar1Display.SelectionPointLabelFontFile = ''
      warpByScalar1Display.PolarAxes = 'PolarAxesRepresentation'

      # init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
      warpByScalar1Display.DataAxesGrid.XTitleFontFile = ''
      warpByScalar1Display.DataAxesGrid.YTitleFontFile = ''
      warpByScalar1Display.DataAxesGrid.ZTitleFontFile = ''
      warpByScalar1Display.DataAxesGrid.XLabelFontFile = ''
      warpByScalar1Display.DataAxesGrid.YLabelFontFile = ''
      warpByScalar1Display.DataAxesGrid.ZLabelFontFile = ''

      # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
      warpByScalar1Display.PolarAxes.Translation = [0.0, 0.0, 4.0]
      warpByScalar1Display.PolarAxes.PolarAxisTitleFontFile = ''
      warpByScalar1Display.PolarAxes.PolarAxisLabelFontFile = ''
      warpByScalar1Display.PolarAxes.LastRadialAxisTextFontFile = ''
      warpByScalar1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

      # show data from slice2
      slice2Display = Show(slice2, renderView1)

      # get color transfer function/color map for 'SenderBlockOwner'
      receiverBlockOwnerLUT = GetColorTransferFunction('SenderBlockOwner')
      receiverBlockOwnerLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 1.5, 0.865003, 0.865003, 0.865003, 3.0, 0.705882, 0.0156863, 0.14902]
      receiverBlockOwnerLUT.ScalarRangeInitialized = 1.0

      # trace defaults for the display properties.
      slice2Display.Representation = 'Surface'
      slice2Display.ColorArrayName = ['CELLS', 'SenderBlockOwner']
      slice2Display.LookupTable = receiverBlockOwnerLUT
      slice2Display.OSPRayScaleArray = 'f_xyt'
      slice2Display.OSPRayScaleFunction = 'PiecewiseFunction'
      slice2Display.SelectOrientationVectors = 'None'
      slice2Display.ScaleFactor = 1.2566399574279785
      slice2Display.SelectScaleArray = 'None'
      slice2Display.GlyphType = 'Arrow'
      slice2Display.GlyphTableIndexArray = 'None'
      slice2Display.GaussianRadius = 0.06283199787139893
      slice2Display.SetScaleArray = ['POINTS', 'f_xyt']
      slice2Display.ScaleTransferFunction = 'PiecewiseFunction'
      slice2Display.OpacityArray = ['POINTS', 'f_xyt']
      slice2Display.OpacityTransferFunction = 'PiecewiseFunction'
      slice2Display.DataAxesGrid = 'GridAxesRepresentation'
      slice2Display.SelectionCellLabelFontFile = ''
      slice2Display.SelectionPointLabelFontFile = ''
      slice2Display.PolarAxes = 'PolarAxesRepresentation'

      # init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
      slice2Display.DataAxesGrid.XTitleFontFile = ''
      slice2Display.DataAxesGrid.YTitleFontFile = ''
      slice2Display.DataAxesGrid.ZTitleFontFile = ''
      slice2Display.DataAxesGrid.XLabelFontFile = ''
      slice2Display.DataAxesGrid.YLabelFontFile = ''
      slice2Display.DataAxesGrid.ZLabelFontFile = ''

      # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
      slice2Display.PolarAxes.PolarAxisTitleFontFile = ''
      slice2Display.PolarAxes.PolarAxisLabelFontFile = ''
      slice2Display.PolarAxes.LastRadialAxisTextFontFile = ''
      slice2Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

      # show data from outline1
      outline1Display = Show(outline1, renderView1)

      # trace defaults for the display properties.
      outline1Display.Representation = 'Surface'
      outline1Display.ColorArrayName = [None, '']
      outline1Display.LineWidth = 3.0
      outline1Display.OSPRayScaleFunction = 'PiecewiseFunction'
      outline1Display.SelectOrientationVectors = 'None'
      outline1Display.ScaleFactor = 1.2566399574279785
      outline1Display.SelectScaleArray = 'None'
      outline1Display.GlyphType = 'Arrow'
      outline1Display.GlyphTableIndexArray = 'None'
      outline1Display.GaussianRadius = 0.06283199787139893
      outline1Display.SetScaleArray = [None, '']
      outline1Display.ScaleTransferFunction = 'PiecewiseFunction'
      outline1Display.OpacityArray = [None, '']
      outline1Display.OpacityTransferFunction = 'PiecewiseFunction'
      outline1Display.DataAxesGrid = 'GridAxesRepresentation'
      outline1Display.SelectionCellLabelFontFile = ''
      outline1Display.SelectionPointLabelFontFile = ''
      outline1Display.PolarAxes = 'PolarAxesRepresentation'

      # init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
      outline1Display.DataAxesGrid.XTitleFontFile = ''
      outline1Display.DataAxesGrid.YTitleFontFile = ''
      outline1Display.DataAxesGrid.ZTitleFontFile = ''
      outline1Display.DataAxesGrid.XLabelFontFile = ''
      outline1Display.DataAxesGrid.YLabelFontFile = ''
      outline1Display.DataAxesGrid.ZLabelFontFile = ''

      # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
      outline1Display.PolarAxes.PolarAxisTitleFontFile = ''
      outline1Display.PolarAxes.PolarAxisLabelFontFile = ''
      outline1Display.PolarAxes.LastRadialAxisTextFontFile = ''
      outline1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

      # ----------------------------------------------------------------
      # setup color maps and opacity mapes used in the visualization
      # note: the Get..() functions create a new object, if needed
      # ----------------------------------------------------------------

      # get opacity transfer function/opacity map for 'f_xyt'
      f_xytPWF = GetOpacityTransferFunction('f_xyt')
      f_xytPWF.Points = [-1.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
      f_xytPWF.ScalarRangeInitialized = 1

      # get opacity transfer function/opacity map for 'SenderBlockOwner'
      receiverBlockOwnerPWF = GetOpacityTransferFunction('SenderBlockOwner')
      receiverBlockOwnerPWF.Points = [0.0, 0.0, 0.5, 0.0, 3.0, 1.0, 0.5, 0.0]
      receiverBlockOwnerPWF.ScalarRangeInitialized = 1

      # ----------------------------------------------------------------
      # finally, restore active source
      SetActiveSource(outline1)
      # ----------------------------------------------------------------
    return Pipeline()

  class CoProcessor(coprocessing.CoProcessor):
    def CreatePipeline(self, datadescription):
      self.Pipeline = _CreatePipeline(self, datadescription)

  coprocessor = CoProcessor()
  # these are the frequencies at which the coprocessor updates.
  freqs = {'mesh': [1, 1, 1, 1, 1]}
  coprocessor.SetUpdateFrequencies(freqs)
  if requestSpecificArrays:
    arrays = [['f_xyt', 0], ['ReceiverBlockOwner', 1], ['SenderBlockOwner', 1]]
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
