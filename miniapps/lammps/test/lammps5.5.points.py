
from paraview.simple import *
from paraview import coprocessing


#--------------------------------------------------------------
# Code generated from cpstate.py to create the CoProcessor.
# paraview version 5.5.2

#--------------------------------------------------------------
# Global screenshot output options
imageFileNamePadding=0
rescale_lookuptable=False


# ----------------------- CoProcessor definition -----------------------

def CreateCoProcessor():
  def _CreatePipeline(coprocessor, datadescription):
    class Pipeline:
      # state file generated using paraview version 5.5.2

      # ----------------------------------------------------------------
      # setup views used in the visualization
      # ----------------------------------------------------------------

      # trace generated using paraview version 5.5.2

      #### disable automatic camera reset on 'Show'
      paraview.simple._DisableFirstRenderCameraReset()

      # get the material library
      materialLibrary1 = GetMaterialLibrary()

      # Create a new 'Render View'
      renderView1 = CreateView('RenderView')
      renderView1.ViewSize = [701, 529]
      renderView1.AnnotationColor = [0.0, 0.0, 0.0]
      renderView1.AxesGrid = 'GridAxes3DActor'
      renderView1.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
      renderView1.OrientationAxesOutlineColor = [0.0, 0.0, 0.0]
      renderView1.CenterOfRotation = [0.0004106329081778881, 0.0005481718430679905, -0.002161462923972124]
      renderView1.StereoType = 0
      renderView1.CameraPosition = [189.76311827480347, 52.33555134242784, 120.01268709941407]
      renderView1.CameraFocalPoint = [0.0004106329081778881, 0.0005481718430679905, -0.002161462923972124]
      renderView1.CameraViewUp = [-0.08589751197441205, 0.9558550751342135, -0.28100301204217143]
      renderView1.CameraParallelScale = 59.67022518864623
      renderView1.Background = [1.0, 1.0, 1.0]
      renderView1.OSPRayMaterialLibrary = materialLibrary1

      # init the 'GridAxes3DActor' selected for 'AxesGrid'
      renderView1.AxesGrid.XTitleColor = [0.0, 0.0, 0.0]
      renderView1.AxesGrid.XTitleFontFile = ''
      renderView1.AxesGrid.YTitleColor = [0.0, 0.0, 0.0]
      renderView1.AxesGrid.YTitleFontFile = ''
      renderView1.AxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
      renderView1.AxesGrid.ZTitleFontFile = ''
      renderView1.AxesGrid.GridColor = [0.0, 0.0, 0.0]
      renderView1.AxesGrid.XLabelColor = [0.0, 0.0, 0.0]
      renderView1.AxesGrid.XLabelFontFile = ''
      renderView1.AxesGrid.YLabelColor = [0.0, 0.0, 0.0]
      renderView1.AxesGrid.YLabelFontFile = ''
      renderView1.AxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
      renderView1.AxesGrid.ZLabelFontFile = ''

      # register the view with coprocessor
      # and provide it with information such as the filename to use,
      # how frequently to write the images, etc.
      coprocessor.RegisterView(renderView1,
          filename='image_%t.png', freq=1, fittoscreen=0, magnification=1, width=701, height=529, cinema={})
      renderView1.ViewTime = datadescription.GetTime()

      # ----------------------------------------------------------------
      # restore active view
      SetActiveView(renderView1)
      # ----------------------------------------------------------------

      # ----------------------------------------------------------------
      # setup the data processing pipelines
      # ----------------------------------------------------------------

      # create a new 'XML MultiBlock Data Reader'
      # create a producer from a simulation input
      test_multiblockvtm = coprocessor.CreateProducer(datadescription, 'atoms')

      # ----------------------------------------------------------------
      # setup the visualization in view 'renderView1'
      # ----------------------------------------------------------------

      # show data from test_multiblockvtm
      test_multiblockvtmDisplay = Show(test_multiblockvtm, renderView1)

      # get color transfer function/color map for 'type'
      typeLUT = GetColorTransferFunction('type')
      typeLUT.RGBPoints = [1.0, 0.231373, 0.298039, 0.752941, 34.5, 0.865003, 0.865003, 0.865003, 68.0, 0.705882, 0.0156863, 0.14902]
      typeLUT.ScalarRangeInitialized = 1.0

      # trace defaults for the display properties.
      test_multiblockvtmDisplay.Representation = 'Points'
      test_multiblockvtmDisplay.AmbientColor = [0.0, 0.0, 0.0]
      test_multiblockvtmDisplay.ColorArrayName = ['POINTS', 'type']
      test_multiblockvtmDisplay.LookupTable = typeLUT
      test_multiblockvtmDisplay.OSPRayScaleArray = 'id'
      test_multiblockvtmDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
      test_multiblockvtmDisplay.SelectOrientationVectors = 'None'
      test_multiblockvtmDisplay.ScaleFactor = 7.6998101099470695
      test_multiblockvtmDisplay.SelectScaleArray = 'None'
      test_multiblockvtmDisplay.GlyphType = 'Arrow'
      test_multiblockvtmDisplay.GlyphTableIndexArray = 'None'
      test_multiblockvtmDisplay.GaussianRadius = 0.3849905054973535
      test_multiblockvtmDisplay.SetScaleArray = ['POINTS', 'id']
      test_multiblockvtmDisplay.ScaleTransferFunction = 'PiecewiseFunction'
      test_multiblockvtmDisplay.OpacityArray = ['POINTS', 'id']
      test_multiblockvtmDisplay.OpacityTransferFunction = 'PiecewiseFunction'
      test_multiblockvtmDisplay.DataAxesGrid = 'GridAxesRepresentation'
      test_multiblockvtmDisplay.SelectionCellLabelFontFile = ''
      test_multiblockvtmDisplay.SelectionPointLabelFontFile = ''
      test_multiblockvtmDisplay.PolarAxes = 'PolarAxesRepresentation'

      # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
      test_multiblockvtmDisplay.ScaleTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 32000.0, 1.0, 0.5, 0.0]

      # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
      test_multiblockvtmDisplay.OpacityTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 32000.0, 1.0, 0.5, 0.0]

      # init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
      test_multiblockvtmDisplay.DataAxesGrid.XTitleColor = [0.0, 0.0, 0.0]
      test_multiblockvtmDisplay.DataAxesGrid.XTitleFontFile = ''
      test_multiblockvtmDisplay.DataAxesGrid.YTitleColor = [0.0, 0.0, 0.0]
      test_multiblockvtmDisplay.DataAxesGrid.YTitleFontFile = ''
      test_multiblockvtmDisplay.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
      test_multiblockvtmDisplay.DataAxesGrid.ZTitleFontFile = ''
      test_multiblockvtmDisplay.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
      test_multiblockvtmDisplay.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
      test_multiblockvtmDisplay.DataAxesGrid.XLabelFontFile = ''
      test_multiblockvtmDisplay.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
      test_multiblockvtmDisplay.DataAxesGrid.YLabelFontFile = ''
      test_multiblockvtmDisplay.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
      test_multiblockvtmDisplay.DataAxesGrid.ZLabelFontFile = ''

      # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
      test_multiblockvtmDisplay.PolarAxes.PolarAxisTitleColor = [0.0, 0.0, 0.0]
      test_multiblockvtmDisplay.PolarAxes.PolarAxisTitleFontFile = ''
      test_multiblockvtmDisplay.PolarAxes.PolarAxisLabelColor = [0.0, 0.0, 0.0]
      test_multiblockvtmDisplay.PolarAxes.PolarAxisLabelFontFile = ''
      test_multiblockvtmDisplay.PolarAxes.LastRadialAxisTextColor = [0.0, 0.0, 0.0]
      test_multiblockvtmDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
      test_multiblockvtmDisplay.PolarAxes.SecondaryRadialAxesTextColor = [0.0, 0.0, 0.0]
      test_multiblockvtmDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''

      # setup the color legend parameters for each legend in this view

      # get color legend/bar for typeLUT in view renderView1
      typeLUTColorBar = GetScalarBar(typeLUT, renderView1)
      typeLUTColorBar.Title = 'type'
      typeLUTColorBar.ComponentTitle = ''
      typeLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
      typeLUTColorBar.TitleFontFile = ''
      typeLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
      typeLUTColorBar.LabelFontFile = ''

      # set color bar visibility
      typeLUTColorBar.Visibility = 1

      # show color legend
      test_multiblockvtmDisplay.SetScalarBarVisibility(renderView1, True)

      # ----------------------------------------------------------------
      # setup color maps and opacity mapes used in the visualization
      # note: the Get..() functions create a new object, if needed
      # ----------------------------------------------------------------

      # get opacity transfer function/opacity map for 'type'
      typePWF = GetOpacityTransferFunction('type')
      typePWF.Points = [1.0, 0.0, 0.5, 0.0, 68.0, 1.0, 0.5, 0.0]
      typePWF.ScalarRangeInitialized = 1

      # ----------------------------------------------------------------
      # finally, restore active source
      SetActiveSource(test_multiblockvtm)
      # ----------------------------------------------------------------
    return Pipeline()

  class CoProcessor(coprocessing.CoProcessor):
    def CreatePipeline(self, datadescription):
      self.Pipeline = _CreatePipeline(self, datadescription)

  coprocessor = CoProcessor()
  # these are the frequencies at which the coprocessor updates.
  freqs = {'atoms': [1]}
  coprocessor.SetUpdateFrequencies(freqs)
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
    if datadescription.GetForceOutput() == True:
        # We are just going to request all fields and meshes from the simulation
        # code/adaptor.
        for i in range(datadescription.GetNumberOfInputDescriptions()):
            datadescription.GetInputDescription(i).AllFieldsOn()
            datadescription.GetInputDescription(i).GenerateMeshOn()
        return

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
