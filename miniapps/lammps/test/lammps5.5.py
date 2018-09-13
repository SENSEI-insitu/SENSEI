
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
      renderView1.CenterOfRotation = [0.0004100799560546875, 0.0005474090576171875, -0.0021610260009765625]
      renderView1.StereoType = 0
      renderView1.CameraPosition = [171.6802520599626, 30.24467261981238, 158.44182161228704]
      renderView1.CameraFocalPoint = [0.0004100799560546875, 0.0005474090576171875, -0.0021610260009765625]
      renderView1.CameraViewUp = [0.033652204088957506, 0.974362722136786, -0.22245182595372806]
      renderView1.CameraParallelScale = 60.96997278257824
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

      # create a new 'Glyph'
      glyph1 = Glyph(Input=test_multiblockvtm,
          GlyphType='Sphere')
      glyph1.Scalars = ['POINTS', 'type']
      glyph1.Vectors = ['POINTS', 'None']
      glyph1.ScaleFactor = 7.6998101099470695
      glyph1.GlyphMode = 'All Points'
      glyph1.GlyphTransform = 'Transform2'

      # init the 'Sphere' selected for 'GlyphType'
      glyph1.GlyphType.Radius = 0.1

      # ----------------------------------------------------------------
      # setup the visualization in view 'renderView1'
      # ----------------------------------------------------------------

      # show data from glyph1
      glyph1Display = Show(glyph1, renderView1)

      # get color transfer function/color map for 'GlyphScale'
      glyphScaleLUT = GetColorTransferFunction('GlyphScale')
      glyphScaleLUT.RGBPoints = [1.0, 0.231373, 0.298039, 0.752941, 34.5, 0.865003, 0.865003, 0.865003, 68.0, 0.705882, 0.0156863, 0.14902]
      glyphScaleLUT.ScalarRangeInitialized = 1.0

      # trace defaults for the display properties.
      glyph1Display.Representation = 'Surface'
      glyph1Display.AmbientColor = [0.0, 0.0, 0.0]
      glyph1Display.ColorArrayName = ['POINTS', 'GlyphScale']
      glyph1Display.LookupTable = glyphScaleLUT
      glyph1Display.OSPRayScaleArray = 'GlyphScale'
      glyph1Display.OSPRayScaleFunction = 'PiecewiseFunction'
      glyph1Display.SelectOrientationVectors = 'GlyphScale'
      glyph1Display.ScaleFactor = 7.714823532104493
      glyph1Display.SelectScaleArray = 'GlyphScale'
      glyph1Display.GlyphType = 'Arrow'
      glyph1Display.GlyphTableIndexArray = 'GlyphScale'
      glyph1Display.GaussianRadius = 0.3857411766052246
      glyph1Display.SetScaleArray = ['POINTS', 'GlyphScale']
      glyph1Display.ScaleTransferFunction = 'PiecewiseFunction'
      glyph1Display.OpacityArray = ['POINTS', 'GlyphScale']
      glyph1Display.OpacityTransferFunction = 'PiecewiseFunction'
      glyph1Display.DataAxesGrid = 'GridAxesRepresentation'
      glyph1Display.SelectionCellLabelFontFile = ''
      glyph1Display.SelectionPointLabelFontFile = ''
      glyph1Display.PolarAxes = 'PolarAxesRepresentation'

      # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
      glyph1Display.ScaleTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 68.0, 1.0, 0.5, 0.0]

      # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
      glyph1Display.OpacityTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 68.0, 1.0, 0.5, 0.0]

      # init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
      glyph1Display.DataAxesGrid.XTitleColor = [0.0, 0.0, 0.0]
      glyph1Display.DataAxesGrid.XTitleFontFile = ''
      glyph1Display.DataAxesGrid.YTitleColor = [0.0, 0.0, 0.0]
      glyph1Display.DataAxesGrid.YTitleFontFile = ''
      glyph1Display.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
      glyph1Display.DataAxesGrid.ZTitleFontFile = ''
      glyph1Display.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
      glyph1Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
      glyph1Display.DataAxesGrid.XLabelFontFile = ''
      glyph1Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
      glyph1Display.DataAxesGrid.YLabelFontFile = ''
      glyph1Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
      glyph1Display.DataAxesGrid.ZLabelFontFile = ''

      # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
      glyph1Display.PolarAxes.PolarAxisTitleColor = [0.0, 0.0, 0.0]
      glyph1Display.PolarAxes.PolarAxisTitleFontFile = ''
      glyph1Display.PolarAxes.PolarAxisLabelColor = [0.0, 0.0, 0.0]
      glyph1Display.PolarAxes.PolarAxisLabelFontFile = ''
      glyph1Display.PolarAxes.LastRadialAxisTextColor = [0.0, 0.0, 0.0]
      glyph1Display.PolarAxes.LastRadialAxisTextFontFile = ''
      glyph1Display.PolarAxes.SecondaryRadialAxesTextColor = [0.0, 0.0, 0.0]
      glyph1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

      # setup the color legend parameters for each legend in this view

      # get color legend/bar for glyphScaleLUT in view renderView1
      glyphScaleLUTColorBar = GetScalarBar(glyphScaleLUT, renderView1)
      glyphScaleLUTColorBar.WindowLocation = 'AnyLocation'
      glyphScaleLUTColorBar.Position = [0.8102710413694723, 0.5595463137996218]
      glyphScaleLUTColorBar.Title = 'GlyphScale'
      glyphScaleLUTColorBar.ComponentTitle = ''
      glyphScaleLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
      glyphScaleLUTColorBar.TitleFontFile = ''
      glyphScaleLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
      glyphScaleLUTColorBar.LabelFontFile = ''
      glyphScaleLUTColorBar.ScalarBarLength = 0.32999999999999974

      # set color bar visibility
      glyphScaleLUTColorBar.Visibility = 1

      # show color legend
      glyph1Display.SetScalarBarVisibility(renderView1, True)

      # ----------------------------------------------------------------
      # setup color maps and opacity mapes used in the visualization
      # note: the Get..() functions create a new object, if needed
      # ----------------------------------------------------------------

      # get opacity transfer function/opacity map for 'GlyphScale'
      glyphScalePWF = GetOpacityTransferFunction('GlyphScale')
      glyphScalePWF.Points = [1.0, 0.0, 0.5, 0.0, 68.0, 1.0, 0.5, 0.0]
      glyphScalePWF.ScalarRangeInitialized = 1

      # ----------------------------------------------------------------
      # finally, restore active source
      SetActiveSource(glyph1)
      # ----------------------------------------------------------------
    return Pipeline()

  class CoProcessor(coprocessing.CoProcessor):
    def CreatePipeline(self, datadescription):
      self.Pipeline = _CreatePipeline(self, datadescription)

  coprocessor = CoProcessor()
  # these are the frequencies at which the coprocessor updates.
  freqs = {'atoms': [1, 1]}
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
