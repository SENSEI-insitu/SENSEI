from paraview.simple import *
from paraview import coprocessing

# work around a bug in the simple module
from paraview import lookuptable
paraview.simple.lookuptable = lookuptable
paraview.simple._lutReader = lookuptable.vtkPVLUTReader()

#--------------------------------------------------------------
# Code generated from cpstate.py to create the CoProcessor.
# ParaView 5.3.0-78-gd6e7170 64 bits

# ----------------------- CoProcessor definition -----------------------

def CreateCoProcessor():
  def _CreatePipeline(coprocessor, datadescription):
    class Pipeline:
      # state file generated using paraview version 5.3.0-78-gd6e7170

      # ----------------------------------------------------------------
      # setup views used in the visualization
      # ----------------------------------------------------------------

      #### disable automatic camera reset on 'Show'
      paraview.simple._DisableFirstRenderCameraReset()

      # Create a new 'Render View'
      renderView1 = CreateView('RenderView')
      renderView1.ViewSize = [795, 820]
      renderView1.InteractionMode = '2D'
      renderView1.AxesGrid = 'GridAxes3DActor'
      renderView1.OrientationAxesVisibility = 0
      renderView1.StereoType = 0
      renderView1.CameraPosition = [0.0, 0.0, 33752832251292.418*0.7]
      renderView1.CameraParallelScale = 7123522406853.433
      renderView1.Background = [0.3176470588235294, 0.3411764705882353, 0.4313725490196079]

      # register the view with coprocessor
      # and provide it with information such as the filename to use,
      # how frequently to write the images, etc.
      coprocessor.RegisterView(renderView1,
          filename='image_%t.png', freq=1, fittoscreen=0, magnification=1, width=795, height=820, cinema={})
      renderView1.ViewTime = datadescription.GetTime()

      # ----------------------------------------------------------------
      # setup the data processing pipelines
      # ----------------------------------------------------------------

      # create a new 'Plane'
      plane1 = Plane()
      plane1.Origin = [-6000000000000.0, -6000000000000.0, 0.0]
      plane1.Point1 = [6000000000000.0, -6000000000000.0, 0.0]
      plane1.Point2 = [-6000000000000.0, 6000000000000.0, 0.0]

      # create a new 'PVD Reader'
      # create a producer from a simulation input
      ptdatamasterpvd = coprocessor.CreateProducer(datadescription, 'input')

      # create a new 'Threshold'
      threshold1 = Threshold(Input=ptdatamasterpvd)
      threshold1.Scalars = ['POINTS', 'm']
      threshold1.ThresholdRange = [0.0, 1e+29]

      # create a new 'Threshold'
      threshold2 = Threshold(Input=ptdatamasterpvd)
      threshold2.Scalars = ['POINTS', 'm']
      threshold2.ThresholdRange = [1e+29, 1.989e+30]

      # create a new 'Glyph'
      glyph2 = Glyph(Input=threshold2,
          GlyphType='2D Glyph')
      glyph2.Scalars = ['POINTS', 'None']
      glyph2.Vectors = ['POINTS', 'None']
      glyph2.ScaleMode = 'scalar'
      glyph2.ScaleFactor = 1000000000000.0
      glyph2.GlyphTransform = 'Transform2'

      # init the '2D Glyph' selected for 'GlyphType'
      glyph2.GlyphType.GlyphType = 'Cross'

      # create a new 'Glyph'
      glyph1 = Glyph(Input=threshold1,
          GlyphType='Sphere')
      glyph1.Scalars = ['POINTS', 'm']
      glyph1.Vectors = ['POINTS', 'None']
      glyph1.ScaleMode = 'scalar'
      glyph1.ScaleFactor = 2.5e-15
      glyph1.GlyphMode = 'All Points'
      glyph1.GlyphTransform = 'Transform2'

      # init the 'Sphere' selected for 'GlyphType'
      glyph1.GlyphType.ThetaResolution = 32
      glyph1.GlyphType.PhiResolution = 32

      # ----------------------------------------------------------------
      # setup color maps and opacity mapes used in the visualization
      # note: the Get..() functions create a new object, if needed
      # ----------------------------------------------------------------

      # get opacity transfer function/opacity map for 'ids'
      idsPWF = GetOpacityTransferFunction('ids')
      idsPWF.Points = [0.0, 0.0, 0.5, 0.0, 59.0, 1.0, 0.5, 0.0]
      idsPWF.ScalarRangeInitialized = 1

      # ----------------------------------------------------------------
      # setup the visualization in view 'renderView1'
      # ----------------------------------------------------------------

      # show data from glyph1
      idsArray = glyph1.PointData.GetArray('ids')

      glyph1Display = Show(glyph1, renderView1)
      # trace defaults for the display properties.
      glyph1Display.Representation = 'Surface'
      glyph1Display.ColorArrayName = ['POINTS', 'ids']
      glyph1Display.LookupTable = AssignLookupTable(idsArray,'jet')
      glyph1Display.OSPRayScaleArray = 'Normals'
      glyph1Display.OSPRayScaleFunction = 'PiecewiseFunction'
      glyph1Display.SelectOrientationVectors = 'None'
      glyph1Display.ScaleFactor = -2.0000000000000002e+298
      glyph1Display.SelectScaleArray = 'None'
      glyph1Display.GlyphType = 'Arrow'
      glyph1Display.PolarAxes = 'PolarAxesRepresentation'
      glyph1Display.GaussianRadius = -1.0000000000000001e+298
      glyph1Display.SetScaleArray = [None, 'f']
      glyph1Display.ScaleTransferFunction = 'PiecewiseFunction'
      glyph1Display.OpacityArray = [None, 'f']
      glyph1Display.OpacityTransferFunction = 'PiecewiseFunction'

      # show data from glyph2
      glyph2Display = Show(glyph2, renderView1)
      # trace defaults for the display properties.
      glyph2Display.Representation = 'Surface'
      glyph2Display.ColorArrayName = ['POINTS', '']
      glyph2Display.LineWidth = 3.0
      glyph2Display.OSPRayScaleArray = 'f'
      glyph2Display.OSPRayScaleFunction = 'PiecewiseFunction'
      glyph2Display.SelectOrientationVectors = 'None'
      glyph2Display.ScaleFactor = 0.010000000149011612
      glyph2Display.SelectScaleArray = 'None'
      glyph2Display.GlyphType = 'Arrow'
      glyph2Display.PolarAxes = 'PolarAxesRepresentation'
      glyph2Display.GaussianRadius = 0.005000000074505806
      glyph2Display.SetScaleArray = ['POINTS', 'f']
      glyph2Display.ScaleTransferFunction = 'PiecewiseFunction'
      glyph2Display.OpacityArray = ['POINTS', 'f']
      glyph2Display.OpacityTransferFunction = 'PiecewiseFunction'

      # show data from plane1
      plane1Display = Show(plane1, renderView1)
      # trace defaults for the display properties.
      plane1Display.Representation = 'Outline'
      plane1Display.ColorArrayName = [None, '']
      plane1Display.LineWidth = 2.0
      plane1Display.OSPRayScaleArray = 'Normals'
      plane1Display.OSPRayScaleFunction = 'PiecewiseFunction'
      plane1Display.SelectOrientationVectors = 'None'
      plane1Display.ScaleFactor = 1200000021299.2
      plane1Display.SelectScaleArray = 'None'
      plane1Display.GlyphType = 'Arrow'
      plane1Display.PolarAxes = 'PolarAxesRepresentation'
      plane1Display.GaussianRadius = 600000010649.6
      plane1Display.SetScaleArray = [None, '']
      plane1Display.ScaleTransferFunction = 'PiecewiseFunction'
      plane1Display.OpacityArray = [None, '']
      plane1Display.OpacityTransferFunction = 'PiecewiseFunction'

      # ----------------------------------------------------------------
      # finally, restore active source
      SetActiveSource(plane1)
      # ----------------------------------------------------------------
    return Pipeline()

  class CoProcessor(coprocessing.CoProcessor):
    def CreatePipeline(self, datadescription):
      self.Pipeline = _CreatePipeline(self, datadescription)

  coprocessor = CoProcessor()
  # these are the frequencies at which the coprocessor updates.
  freqs = {'input': [1, 1, 1, 1, 1]}
  coprocessor.SetUpdateFrequencies(freqs)
  return coprocessor

#--------------------------------------------------------------
# Global variables that will hold the pipeline for each timestep
# Creating the CoProcessor object, doesn't actually create the ParaView pipeline.
# It will be automatically setup when coprocessor.UpdateProducers() is called the
# first time.
coprocessor = CreateCoProcessor()

#--------------------------------------------------------------
# Enable Live-Visualizaton with ParaView
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
    coprocessor.WriteImages(datadescription, rescale_lookuptable=False)

    # Live Visualization, if enabled.
    coprocessor.DoLiveVisualization(datadescription, "localhost", 22222)
