
from paraview.simple import *
from paraview import coprocessing


#--------------------------------------------------------------
# Code generated from cpstate.py to create the CoProcessor.
# ParaView 5.4.1 64 bits

#--------------------------------------------------------------
# Global screenshot output options
imageFileNamePadding=2
rescale_lookuptable=False


# ----------------------- CoProcessor definition -----------------------

def CreateCoProcessor():
  def _CreatePipeline(coprocessor, datadescription):
    class Pipeline:
      # state file generated using paraview version 5.4.1

      # ----------------------------------------------------------------
      # setup views used in the visualization
      # ----------------------------------------------------------------

      #### disable automatic camera reset on 'Show'
      paraview.simple._DisableFirstRenderCameraReset()

      # Create a new 'Render View'
      renderView1 = CreateView('RenderView')
      renderView1.ViewSize = [480, 480]
      renderView1.AxesGrid = 'GridAxes3DActor'
      renderView1.CenterOfRotation = [32.0, 32.0, 32.0]
      renderView1.StereoType = 0
      renderView1.CameraPosition = [32.0, 32.0, 198.73086058648875]
      renderView1.CameraFocalPoint = [32.0, 32.0, -15.4173131703901]
      renderView1.CameraParallelScale = 55.42562584220407
      renderView1.Background = [0.0, 0.0, 0.0]

      # register the view with coprocessor
      # and provide it with information such as the filename to use,
      # how frequently to write the images, etc.
      coprocessor.RegisterView(renderView1,
          filename='image_%t.png', freq=1, fittoscreen=0, magnification=1, width=480, height=480, cinema={})
      renderView1.ViewTime = datadescription.GetTime()

      # ----------------------------------------------------------------
      # setup the data processing pipelines
      # ----------------------------------------------------------------

      # create a new 'PVD Reader'
      # create a producer from a simulation input
      meshpvd = coprocessor.CreateProducer(datadescription, 'mesh')

      # create a new 'Cell Data to Point Data'
      cellDatatoPointData1 = CellDatatoPointData(Input=meshpvd)

      # create a new 'Contour'
      contour1 = Contour(Input=cellDatatoPointData1)
      contour1.ContourBy = ['POINTS', 'data']
      contour1.Isosurfaces = [-0.619028, -0.0739720000000001, 0.47108399999999995, 1.01614]
      contour1.PointMergeMethod = 'Uniform Binning'

      # ----------------------------------------------------------------
      # setup color maps and opacity mapes used in the visualization
      # note: the Get..() functions create a new object, if needed
      # ----------------------------------------------------------------

      # get color transfer function/color map for 'vtkBlockColors'
      vtkBlockColorsLUT = GetColorTransferFunction('vtkBlockColors')
      vtkBlockColorsLUT.InterpretValuesAsCategories = 1
      vtkBlockColorsLUT.Annotations = ['0', '0', '1', '1', '2', '2', '3', '3', '4', '4', '5', '5', '6', '6', '7', '7', '8', '8', '9', '9', '10', '10', '11', '11']
      vtkBlockColorsLUT.ActiveAnnotatedValues = ['0', '1', '2', '3']
      vtkBlockColorsLUT.IndexedColors = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.63, 0.63, 1.0, 0.67, 0.5, 0.33, 1.0, 0.5, 0.75, 0.53, 0.35, 0.7, 1.0, 0.75, 0.5]

      # get opacity transfer function/opacity map for 'vtkBlockColors'
      vtkBlockColorsPWF = GetOpacityTransferFunction('vtkBlockColors')

      # ----------------------------------------------------------------
      # setup the visualization in view 'renderView1'
      # ----------------------------------------------------------------

      # show data from cellDatatoPointData1
      cellDatatoPointData1Display = Show(cellDatatoPointData1, renderView1)
      # trace defaults for the display properties.
      cellDatatoPointData1Display.Representation = 'Outline'
      cellDatatoPointData1Display.ColorArrayName = ['FIELD', 'vtkBlockColors']
      cellDatatoPointData1Display.LookupTable = vtkBlockColorsLUT
      cellDatatoPointData1Display.OSPRayScaleArray = 'data'
      cellDatatoPointData1Display.OSPRayScaleFunction = 'PiecewiseFunction'
      cellDatatoPointData1Display.SelectOrientationVectors = 'None'
      cellDatatoPointData1Display.ScaleFactor = 6.4
      cellDatatoPointData1Display.SelectScaleArray = 'None'
      cellDatatoPointData1Display.GlyphType = 'Arrow'
      cellDatatoPointData1Display.GlyphTableIndexArray = 'None'
      cellDatatoPointData1Display.DataAxesGrid = 'GridAxesRepresentation'
      cellDatatoPointData1Display.PolarAxes = 'PolarAxesRepresentation'
      cellDatatoPointData1Display.GaussianRadius = 3.2
      cellDatatoPointData1Display.SetScaleArray = ['POINTS', 'data']
      cellDatatoPointData1Display.ScaleTransferFunction = 'PiecewiseFunction'
      cellDatatoPointData1Display.OpacityArray = ['POINTS', 'data']
      cellDatatoPointData1Display.OpacityTransferFunction = 'PiecewiseFunction'

      # show data from contour1
      contour1Display = Show(contour1, renderView1)
      # trace defaults for the display properties.
      contour1Display.Representation = 'Surface'
      contour1Display.ColorArrayName = ['FIELD', 'vtkBlockColors']
      contour1Display.LookupTable = vtkBlockColorsLUT
      contour1Display.OSPRayScaleArray = 'Normals'
      contour1Display.OSPRayScaleFunction = 'PiecewiseFunction'
      contour1Display.SelectOrientationVectors = 'None'
      contour1Display.ScaleFactor = 6.4
      contour1Display.SelectScaleArray = 'None'
      contour1Display.GlyphType = 'Arrow'
      contour1Display.GlyphTableIndexArray = 'None'
      contour1Display.DataAxesGrid = 'GridAxesRepresentation'
      contour1Display.PolarAxes = 'PolarAxesRepresentation'
      contour1Display.GaussianRadius = 3.2
      contour1Display.SetScaleArray = [None, '']
      contour1Display.ScaleTransferFunction = 'PiecewiseFunction'
      contour1Display.OpacityArray = [None, '']
      contour1Display.OpacityTransferFunction = 'PiecewiseFunction'

      # ----------------------------------------------------------------
      # finally, restore active source
      SetActiveSource(contour1)
      # ----------------------------------------------------------------
    return Pipeline()

  class CoProcessor(coprocessing.CoProcessor):
    def CreatePipeline(self, datadescription):
      self.Pipeline = _CreatePipeline(self, datadescription)

  coprocessor = CoProcessor()
  # these are the frequencies at which the coprocessor updates.
  freqs = {'mesh': [1, 1, 1]}
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
