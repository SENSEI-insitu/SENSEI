"""
PYTHON ANAYLSIS ENDPOINT

Simple endpoint for python based computations in the Sensei-FFT pipeline.
This script will be invoked by sensei pythonAnalysis adaptor and display different
visualizations for data computed by FFTW endpoint.

# Project: 
Sensei - FFTW - SFSU

# Contributions:
S. Kulkarni, E. Wes Bethel, B. Loring
2023
"""

import sys
import os
sys.stderr.write(f"Python version: {sys.version} \n Python Executable: {sys.executable} / {os.__file__}\n")
sys.stderr.flush()

import svtk.numpy_support as svtknp
from svtk import svtkDataObject
from matplotlib import pyplot as plt
import numpy
import sensei as sensei

# Defaults:
mesh = "mesh"
array = "data"
out_file = 'fft_python_output.png'

def Initialize():
  """ Initialization code """
  
  return

def Execute(dataAdaptor):
  """ Use sensei::DataAdaptor instance passed in
      dataAdaptor to access and process simulation data """
  
  sys.stderr.write("\n:: FFT PYTHON ENDPOINT ::\n")

  my_rank = comm.Get_rank()

  # Get mesh from simulation data which will be a svtkMultiBlockDataSet object and array to it from simulation data
  dobj_mb = dataAdaptor.GetMesh(mesh, False)
  dataAdaptor.AddArray(dobj_mb, mesh, svtkDataObject.POINT, array)

  # Get the current block (svtkImageData) and extract point data
  dobj = dobj_mb.GetBlock(my_rank) 
  point_data = dobj.GetPointData()
  
  # Get the array and comvert it to a numpy array
  data = point_data.GetArray(array)
  data_np: numpy.ndarray = svtknp.svtk_to_numpy(data).flatten()

  # Get dimensions from our data object via meshMetaData
  mmdFlags = sensei.MeshMetadataFlags()
  mmdFlags.SetBlockDecomp()
  mmdFlags.SetBlockBounds()
  mmdFlags.SetBlockExtents()


  mmd = sensei.MeshMetadata.New(mmdFlags)
  mmd = dataAdaptor.GetMeshMetadata(0, mmdFlags)

  # DEBUG
  # sys.stderr.write(f"MESHMETADATA PY: : {mmd.Validate(comm, mmdFlags)}\n{mmd}")
  # sys.stderr.flush()

  dimensions = mmd.BlockBounds[0]
  dimensions = [int(y)-int(x)+1 for x,y in zip(dimensions[0::2], dimensions[1::2]) if x != 0 or y != 0]
  
  data_np = data_np.reshape(*dimensions)

  # DEBUG:
  # sys.stderr.write(f"-> FFT_Python :: current state - rank {my_rank}:\n {data_np}\n{type(data_np)}\t{data_np.size}, {data_np.shape}\n")
  
  # display the content
  plt.figure()
  plt.imshow(data_np, cmap='OrRd', origin='lower', alpha=1, aspect='auto')
  plt.colorbar()
  plt.title("FFTW PYTHON OUTPUT")
  plt.savefig(out_file)

  return

def Finalize():
  """ Finalization code """
  # YOUR CODE HERE
  return

