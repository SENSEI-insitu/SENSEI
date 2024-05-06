"""
PYTHON VISUALIZER

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
import copy

sys.stderr.write(f"\n--> Python version: {sys.version} \n--> Python Executable: {sys.executable} / {os.__file__}\n")
sys.stderr.flush()

import svtk.numpy_support as svtknp
from svtk import svtkDataObject
import matplotlib
import matplotlib.pyplot as plt
import numpy
import sensei as sensei

# Defaults:
mesh = "mesh"
array = "data"
out_file = 'fft_python_output'

def Initialize():
    """ Initialization code """
    
    return

def Execute(dataAdaptor):
    """ Use sensei::DataAdaptor instance passed in
        dataAdaptor to access and process simulation data """
    
    sys.stderr.write("\n:: FFT PYTHON VISUALIZER ::\n")

    # MPI data
    my_rank = comm.Get_rank()
    nrank = comm.Get_size()

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

    # Get dimensions from MeshMetaData
    dimensions = mmd.BlockBounds[0]
    dimensions = [int(y)-int(x)+1 for x,y in zip(dimensions[0::2], dimensions[1::2]) if x != 0 or y != 0]

    # Reshape into 2D array
    data_np = data_np.reshape(*dimensions)
    
    # display the content
    plt.figure()
    plt.imshow(data_np, cmap=matplotlib.colormaps['viridis'], origin='lower', alpha=1, aspect='auto')
    plt.colorbar()
    plt.suptitle(t="FFT OUTPUT (spectral domain)", fontsize=14, x=0.43)
    plt.title(label=f"Size: {dimensions}, Mesh: '{mesh}', Array: '{array}', Rank: {my_rank+1}/{nrank}", fontsize=10)
    plt.savefig(f"{out_file}_{mesh}_{array}_{my_rank}.png")

    return

def Finalize():
    """ Finalization code """
    # YOUR CODE HERE
    return
