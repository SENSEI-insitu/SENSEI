#
# direct test of the AscentAnalysisAdaptor
# basic idea: build some vtk data objects, and then directly invoke
# the Initialize(), Execute(), and Finalize() AscentAnalysisAdaptor methods.
#
# expects 2 command line arguments, in this order
# 1.  name of the sensei xml file that contains the ascent config info
# 2.  an integer number of time steps to iterate over
#

from mpi4py import *
#from sensei import VTKDataAdaptor,ADIOS2DataAdaptor,ADIOS2AnalysisAdaptor
from sensei import VTKDataAdaptor,ConfigurableAnalysis
import sys,os
import numpy as np
import vtk, vtk.util.numpy_support as vtknp
from time import sleep
from random import seed,random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()

def error_message(msg):
  sys.stderr.write('ERROR[%d] : %s\n'%(rank, msg))

def status_message(msg, io_rank=0):
  if rank == io_rank:
    sys.stderr.write('STATUS[%d] : %s\n'%(rank, msg))

def get_data_array(name, size, dtype):
  a = np.empty(size, dtype=dtype)
  i = 0
  while i < size:
    a[i] = i
    i += 1
  va = vtknp.numpy_to_vtk(a, deep=1)
  va.SetName(name)
  return va

def get_data_arrays(size, dsa):
  dsa.AddArray(get_data_array('char_array', size, np.int8))
  dsa.AddArray(get_data_array('int_array', size, np.int32))
  dsa.AddArray(get_data_array('long_array', size, np.int64))
  dsa.AddArray(get_data_array('unsigned_char_array', size, np.uint8))
  dsa.AddArray(get_data_array('unsigned_int_array', size, np.uint32))
  dsa.AddArray(get_data_array('unsigned_long_array', size, np.uint64))
  dsa.AddArray(get_data_array('float_array', size, np.float32))
  dsa.AddArray(get_data_array('double_array', size, np.float64))
  return dsa

def get_image(i0,i1,j0,j1,k0,k1):
  im = vtk.vtkImageData()
  im.SetExtent(i0,i1,j0,j1,k0,k1)
  nx = i1 - i0 + 1
  ny = j1 - j0 + 1
  nz = k1 - k0 + 1
  print(" nx/ny/nz = {}, {}, {}".format(nx,ny,nz))
  #nx = i1 - i0 
  #ny = j1 - j0
  #nz = k1 - k0
  npts = (nx + 1)*(ny + 1)*(nz + 1)
  # wes npts = nx*ny*nz
  ncells = nx*ny*nz
  print(" npts={}, ncells={}".format(npts, ncells))

  get_data_arrays(npts, im.GetPointData())
  get_data_arrays(ncells, im.GetCellData())
  return im

def points_to_polydata(x,y,z):
  nx = len(x)
  # points
  xyz = np.zeros(3*nx, dtype=np.float32)
  xyz[::3] = x[:]
  xyz[1::3] = y[:]
  xyz[2::3] = z[:]
  vxyz = vtknp.numpy_to_vtk(xyz, deep=1)
  vxyz.SetNumberOfComponents(3)
  vxyz.SetNumberOfTuples(nx)
  pts = vtk.vtkPoints()
  pts.SetData(vxyz)
  # cells
  cids = np.empty(2*nx, dtype=np.int32)
  cids[::2] = 1
  cids[1::2] = np.arange(0,nx,dtype=np.int32)
  cells = vtk.vtkCellArray()
  cells.SetCells(nx, vtknp.numpy_to_vtk(cids, \
      deep=1, array_type=vtk.VTK_ID_TYPE))
  # package it all up in a poly data set
  pd = vtk.vtkPolyData()
  pd.SetPoints(pts)
  pd.SetVerts(cells)
  # add some scalar data
  get_data_arrays(nx, pd.GetPointData())
  get_data_arrays(nx, pd.GetCellData())
  return pd

def get_polydata(nx):
  seed(2)
  x = np.empty(nx)
  y = np.empty(nx)
  z = np.empty(nx)
  i = 0
  while i < nx:
    x[i] = random()
    y[i] = random()
    z[i] = random()
    i += 1
  pd = points_to_polydata(x,y,z)
  get_data_arrays(nx, pd.GetPointData())
  get_data_arrays(nx, pd.GetCellData())
  return pd

#def write_data(file_name, engine, n_its):
def write_data(config_filename, n_its):
  # initialize the analysis adaptor
#  aw = ADIOS2AnalysisAdaptor.New()
  aw = ConfigurableAnalysis.New()

  aw.Initialize(config_filename)

  #aw.SetFileName(file_name)
  #aw.SetEngineName(engine)
  print("Setting number of blocks to be n_ranks, which is {}".format(n_ranks))

  # create the datasets
  # the first mesh is an image
  im = vtk.vtkMultiBlockDataSet()
  im.SetNumberOfBlocks(n_ranks)
  #im.SetBlock(rank, get_image(rank,rank,0,16,0,1))
  im.SetBlock(rank, get_image(0,16,0,8,rank,rank))
  # the second mesh is unstructured
  ug = vtk.vtkMultiBlockDataSet()
  ug.SetNumberOfBlocks(n_ranks)
  ug.SetBlock(rank, get_polydata(16))
  # associate a name with each mesh
  meshes = {'image':im, 'unstructured':ug}

  # loop over time steps
  i = 0
  while i < n_its:
    t = float(i)
    it = i
    # pass into the data adaptor
    status_message('initializing the VTKDataAdaptor ' \
      'step %d time %0.1f'%(it,t))

    da = VTKDataAdaptor.New()
    da.SetDataTime(t)
    da.SetDataTimeStep(it)

    status_message('there are %d total meshes here'%(len(meshes)))

    for meshName,mesh in meshes.items():
      status_message('adding mesh named %s'%(meshName))
      da.SetDataObject(meshName, mesh)

    # execute the analysis adaptor
    status_message('executing AscentAnalysisAdaptor ' \
      'step %d time %0.1f'%(it,t))

    nMeshes = -2
    da.GetNumberOfMeshes()
    status_message('after adding meshes, there are %d total meshes in the data object'%(nMeshes))

    aw.Execute(da)

    # free up data
    da.ReleaseData()
    da = None
    i += 1

  # force free up the adaptor
  aw.Finalize()
  status_message('finished processing %d steps'%(n_its))
  # set the return value
  return 0

if __name__ == '__main__':
  # process command line
  config_file_name = sys.argv[1] # the xml for ascent
  #engine = sys.argv[2]
  n_its = int(sys.argv[2])
  # write data
  #ierr = write_data(file_name, engine, n_its)
  ierr = write_data(config_file_name, n_its)
  if ierr:
    error_message('write failed')
  # return the error code
  sys.exit(ierr)
