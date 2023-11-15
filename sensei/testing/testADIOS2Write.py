from mpi4py import *
from sensei import SVTKDataAdaptor,ADIOS2DataAdaptor,ADIOS2AnalysisAdaptor
import sys,os
import numpy as np
import svtk, svtk.numpy_support as svtknp
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
  va = svtknp.numpy_to_svtk(a, deep=1)
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
  im = svtk.svtkImageData()
  im.SetExtent(i0,i1,j0,j1,k0,k1)
  nx = i1 - i0 + 1
  ny = j1 - j0 + 1
  nz = k1 - k0 + 1
  npts = (nx + 1)*(ny + 1)*(nz + 1)
  ncells = nx*ny*nz
  get_data_arrays(npts, im.GetPointData())
  get_data_arrays(ncells, im.GetCellData())
  return im

def points_to_unstructured(x,y,z):
  nx = len(x)
  # points
  xyz = np.zeros(3*nx, dtype=np.float32)
  xyz[::3] = x[:]
  xyz[1::3] = y[:]
  xyz[2::3] = z[:]
  vxyz = svtknp.numpy_to_svtk(xyz, deep=1)
  vxyz.SetNumberOfComponents(3)
  vxyz.SetNumberOfTuples(nx)
  pts = svtk.svtkPoints()
  pts.SetData(vxyz)
  # cells
  ct = svtknp.numpy_to_svtk(np.full(nx, svtk.SVTK_VERTEX, dtype=np.uint8),
                            deep=1, array_type=svtk.SVTK_UNSIGNED_CHAR)

  cc = np.empty(nx,dtype=np.int64)
  cc[:] = np.arange(0,nx,dtype=np.int64)

  co = np.empty(nx+1,dtype=np.int64)
  co[:] = np.arange(0,nx+1,dtype=np.int64)

  ca = svtk.svtkCellArray()
  ca.SetData(svtknp.numpy_to_svtk(co, deep=1, array_type=svtk.SVTK_ID_TYPE),
             svtknp.numpy_to_svtk(cc, deep=1, array_type=svtk.SVTK_ID_TYPE))
  # package it all up in a poly data set
  ug = svtk.svtkUnstructuredGrid()
  ug.SetPoints(pts)
  ug.SetCells(ct, ca)
  # add some scalar data
  get_data_arrays(nx, ug.GetPointData())
  get_data_arrays(nx, ug.GetCellData())


  return ug

def get_unstructured(nx):
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
  ug = points_to_unstructured(x,y,z)
  get_data_arrays(nx, ug.GetPointData())
  get_data_arrays(nx, ug.GetCellData())
  return ug

def points_to_polydata(x,y,z):
  nx = len(x)
  # points
  xyz = np.zeros(3*nx, dtype=np.float32)
  xyz[::3] = x[:]
  xyz[1::3] = y[:]
  xyz[2::3] = z[:]
  vxyz = svtknp.numpy_to_svtk(xyz, deep=1)
  vxyz.SetNumberOfComponents(3)
  vxyz.SetNumberOfTuples(nx)
  pts = svtk.svtkPoints()
  pts.SetData(vxyz)
  # cells
  cc = np.empty(nx,dtype=np.int64)
  cc[:] = np.arange(0,nx,dtype=np.int64)

  co = np.empty(nx+1,dtype=np.int64)
  co[:] = np.arange(0,nx+1,dtype=np.int64)

  ca = svtk.svtkCellArray()
  ca.SetData(svtknp.numpy_to_svtk(co, deep=1, array_type=svtk.SVTK_ID_TYPE),
             svtknp.numpy_to_svtk(cc, deep=1, array_type=svtk.SVTK_ID_TYPE))
  # package it all up in a poly data set
  pd = svtk.svtkPolyData()
  pd.SetPoints(pts)
  pd.SetVerts(ca)
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

def write_data(engine, file_name, steps_per_file, n_its):
  # initialize the analysis adaptor
  aw = ADIOS2AnalysisAdaptor.New()
  aw.SetEngineName(engine)
  aw.SetFileName(file_name)
  aw.SetStepsPerFile(steps_per_file)
  aw.SetVerbose(1)

  # create the datasets
  # the first mesh is an image
  im = svtk.svtkMultiBlockDataSet()
  im.SetNumberOfBlocks(n_ranks)
  im.SetBlock(rank, get_image(rank,rank,0,16,0,1))
  # the second mesh is polydata
  pd = svtk.svtkMultiBlockDataSet()
  pd.SetNumberOfBlocks(n_ranks)
  pd.SetBlock(rank, get_polydata(16))
  # the third mesh is unstructured
  ug = svtk.svtkMultiBlockDataSet()
  ug.SetNumberOfBlocks(n_ranks)
  ug.SetBlock(rank, get_unstructured(16))
  # associate a name with each mesh
  meshes = {'image':im, 'polydata':pd, 'unstructured':ug}

  # loop over time steps
  i = 0
  while i < n_its:
    t = float(i)
    it = i
    # pass into the data adaptor
    status_message('initializing the SVTKDataAdaptor ' \
      'step %d time %0.1f'%(it,t))

    da = SVTKDataAdaptor.New()
    da.SetDataTime(t)
    da.SetDataTimeStep(it)

    for meshName,mesh in meshes.items():
      da.SetDataObject(meshName, mesh)

    # execute the analysis adaptor
    status_message('executing ADIOS2AnalysisAdaptor %s ' \
      'step %d time %0.1f'%(engine,it,t))

    aw.Execute(da)

    # free up data
    da.ReleaseData()
    da = None
    i += 1

  # force free up the adaptor
  aw.Finalize()
  status_message('finished writing %d steps'%(n_its))
  # set the return value
  return 0

if __name__ == '__main__':
  # process command line
  engine = sys.argv[1]
  file_name = sys.argv[2]
  steps_per_file = int(sys.argv[3])
  n_its = int(sys.argv[4])
  # write data
  ierr = write_data(engine, file_name, steps_per_file, n_its)
  if ierr:
    error_message('write failed')
  # return the error code
  sys.exit(ierr)
