from mpi4py import *
from multiprocessing import Process,Lock,Value
from sensei import VTKDataAdaptor,ADIOS1DataAdaptor, \
  ADIOS1AnalysisAdaptor,BlockPartitioner,CyclicPartitioner
import sys,os
import numpy as np
import vtk, vtk.util.numpy_support as vtknp
from time import sleep

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()

def error_message(msg):
  sys.stderr.write('ERROR[%d] : %s\n'%(rank, msg))

def status_message(msg, io_rank=0):
  if rank == io_rank:
    sys.stderr.write('STATUS[%d] : %s\n'%(rank, msg))

def check_array(array):
  # checks that array[i] == i
  test_array = vtknp.vtk_to_numpy(array)
  n_vals = len(test_array)
  base_array = np.empty(n_vals, dtype=test_array.dtype)
  i = 0
  while i < n_vals:
    base_array[i] = i
    i += 1
  ids = np.where(base_array != test_array)[0]
  if len(ids):
    error_message('wrong values at %s'%(str(ids)))
    return -1
  return 0

def read_data(fileName, method):
  # initialize the data adaptor
  status_message('initializing ADIOS1DataAdaptor file=%s method=%s'%(fileName,method))
  da = ADIOS1DataAdaptor.New()
  da.SetReadMethod(method)
  da.SetFileName(fileName)
  da.SetPartitioner(BlockPartitioner.New())
  da.OpenStream()
  # process all time steps
  n_steps = 0
  retval = 0
  while True:
    # get the time info
    t = da.GetDataTime()
    it = da.GetDataTimeStep()
    status_message('received step %d time %0.1f'%(it, t))

    # get the mesh info
    nMeshes = da.GetNumberOfMeshes()
    status_message('receveied %d meshes'%(nMeshes))
    i = 0
    while i < nMeshes:
      md = da.GetMeshMetadata(i)
      meshName = md.MeshName
      status_message('received mesh %s'%(meshName))

      # report on data partitioning
      smd = da.GetSenderMeshMetadata(i)
      status_message('BlockIds=%s'%(str(md.BlockIds)))
      status_message('SenderBlockOwner=%s'%(str(smd.BlockOwner)))
      status_message('ReceiverBlockOwner=%s'%(str(md.BlockOwner)))

      # get a VTK dataset with all the arrays
      ds = da.GetMesh(meshName, False)

      # request each array
      n_arrays = md.NumArrays
      status_message('%d arrays %s'%(n_arrays, str(md.ArrayName)))
      j = 0
      while j < n_arrays:
        array_name = md.ArrayName[j]
        assoc = md.ArrayCentering[j]
        da.AddArray(ds, meshName, assoc, array_name)
        #status_message('receive array %s'%(array_name))
        j += 1

      # this often will cause segv's if the dataset has been
      # improperly constructed, thus serves as a good check
      str_rep = str(ds)

      # check the arrays have the expected data
      it = ds.NewIterator()
      while not it.IsDoneWithTraversal():

        bds = it.GetCurrentDataObject()
        idx = it.GetCurrentFlatIndex()

        n_arrays = md.NumArrays

        status_message('checking %d data arrays ' \
          'in block %d of mesh "%s" type %s'%(n_arrays, idx-1, md.MeshName, \
          bds.GetClassName()), rank)

        j = 0
        while j < n_arrays:

          array = bds.GetPointData().GetArray(md.ArrayName[j]) \
            if md.ArrayCentering[j] == vtk.vtkDataObject.POINT else \
              bds.GetCellData().GetArray(md.ArrayName[j])

          if not 'BlockOwner' in md.ArrayName[j] and check_array(array):
            error_message('Test failed on array %d "%s"'%(j, array.GetName()))
            retval = -1

          #status_message('checking %s ... OK'%(md.ArrayName[j]))

          j += 1
        it.GoToNextItem()
      i += 1

    n_steps += 1
    if (da.AdvanceStream()):
      break

  # close down the stream
  da.CloseStream()
  status_message('closed stream after receiving %d steps'%(n_steps))
  return retval

if __name__ == '__main__':
  # process command line
  fileName = sys.argv[1]
  method = sys.argv[2]

  # write data
  ierr = read_data(fileName, method)
  if ierr:
    error_message('read failed')

  # return the error code
  sys.exit(ierr)
