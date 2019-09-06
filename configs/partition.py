import sys


sys.stderr.write('===========================IMPORT\n')


#import Image
import numpy as np, matplotlib.pyplot as plt
from mpi4py_fft.pencil import Pencil, Subcomm
from mpi4py_fft import PFFT
from vtk.util.numpy_support import *
from vtk import vtkDataObject, vtkCompositeDataSet
import mpi4py_fft as fft
from PIL import Image as image


def pt_centered(c):
  return c == vtkDataObject.POINT

def GetDistribution(direction):
  if direction == 0: return [1,0,0]
  if direction == 1: return [0,1,0]
  if direction == 2: return [0,0,1]

def GetAxes(direction):
  if direction == 0: return [0,1,2]
  if direction == 1: return [2,0,1]
  if direction == 2: return [1,2,0]

def GetGrid(num_procs, direction): #TODO:THIS MIGHT NOT BE CORRECT BUT IT'S A BEST GUESS FOR NOW
  if direction == 0: return [1,num_procs,1]
  if direction == 1: return [num_procs,1,1]
  if direction == 2: return [num_procs,num_procs,1]

def Execute(adaptor):
  sys.stderr.write('===========================EXECUTE\n')

  comm  = MPI.COMM_WORLD
  ranks = comm.Get_size()
  rank  = comm.Get_rank()
  direction = 0
  percent = .3
  array = 'data'
  send_group = np.zeros(ranks, dtype=np.int)
  nonempty = False

  # get the mesh and arrays we need
  # set the partitioner
  itda = AsInTransitDataAdaptor(adaptor)
  if itda is not None:
    flags = MeshMetadataFlags(0x20)
    flags.SetAll()
    part = PencilPartitioner.New()
    part.Initialize(direction)
    itda.SetPartitioner(part)

    mdout = itda.GetMeshMetadata(0, flags)
    mdin = itda.GetSenderMeshMetadata(0)
    if rank == 0:
      sys.stderr.write('============\n')
      sys.stderr.write('mdin = %s\n'%(str(mdin)))
      sys.stderr.write('============\n')
      sys.stderr.write('mdout = %s\n'%(str(mdout)))
      sys.stderr.write('============\n')

    mesh = mdout.MeshName
    global_extent = mdout.Extent
    dobj  = itda.GetMesh(mesh, False)
    if dobj is None:
      sys.stderr.write("dobj is none")
    itda.AddArray(dobj, mesh, mdout.ArrayCentering[0], mdout.ArrayName[0])

    #Iterate over multiblockdataset -- get dimensions for data collection
    it = dobj.NewIterator()
    extent = [1000,-1000,1000,-1000,1000,-1000]
    while not it.IsDoneWithTraversal():
      nonempty = True
      send_group[rank] = 1
      #get the local data block and its props
      blk = it.GetCurrentDataObject()
      blk_ext = blk.GetExtent()
      for i in range(3):
        extent[i*2] = min(extent[i*2],blk_ext[2*i])
        extent[i*2+1] = max(extent[i*2+1], blk_ext[2*i+1])
      it.GoToNextItem()

    dims = [0,0,0]
    if nonempty:
      if(pt_centered(mdout.ArrayCentering[0])):
        for i in range(3):
          dims[i] = extent[2*i+1] - extent[2*i] + 1
      else:
        for i in range(3):
          dims[i] = extent[2*i+1] - extent[2*i]

      data_collector = np.zeros(dims)

      #Iterate over multiblockdata and get the data of each block
      it = dobj.NewIterator()
      while not it.IsDoneWithTraversal():
      # get the local data block and it's data
        blk = it.GetCurrentDataObject()
        atts = blk.GetPointData if pt_centered(mdout.ArrayCentering[0]) \
          else blk.GetCellData()
        data = vtk_to_numpy(atts.GetArray(mdout.ArrayName[0]))
        dim = [i-j for i,j in zip(blk.GetDimensions(), \
        [0]*3 if pt_centered(mdout.ArrayCentering[0]) else [1]*3)]
        blk_data = np.reshape(data, dim)

        blk_ext = blk.GetExtent()
        tmp_ext = [0,0,0,0,0,0]
        for i in range(3):
          if blk_ext[2*i] < 0:
            tmp_ext[2*i+1] = (abs(blk_ext[2*i]) + blk_ext[2*i+1])
            tmp_ext[2*i]   = 0
          else:
            tmp_ext[2*i]   = blk_ext[2*i]
            tmp_ext[2*i+1] = blk_ext[2*i+1]

        if (tmp_ext[1] <= dims[0]) and (tmp_ext[3] <= dims[1]) and (tmp_ext[5] <= dims[2]):
          data_collector[tmp_ext[0]:tmp_ext[1], tmp_ext[2]:tmp_ext[3], tmp_ext[4]:tmp_ext[5]] = blk_data
        if (tmp_ext[1] > dims[0]):
          data_collector[0:(tmp_ext[1]-tmp_ext[0]), tmp_ext[2]:tmp_ext[3], tmp_ext[4]:tmp_ext[5]] = blk_data
        if (tmp_ext[3] > dims[1]):
          data_collector[tmp_ext[0]:tmp_ext[1], 0:(tmp_ext[3]-tmp_ext[2]), tmp_ext[4]:tmp_ext[5]] = blk_data
        if (tmp_ext[5] > dims[2]):
          data_collector[tmp_ext[0]:tmp_ext[1], tmp_ext[2]:tmp_ext[3], 0:(tmp_ext[5]-tmp_ext[4])] = blk_data
        it.GoToNextItem()

      #Add noise
      for x in np.nditer(data_collector, op_flags=['readwrite']):
        x[...] = x + np.random.normal(0,0.5)

    sys.stderr.write('==========================================%d before fft\n'%rank)

    #Build new comm group with only owner ranks
    rec_group = np.zeros(ranks,dtype=np.int)
    comm.Allreduce(send_group, rec_group, op=MPI.SUM)
    comm_group = np.empty(0,dtype=np.int)

    for i in range(rec_group.size):
      if rec_group[i] == 1:
        comm_group = np.append(comm_group, i)
    group = comm.group.Incl(comm_group)
    new_comm = comm.Create_group(group)

    #For eacher owner rank, transform data using mpi4py-fft
    if nonempty:
      g_ext = mdout.Extent
      N = (g_ext[1] - g_ext[0], g_ext[3] - g_ext[2], g_ext[5] - g_ext[4]) 
      g = GetGrid(comm_group.size, direction)
      a = GetAxes(direction)
      fft = PFFT(new_comm, N, grid=g, axes=a)
      #data_hat = np.zeros_like(data_collector)
      data_hat = fft.forward(data_collector)
      data_shape = np.zeros_like(data_collector)
      #Low pass filter -- remove a percentage of the boundary
      (w, h, l) = data_hat.shape
      low_w = int(float(w)*percent)
      low_h = int(float(h)*percent)
      low_l = int(float(l)*percent)
      for i in range(w):
        for j in range(h):
          for k in range(l):
            if (i < low_w or i > (w - low_w)):
              data_hat[i][j][k] = 0
            if (j < low_h or j > (h - low_h)):
              data_hat[i][j][k] = 0
            if (k < low_l or k > (l - low_l)):
              data_hat[i][j][k] = 0

      data_back = fft.backward(data_hat, data_shape)
  return 0

def Finalize():
  sys.stderr.write('===============================FINALIZE\n')
  return 0
