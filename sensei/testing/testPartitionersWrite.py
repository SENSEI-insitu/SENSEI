from mpi4py import *
from sensei import VTKDataAdaptor,ConfigurableAnalysis, \
  ADIOS1AnalysisAdaptor,BlockPartitioner,MeshMetadata,Profiler
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

# make a Cartesian block defined by xo,y0 dx,dy nx,ny
# nested intside a Cartesian domain decomposition
# of nbx * nby blocks at position bi, bj
def MakeBlock(t, x0,y0, dx,dy, nx, ny, nbx, nby, bi, bj):
    npx = nx+1
    npy = ny+1
    npz = 2
    xx = x0 + nx*bi*dx
    yy = y0 + ny*bj*dy
    blk = vtk.vtkImageData()
    blk.SetOrigin(xx, yy, -1.0e-2)
    blk.SetSpacing(dx, dy, 2.0e-2)
    blk.SetDimensions(npx, npy, npz)
    dat = vtk.vtkFloatArray()
    dat.SetName('f_xyt')
    dat.SetNumberOfTuples(npx*npy*npz)
    k = 0
    while k < npz:
        j = 0
        while j < npy:
            i = 0
            while i < npx:
                px = xx + dx*i
                py = yy + dy*j
                q = k*npx*npy + j*npx + i
                dat.SetValue(q, np.cos(t+px)*np.cos(t+py))
                i += 1
            j += 1
        k += 1
    blk.GetPointData().AddArray(dat)
    return blk

# makes a multi block dataset witha Cartesian domain decomp
# of nbx * nby. The blocks are defined by xo,y0 dx,dy nx,ny
def MakeMultiBlock(t, x0,y0, dx,dy, nx, ny, nbx, nby):
    nBlocks = nbx*nby
    md = MeshMetadata.New()
    md.NumBlocks = nBlocks
    md.BlockOwner = [-1]*nBlocks
    md.BlockIds = range(0,nBlocks)
    part = BlockPartitioner.New()
    md = part.GetPartition(comm, md)
    mbds = vtk.vtkMultiBlockDataSet()
    mbds.SetNumberOfBlocks(nBlocks)
    bit = mbds.NewIterator()
    bit.SetSkipEmptyNodes(0)
    bit.InitTraversal()
    j = 0
    while j < nby:
        i = 0
        while i < nbx:
            q = j*nbx + i
            if md.BlockOwner[q] == rank:
                blk = MakeBlock(t, x0,y0, dx,dy, nx,ny, nbx,nby, i,j)
                mbds.SetDataSet(bit, blk)
            bit.GoToNextItem()
            i += 1
        j += 1
    return mbds

def run_simulation(xmlConfig, nIts, theta, dom, nx, nblks):
  dx = [(dom[1] - dom[0])/(nx[0]*nblks[0]), (dom[3] - dom[2])/(nx[1]*nblks[1])]
  dt = (theta[1] - theta[0])/(nIts - 1)

  status_message('running nIts=%d with mesh parameters' \
    'nblks=%s nx=%s dom=%s dx=%s theta=%s dt=%s'%(nIts, \
    str(nblks), str(nx), str(dom), str(dx), str(theta), str(dt)))

  # initialize the analysis adaptor
  ca = ConfigurableAnalysis.New()
  if ca.Initialize(xmlConfig):
    error_message('failed to initialize the analysis')
    sys.exit(-1)

  # loop over time steps
  i = 0
  while i < nIts:

    t = theta[0] + dt*i

    # make the mesh
    mbds = MakeMultiBlock(t, dom[0],dom[2], \
      dx[0],dx[1], nx[0],nx[1], nblks[0],nblks[1])

    da = VTKDataAdaptor.New()
    da.SetDataTime(t)
    da.SetDataTimeStep(i)
    da.SetDataObject('mesh', mbds)

    # run the analysis
    ca.Execute(da)

    # free up data
    da.ReleaseData()

    if rank == 0:
      sys.stderr.write('.')

    i += 1

  if rank == 0:
    sys.stderr.write('\n')

  # force free up the adaptor
  ca.Finalize()

  status_message('finished writing %d steps'%(nIts))

  return 0

if __name__ == '__main__':
  Profiler.Initialize()

  # process command line
  if len(sys.argv) != 13:
      if rank == 0:
        error_message('Usage error:\n' \
          'testPartitionersWrite.py [xml config] [n iterations] ' \
          '[n blocks x] [n blocks y] [n cells per block x] [n cellsper block y] ' \
          '[domain x0] [domain x1] [domain y0] [domain y1] [theta 0] [theta 1]')
      sys.exit(-1)

  xmlConfig = sys.argv[1]
  nIts = int(sys.argv[2])
  nblks = [int(sys.argv[3]), int(sys.argv[4])]
  nx = [int(sys.argv[5]), int(sys.argv[6])]
  dom = [float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]), float(sys.argv[10])]
  theta = [float(sys.argv[11]), float(sys.argv[12])]

  # write data
  ierr = run_simulation(xmlConfig, nIts, theta, dom, nx, nblks)
  if ierr:
    error_message('write failed')

  Profiler.Finalize()

  # return the error code
  sys.exit(ierr)
