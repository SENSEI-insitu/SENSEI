from mpi4py import *
from sensei import VTKDataAdaptor,ConfigurableInTransitDataAdaptor, \
  ConfigurableAnalysis,ADIOS1DataAdaptor,Profiler
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

def display_decomp(da):
  rmd = da.GetMeshMetadata(0)
  smd = da.GetSenderMeshMetadata(0)
  nArrays = rmd.NumArrays
  arrays = rmd.ArrayName
  meshName = rmd.MeshName
  status_message('received mesh %s with %d arrays %s'%(meshName, nArrays, arrays))
  status_message('BlockIds=%s'%(str(rmd.BlockIds)))
  status_message('SenderBlockOwner=%s'%(str(smd.BlockOwner)))
  status_message('ReceiverBlockOwner=%s'%(str(rmd.BlockOwner)))

def run_endpoint(analysisXml, transportXml):
  # initialize the data adaptor
  status_message('initializing the transport layer')

  da = ConfigurableInTransitDataAdaptor.New()
  if da.Initialize(transportXml):
    error_message('failed to configure the transport')
    sys.exit(-1)

  # connect
  da.OpenStream()
  display_decomp(da)

  # create the analysis
  ca = ConfigurableAnalysis.New()
  if ca.Initialize(analysisXml):
    error_message('failed to configure the analysis')
    sys.exit(-1)

  # process all time steps
  status_message('receiving data...')
  status = 0
  n_steps = 0
  while status == 0:

    # execute the analysis
    if not ca.Execute(da):
      error_message('analysis failed')
      sys.exit(-1)

    # free up data
    da.ReleaseData()

    # next step
    status = da.AdvanceStream()
    n_steps += 1

    if rank == 0:
      sys.stderr.write('.')

  # close down the stream
  if rank == 0:
    sys.stderr.write('\n')

  ca.Finalize()
  da.CloseStream()

  status_message('completed after processing %d steps'%(n_steps))
  return 0

if __name__ == '__main__':

  Profiler.Initialize()

  # process command line
  if len(sys.argv) != 3:
    if rank == 0:
        error_message('usage error\ntestPartitionerRead.py [analysis xml] [transport xml]')
    sys.exit(-1)

  analysisXml = sys.argv[1]
  transportXml = sys.argv[2]

  # write data
  ierr = run_endpoint(analysisXml, transportXml)
  if ierr:
    error_message('read failed')

  Profiler.Finalize()

  # return the error code
  sys.exit(ierr)
