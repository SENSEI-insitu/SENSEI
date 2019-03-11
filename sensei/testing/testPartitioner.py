import sys
import random
from sensei import *
from mpi4py import MPI
random.seed(1)

comm = MPI.COMM_WORLD

# block per rank config on the sender side
numSenderRanks = int(sys.argv[1])
numSenderBlocks = numSenderRanks
numRecvrRanks = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    sys.stderr.write('sender ranks = %d blocks = %d\n'%(numSenderRanks, numSenderBlocks))
    sys.stderr.write('receiver ranks = %d blocks = %d\n'%(numRecvrRanks, numSenderBlocks))

# set up the sender metadata
mdIn = MeshMetadata.New()
mdIn.NumBlocks = numSenderBlocks
mdIn.BlockIds = range(0,numSenderBlocks)
mdIn.BlockOwner = range(0,numSenderRanks)

if rank == 0:
    sys.stderr.write('sender MeshMetadata = %s\n'%(str(mdIn)))

# run the partitioners
p = BlockPartitioner()
mdOut = p.GetPartition(comm, mdIn)

if rank == 0:
    sys.stderr.write('== BlockPartitioner ==\n')
    sys.stderr.write('receiver MeshMetadata = %s\n'%(str(mdOut)))

p = CyclicPartitioner()
mdOut = p.GetPartition(comm, mdIn)

if rank == 0:
    sys.stderr.write('== CyclicPartitioner ==\n')
    sys.stderr.write('receiver MeshMetadata = %s\n'%(str(mdOut)))

planeSize = 2
p = PlanarPartitioner(planeSize)
mdOut = p.GetPartition(comm, mdIn)

if rank == 0:
    sys.stderr.write('== PlanarPartitioner(%d) ==\n'%(planeSize))
    sys.stderr.write('receiver MeshMetadata = %s\n'%(str(mdOut)))

p = MappedPartitioner()
bids = []
owner = []
i = 0
while i < numSenderBlocks:
    bids.append(i)
    owner.append(random.randint(1,numRecvrRanks) - 1)
    i += 1
p.SetBlockOwner(owner)
p.SetBlockIds(bids)
mdOut = p.GetPartition(comm, mdIn)

if rank == 0:
    sys.stderr.write('== MappedPartitioner ==\n')
    sys.stderr.write('receiver MeshMetadata = %s\n'%(str(mdOut)))
