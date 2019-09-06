#include "PencilPartitioner.h"
#include "XMLUtils.h"

#include <pugixml.hpp>
#include <sstream>
#include <stdlib.h>
#include <unistd.h>

using namespace std;

namespace sensei
{

// --------------------------------------------------------------------------
PencilPartitioner::PencilPartitioner(const std::vector<int> &blkOwner,
  const std::vector<int> &blkIds) : BlockOwner(blkOwner), BlockIds(blkIds)
{
}

// --------------------------------------------------------------------------
void PencilPartitioner::SetBlockOwner(const std::vector<int> &blkOwner)
{
  this->BlockOwner = blkOwner;
}

// --------------------------------------------------------------------------
void PencilPartitioner::SetBlockIds(const std::vector<int> &blkIds)
{
  this->BlockIds = blkIds;
}

// --------------------------------------------------------------------------
//Helper functions
// --------------------------------------------------------------------------
void print_extents(std::array<int, 6> block_ext)
{
  std::cerr << "block extents: " << std::endl;
  std::cerr << "i: [ " << block_ext[0] << ", " << block_ext[1] << " ]" << std::endl;
  std::cerr << "j: [ " << block_ext[2] << ", " << block_ext[3] << " ]" << std::endl;
  std::cerr << "k: [ " << block_ext[4] << ", " << block_ext[5] << " ]" << std::endl;
  std::cerr << std::endl;
}

// --------------------------------------------------------------------------
void print_block(std::vector<int> Block)
{
  int size = Block.size();
  std::cerr << "Block: ";
  for(int i = 0; i < size; i++)
  {
    std::cerr << Block[i] << " ";
  }
  std::cerr << endl;
}

// --------------------------------------------------------------------------
bool is_intersection_empty(std::array<int,6> block)
{
  for(int i = 0; i < 3; i++)
  {
    if(block[2*i] >= block[2*i+1])
    {
      return true;
    }
  }
  return false;
}


// --------------------------------------------------------------------------
std::array<int, 6> intersect(std::array<int,6> blockA, std::array<int,6> blockB)
{
  std::array<int, 6> blockOut = {0,0,0,0,0,0};
  for(int i = 0; i < 3; i++)
  {
    //FOR OSCILLATOR EXTENTS ===============
    //blockOut[2*i]     = std::max(blockA[2*i] + 1, blockB[2*i] + 1);
    blockOut[2*i]     = std::max(blockA[2*i], blockB[2*i]);
    blockOut[2*i + 1] = std::min(blockA[2*i + 1], blockB[2*i + 1]);
  }

  return blockOut;
}

// --------------------------------------------------------------------------
int GetIndex(std::vector<int> BlockIds, int block)
{
  int size = BlockIds.size();
  for(int i = 0; i < size; i++)
  {
    if(BlockIds[i] == block) return i;
  }
  return -1;
}

// --------------------------------------------------------------------------

std::array<int,6> grow(std::array<int,6> blockIn, int dir)
{
  std::array<int, 6> blockOut = blockIn;
  blockOut[dir*2] -= 1;
  blockOut[dir*2 + 1] += 1;

  return blockOut;
}


//GetMesh/AddArray breaks if BlockIds aren't in sequential order. Why? idk
// --------------------------------------------------------------------------
int reorder(std::vector<int> &BlockOwners, std::vector<int> &BlockIds, int numBlocks)
{
  if(BlockIds.size() == 0 || BlockOwners.size() == 0)
  {
    SENSEI_ERROR( "Partition Error: BlockOwner or BlockIds unassigned " )
    return -1;
  }
  std::vector<int> bi;
  std::vector<int> bo;
  for(int j = 0; j < numBlocks; j++)
  {
    for(int i = 0; i < numBlocks; i++)
    {
      if(j == BlockIds[i])
      {
        bi.push_back(j);
        bo.push_back(BlockOwners[i]);
        break;
      }
    }
  }
  BlockOwners = bo;
  BlockIds    = bi;
  return 0;
}


// --------------------------------------------------------------------------
int assign_blocks( int dir, std::vector<int> &BlockOwner, std::vector<int> &BlockIds, const MeshMetadataPtr &mdIn)
{
  BlockIds.clear();
  BlockOwner.clear();

  int NumBlocks = mdIn->NumBlocks;

  std::vector<int> assigned_blocks(NumBlocks, 0);
  for(int i = 0; i < NumBlocks; i++)
  {
    std::array<int, 6> current_block_ext = grow(mdIn->BlockExtents[i], dir);
    for(int j = i+1; j < NumBlocks; j++)
    {
      std::array<int, 6> next_block_ext = mdIn->BlockExtents[j];
      std::array<int, 6> block_intersect_ext = intersect(current_block_ext, next_block_ext);
      if(!is_intersection_empty(block_intersect_ext))
      {
        if(!assigned_blocks[i] && !assigned_blocks[j])
        {
          assigned_blocks[i] = 1;
          assigned_blocks[j] = 1;
          BlockOwner.push_back(mdIn->BlockOwner[i]);
          BlockIds.push_back(mdIn->BlockIds[i]);
          BlockOwner.push_back(mdIn->BlockOwner[i]);
          BlockIds.push_back(mdIn->BlockIds[j]);
        }
        else if(!assigned_blocks[i] && assigned_blocks[j])
        {
          assigned_blocks[i] = 1;
          int j_index = GetIndex(BlockIds, mdIn->BlockIds[j]);
          BlockOwner.push_back(BlockOwner[j_index]);
          BlockIds.push_back(mdIn->BlockIds[i]);
        }
        else if(assigned_blocks[i] && !assigned_blocks[j])
        {
          assigned_blocks[j] = 1;
          int i_index = GetIndex(BlockIds, mdIn->BlockIds[i]);
          BlockOwner.push_back(BlockOwner[i_index]);
          BlockIds.push_back(mdIn->BlockIds[j]);
        }
        else
        {
          int i_index = GetIndex(BlockIds, mdIn->BlockIds[i]);
          int j_index = GetIndex(BlockIds,mdIn->BlockIds[j]);
          if(BlockOwner[i_index] != BlockOwner[j_index])
          {
            SENSEI_ERROR( "ASSIGNED BLOCKS DON'T MATCH AND THEY SHOULD " <<  i_index << " != "     <<  j_index)
            return -1;
          }
        }
      }
    }
  }
  return 0;
}




// --------------------------------------------------------------------------
int PencilPartitioner::GetPartition(MPI_Comm comm, const MeshMetadataPtr &mdIn,
  MeshMetadataPtr &mdOut)
{
  (void)comm;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  mdOut = mdIn->NewCopy();

  std::vector<int> assigned_blocks(mdIn->NumBlocks, 0);

  /*
  if(rank == 0)
  {
    std::cerr << "BEFORE GET PARTITION ================" << endl;
    cerr << "block ids: " << endl;
    for(int i = 0; i < (int)mdOut->BlockIds.size(); i++)
      cerr << mdOut->BlockIds[i] << " ";
    cerr << endl;
    cerr << "block owners: "  << endl;
    for(int i = 0; i < (int)mdOut->BlockOwner.size(); i++)
      cerr << mdOut->BlockOwner[i] << " ";
    cerr << endl;
  }
  */
  if(rank == 0)
  {
    cerr << "Alignment Direction: " << this->dir << endl;
  }
  assign_blocks(this->dir, this->BlockOwner, this->BlockIds, mdIn);
  reorder(this->BlockOwner, this->BlockIds, mdIn->NumBlocks);
  mdOut->BlockOwner = this->BlockOwner;
  mdOut->BlockIds = this->BlockIds;

  /*
  if(rank == 0)
  {
    std::cerr << "AFTER GET PARTITION ================" << endl;
    cerr << "block ids: " << endl;
    for(int i = 0; i < (int)this->BlockIds.size(); i++)
      cerr << this->BlockIds[i] << " ";
    cerr << endl;
    cerr << "block owners: "  << endl;
    for(int i = 0; i < (int)this->BlockOwner.size(); i++)
      cerr << this->BlockOwner[i] << " ";
    cerr << endl;
  }
  */
  return 0;
}

// --------------------------------------------------------------------------
int PencilPartitioner::Initialize(pugi::xml_node &node)
{
  //TODO:: GET DIR FROM NODE
  std::ostringstream oss;
  SENSEI_STATUS("Configured PencilPartitioner " << oss.str())
  return 0;
}

// --------------------------------------------------------------------------
int PencilPartitioner::Initialize(int direction)
{
  this->dir = direction;
  SENSEI_STATUS("Configured PencilPartitioner ")
  return 0;
}


}
