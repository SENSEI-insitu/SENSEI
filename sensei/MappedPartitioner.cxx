
#include "MappedPartitioner.h"


namespace sensei
{

MappedPartitioner::MappedPartitioner(const std::vector<int>& blkOwner, 
  const std::vector<int>& blkIds) : BlockOwner(blkOwner), BlockIds(blkIds)
{

}

int MappedPartitioner::GetPartition(MPI_Comm comm, const MeshMetadataPtr &mdIn,
  MeshMetadataPtr &mdOut)
{
  mdOut = mdIn->NewCopy();

  mdOut->BlockOwner = this->BlockOwner;
  mdOut->BlockIds = this->BlockIds;

  return 0;
}


int MappedPartitioner::Initialize(pugi::xml_node &node)
{
  std::string blkOwnerElem = node.child("block_owner").text().as_string();
  std::string blkIdsElem = node.child("block_id").text().as_string();

  std::string delims = " \t";

  std::size_t curr = blkOwnerElem.find_first_of(delims, 0);
  std::size_t next = blkOwnerElem.find_first_of(delims, curr + 1);

  while (curr != std::string::npos)
    {
    this->BlockOwner.push_back(std::stoi(blkOwnerElem.substr(curr, next - curr)));
    curr = next;
    next = blkOwnerElem.find_first_of(delims, curr + 1);    
    }

  curr = blkIdsElem.find_first_of(delims, 0);
  next = blkIdsElem.find_first_of(delims, curr + 1);

  while (curr != std::string::npos)
    {
    this->BlockIds.push_back(std::stoi(blkIdsElem.substr(curr, next - curr)));
    curr = next;
    next = blkIdsElem.find_first_of(delims, curr + 1);    
    }

  return 0;
}

}
