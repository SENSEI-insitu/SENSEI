#include "MappedPartitioner.h"

#include <pugixml.hpp>

namespace sensei
{

// --------------------------------------------------------------------------
MappedPartitioner::MappedPartitioner(const std::vector<int> &blkOwner,
  const std::vector<int> &blkIds) : BlockOwner(blkOwner), BlockIds(blkIds)
{
}

// --------------------------------------------------------------------------
void MappedPartitioner::SetBlockOwner(const std::vector<int> &blkOwner)
{
  this->BlockOwner = blkOwner;
}

// --------------------------------------------------------------------------
void MappedPartitioner::SetBlockIds(const std::vector<int> &blkIds)
{
  this->BlockIds = blkIds;
}

// --------------------------------------------------------------------------
int MappedPartitioner::GetPartition(MPI_Comm comm, const MeshMetadataPtr &mdIn,
  MeshMetadataPtr &mdOut)
{
  (void)comm;

  mdOut = mdIn->NewCopy();

  mdOut->BlockOwner = this->BlockOwner;
  mdOut->BlockIds = this->BlockIds;

  return 0;
}

// --------------------------------------------------------------------------
int MappedPartitioner::Initialize(pugi::xml_node &node)
{
  std::string blkOwnerElem = node.child("block_owner").text().as_string();
  std::string blkIdsElem = node.child("block_id").text().as_string();

  // TODO -- error checking? What should happen if the elements are not
  //  found? Is that an error?

  // TODO -- use the methods provided by pugi to get the contents
  // of the elements.

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
