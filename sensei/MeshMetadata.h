#ifndef MeshMetadata_h
#define MeshMetadata_h

#include "BinaryStream.h"

#include <mpi.h>
#include <ostream>
#include <vector>
#include <vector>
#include <string>
#include <memory>

namespace sensei
{
/** a set of flags describing which optional fields in the MeshMetadata
 * structure should be generated.
 */
class SENSEI_EXPORT MeshMetadataFlags
{
public:
  MeshMetadataFlags() : Flags(0) {}
  MeshMetadataFlags(long long flags) : Flags(flags) {}

  void SetAll(){ Flags = 0xffffffffffffffff; }
  void ClearAll(){ Flags = 0; }

  // The following API is used to enable optional metadata. This
  // metadata is crucial for some appilications but can be expensive
  // to generate, thus it is only provided when requested.

  /** set, clear, or check flag to generate arrays describing
   * the domain decomposition. This incudles MPI rank block ownership
   * arrays (MeshMetadata.BlockOwner and MeshMetadata.BlockIds)
   */
  void SetBlockDecomp(){ Flags |= DECOMP; }
  /// @copydoc SetBlockDecomp
  void ClearBlockDecomp(){ Flags &= ~DECOMP; }
  /// @copydoc SetBlockDecomp
  bool BlockDecompSet() const { return Flags & DECOMP; }

  /** set, clear, or check flag to generate global and block size arrays
   * MeshMetadata.BlockNumPoints, MeshMetadata.BlockNumCells, and
   *  MeshMetadata.BlockCellArraySize)
   */
  void SetBlockSize(){ Flags |= SIZE; }
  /// @copydoc SetBlockSize
  void ClearBlockSize(){ Flags &= ~SIZE; }
  /// @copydoc SetBlockSize
  bool BlockSizeSet() const { return Flags & SIZE; }

  /** set, clear, or check flag to generate block extent arrays
   * MeshMetadata.BlockExtents
   */
  void SetBlockExtents(){ Flags |= EXTENTS; }
  /// @copydoc SetBlockExtents
  void ClearBlockExtents(){ Flags &= ~EXTENTS; }
  /// @copydoc SetBlockExtents
  bool BlockExtentsSet() const { return Flags & EXTENTS; }

  /** set, clear, or check flag to generate block bounds arrays
   * MeshMetaData.BlockBounds
   */
  void SetBlockBounds(){ Flags |= BOUNDS; }
  /// @copydoc SetBlockBounds
  void ClearBlockBounds(){ Flags &= ~BOUNDS; }
  /// @copydoc SetBlockBounds
  bool BlockBoundsSet() const { return Flags & BOUNDS; }

  /** set, clear, or check flag to generate block array ranges
   * (MeshMetadata.BlockArrayRange)
   */
  void SetBlockArrayRange(){ Flags |= RANGE; }
  /// @copydoc SetBlockArrayRange
  void ClearBlockArrayRange(){ Flags &= ~RANGE; }
  /// @copydoc SetBlockArrayRange
  bool BlockArrayRangeSet() const { return Flags & RANGE; }


  /// serialize for communication and/or I/O
  int ToStream(sensei::BinaryStream &str) const;

  /// deserialize for communication and/or I/O
  int FromStream(sensei::BinaryStream &str);

  /// serialize for human readable I/O
  int ToStream(ostream &str) const;

private:

  /** flags indicate which optional fields are needed some feilds are optional
   * because they are costly to generate and not universally used on the
   * analysis side. see Set/Clear methods above.
   */
  long long Flags;

 /// flag values
 enum { DECOMP = 0x1, SIZE = 0x2, EXTENTS = 0x4,
   BOUNDS = 0x8, RANGE = 0x10 };
};



struct MeshMetadata;
using MeshMetadataPtr = std::shared_ptr<sensei::MeshMetadata>;

/// A container for capturing metadata describing a mesh.
struct MeshMetadata
{
  /// allocate a new instance
  static
  sensei::MeshMetadataPtr New() { return MeshMetadataPtr(new MeshMetadata); }

  /// allocate a new instance intialize the flags
  static
  sensei::MeshMetadataPtr New(const MeshMetadataFlags flags)
  {
    MeshMetadataPtr mdp = MeshMetadataPtr(new MeshMetadata);
    mdp->Flags = flags;
    return mdp;
  }

  /// copy the meatdata in a new instance
  sensei::MeshMetadataPtr NewCopy()
  {
      MeshMetadataPtr md = MeshMetadata::New();
      *md = *this;
      return md;
  }

  /// serialize/deserialize for communication and/or I/O
  int ToStream(sensei::BinaryStream &str) const;

  /// serialize/deserialize for communication and/or I/O
  int FromStream(sensei::BinaryStream &str);

  /// serialize/deserialize for communication and/or I/O
  int ToStream(ostream &str) const;

  /** return true if the Flags match the arrays. will return false
   * if the a flag is set and a coresponding array is empty. an
   * error message will be printed naming the missing the array
   * this method tries to be context sensative. for instance if
   * the extents flag was set but the simulation reports unstructured
   * data no error will be issued. Optionally a set of required flags
   * may be specified. Only errors in fields with corresponding bits set are
   * reported. This lets you have optional fields and handle the error on
   * your own. Finally some convenience functionality is packaged here
   * for instance global extents and bounds are automatically generated
   * if requested but not provided.
   */
  int Validate(MPI_Comm comm,
    const sensei::MeshMetadataFlags &requiredFlags = 0xffffffffffffffff);

  /** construct a global view of the metadata. return 0 if successful.
   * this call uses MPI collectives
   */
  int GlobalizeView(MPI_Comm);

  /** removes all block level information from the instance. initialize
   * the related dataset level information.
   */
  int ClearBlockInfo();

  /// appends block level information of block bid from other.
  int CopyBlockInfo(const sensei::MeshMetadataPtr &other, int bid);

  /// removes all array metadata from the instance
  int ClearArrayInfo();

  /// appends metadata for the named array to the instance
  int CopyArrayInfo(const sensei::MeshMetadataPtr &other,
    const std::string &arrayName);

  // metadata: the following metadata fields are available.  fields marked
  // "all" are required for all mesh types.  other fields may be required for
  // specific mesh types as indicated and/or be optional. optional fields are
  // expensive to generate and only generated when their corresponding flag is
  // see. See MeshMetadataFlags. AMR metadata is stored in flat arrays, first
  // level 0, then level 1 and so on until the finest level. The index of the
  // array will give you the SVTK composite dataset index,

  bool GlobalView; //< tells if the information describes data on this rank or all ranks.
                   // Passed into Set methods, Get methods generate the desired view.

  std::string MeshName;              ///< name of mesh (all)
  int MeshType;                      ///< container mesh type (all)
  int BlockType;                     ///< block mesh type (all)
  int NumBlocks;                     ///< global number of blocks (all)
  std::vector<int> NumBlocksLocal;   ///< number of blocks on each rank (all)
  std::array<int,6> Extent;          ///< global cell index space extent (Cartesian, AMR, optional)
  std::array<double,6> Bounds;       ///< global bounding box (all, optional)
  int CoordinateType;                ///< type enum of point data (unstructured, optional)
  long NumPoints;                    ///< total number of points in all blocks (all, optional)
  long NumCells;                     ///< total number of cells in all blocks (all, optional)
  long CellArraySize;                ///< total cell array size in all blocks (unstructured, optional)
  int CellArrayType;                 ///< element type of the cell array (unstructured, optional)
  int NumArrays;                     ///< number of arrays (all)
  int NumGhostCells;                 ///< number of ghost cell layers (all)
  int NumGhostNodes;                 ///< number of ghost node layers (all)
  int NumLevels;                     ///< number of AMR levels (AMR)
  int StaticMesh;                    ///< non zero if the mesh does not change in time (all)

  std::vector<std::string> ArrayName; ///< name of each data array (all)
  std::vector<int> ArrayCentering;    ///< centering of each data array (all)
  std::vector<int> ArrayComponents;   ///< number of components of each array (all)
  std::vector<int> ArrayType;         ///< type enum of each data array (all)
  std::vector<std::array<double,2>> ArrayRange; ///< global min,max of each array (all, optional)

  std::vector<int> BlockOwner;             ///< rank where each block resides (all, optional)
  std::vector<int> BlockIds;               ///< global id of each block (all, optional)

                                           ///< note: for AMR BlockNumPoints and BlockNumCells are always global
  std::vector<long> BlockNumPoints;        ///< number of points for each block (all, optional)
  std::vector<long> BlockNumCells;         ///< number of cells for each block (all, optional)
  std::vector<long> BlockCellArraySize;    ///< cell array size for each block (unstructured, optional)

                                                 // note: for AMR BlockExtents and BlockBounds are always global
  std::vector<std::array<int,6>> BlockExtents;   //< index space extent of each block [i0,i1, j0,j1, k0,k1] (Cartesian, AMR, optional)
  std::vector<std::array<double,6>> BlockBounds; //< bounds of each block [x0,x1, y0,y1, z0,z1] (all, optional)

  std::vector<std::vector<std::array<double,2>>> BlockArrayRange; //< min max of each array on each block.
                                                                  // indexed by block then array. (all, optional)

  std::vector<std::array<int,3>> RefRatio; //< refinement ratio in i,j, and k directions for each level (AMR)
  std::vector<int> BlocksPerLevel;         //< number of blocks in each level (AMR)
  std::vector<int> BlockLevel;             //< AMR level of each block (AMR)
  std::array<int,3> PeriodicBoundary;      //< flag indicating presence of a periodic boundary in the i,j,k direction (all)


  sensei::MeshMetadataFlags Flags;  //< flags indicate which optional fields are needed
                                    // some feilds are optional because they are costly
                                    // to generate and not universally used on the analysis
                                    // side.

protected:
  MeshMetadata() : GlobalView(false), MeshName(),
    MeshType(SVTK_MULTIBLOCK_DATA_SET), BlockType(SVTK_DATA_SET), NumBlocks(0),
    NumBlocksLocal(), Extent(), Bounds(), CoordinateType(SVTK_DOUBLE),
    NumPoints(0), NumCells(0), CellArraySize(0), CellArrayType(SVTK_TYPE_INT64),
    NumArrays(0), NumGhostCells(0), NumGhostNodes(0), NumLevels(0),
    StaticMesh(0), ArrayName(), ArrayCentering(), ArrayType(),
    ArrayRange(),BlockOwner(), BlockIds(), BlockNumPoints(), BlockNumCells(),
    BlockCellArraySize(), BlockExtents(), BlockBounds(), BlockArrayRange(),
    RefRatio(), BlocksPerLevel(), BlockLevel(), PeriodicBoundary(), Flags()
    {}
};

};

#endif
