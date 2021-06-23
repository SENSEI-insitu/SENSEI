/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPolyDataInternals.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class svtkPolyDataInternals
 * @brief Store mapping from svtkPolyData cell ids to internal cell array ids.
 *
 * Optimized data structure for storing internal cell ids and type information
 * for svtkPolyData datasets.
 *
 * Since svtkPolyData only supports a handful of types, the type information
 * is compressed to four bits -- the first two indicate which internal
 * svtkCellArray object a cell is stored in (verts, lines, polys, strips), and
 * the second two bits indicate which type of cell (e.g. lines vs polylines,
 * triangles vs quads vs polygons, etc), as well as whether or not the cell
 * has been deleted from the svtkPolyData.
 *
 * These four bits are stored at the top of a 64 bit index, and the remaining
 * 60 bits store the cell id. This implies that the internal cell arrays cannot
 * store more than 2^60 cells each, a reasonable limit for modern hardware.
 *
 * TaggedCellId structure:
 *  66 66 555555555544444444443333333333222222222211111111110000000000
 *  32 10 987654321098765432109876543210987654321098765432109876543210
 * +--+--+------------------------------------------------------------+
 * |00|00|000000000000000000000000000000000000000000000000000000000000|
 * +^-+^-+^-----------------------------------------------------------+
 *  |  |  |
 *  |  |  |> Bottom 60 bits of cellId
 *  |  |> Type variant / deleted
 *  |> Target cell array
 *
 * The supported cell types are:
 *
 * - SVTK_VERTEX
 * - SVTK_POLY_VERTEX
 * - SVTK_LINE
 * - SVTK_POLY_LINE
 * - SVTK_TRIANGLE
 * - SVTK_QUAD
 * - SVTK_POLYGON
 * - SVTK_TRIANGLE_STRIP
 */

#ifndef svtkPolyDataInternals_h
#define svtkPolyDataInternals_h

#ifndef __SVTK_WRAP__ // Don't wrap this class.

#include "svtkCommonDataModelModule.h" // For export macro

#include "svtkCellType.h"
#include "svtkObject.h"
#include "svtkType.h"

#include <cstdlib> // for std::size_t
#include <vector>  // for CellMap implementation.

namespace svtkPolyData_detail
{

static constexpr svtkTypeUInt64 CELLID_MASK = 0x0fffffffffffffffull;
static constexpr svtkTypeUInt64 SHIFTED_TYPE_INDEX_MASK = 0xf000000000000000ull;
static constexpr svtkTypeUInt64 TARGET_MASK = 0x3ull << 62;
static constexpr svtkTypeUInt64 TYPE_VARIANT_MASK = 0x3ull << 60;

// Enumeration of internal cell array targets.
enum class Target : svtkTypeUInt64
{
  Verts = 0x0ull << 62,
  Lines = 0x1ull << 62,
  Polys = 0x2ull << 62,
  Strips = 0x3ull << 62,
};

// Enumeration of type variants.
enum class TypeVariant : svtkTypeUInt64
{
  Dead = 0x0ull << 60,
  Var1 = 0x1ull << 60,
  Var2 = 0x2ull << 60,
  Var3 = 0x3ull << 60,
};

// Lookup table to convert a type index (TaggedCellId.GetTypeIndex()) into
// a SVTK cell type.
// The type index is the highest four bits of the encoded value, eg. the
// target and type variant information.
static constexpr unsigned char TypeTable[16] = {
  SVTK_EMPTY_CELL,     // 0000b | Verts  | Dead
  SVTK_VERTEX,         // 0001b | Verts  | Var1
  SVTK_POLY_VERTEX,    // 0010b | Verts  | Var2
  SVTK_EMPTY_CELL,     // 0011b | Verts  | Var3
  SVTK_EMPTY_CELL,     // 0100b | Lines  | Dead
  SVTK_LINE,           // 0101b | Lines  | Var1
  SVTK_POLY_LINE,      // 0110b | Lines  | Var2
  SVTK_EMPTY_CELL,     // 0111b | Lines  | Var3
  SVTK_EMPTY_CELL,     // 1000b | Polys  | Dead
  SVTK_TRIANGLE,       // 1001b | Polys  | Var1
  SVTK_QUAD,           // 1010b | Polys  | Var2
  SVTK_POLYGON,        // 1011b | Polys  | Var3
  SVTK_EMPTY_CELL,     // 1100b | Strips | Dead
  SVTK_TRIANGLE_STRIP, // 1101b | Strips | Var1
  SVTK_EMPTY_CELL,     // 1110b | Strips | Var2
  SVTK_EMPTY_CELL,     // 1111b | Strips | Var3
};

// Convenience method to concatenate a target and type variant into the low
// four bits of a single byte. Used to build the TargetVarTable.
static constexpr unsigned char GenTargetVar(Target target, TypeVariant var) noexcept
{
  return static_cast<unsigned char>(
    (static_cast<svtkTypeUInt64>(target) | static_cast<svtkTypeUInt64>(var)) >> 60);
}

// Lookup table that maps a SVTK cell type (eg. SVTK_TRIANGLE) into a target +
// type variant byte.
static constexpr unsigned char TargetVarTable[10] = {
  GenTargetVar(Target::Verts, TypeVariant::Dead),  // 0 | SVTK_EMPTY_CELL
  GenTargetVar(Target::Verts, TypeVariant::Var1),  // 1 | SVTK_VERTEX
  GenTargetVar(Target::Verts, TypeVariant::Var2),  // 2 | SVTK_POLY_VERTEX
  GenTargetVar(Target::Lines, TypeVariant::Var1),  // 3 | SVTK_LINE
  GenTargetVar(Target::Lines, TypeVariant::Var2),  // 4 | SVTK_POLY_LINE
  GenTargetVar(Target::Polys, TypeVariant::Var1),  // 5 | SVTK_TRIANGLE
  GenTargetVar(Target::Strips, TypeVariant::Var1), // 6 | SVTK_TRIANGLE_STRIP
  GenTargetVar(Target::Polys, TypeVariant::Var3),  // 7 | SVTK_POLYGON
  GenTargetVar(Target::Polys, TypeVariant::Var2),  // 8 | SVTK_PIXEL (treat as quad)
  GenTargetVar(Target::Polys, TypeVariant::Var2),  // 9 | SVTK_QUAD
};

// Thin wrapper around a svtkTypeUInt64 that encodes a target cell array,
// cell type, deleted status, and 60-bit cell id.
struct SVTKCOMMONDATAMODEL_EXPORT TaggedCellId
{
  // Encode a cell id and a SVTK cell type (eg. SVTK_TRIANGLE) into a
  // svtkTypeUInt64.
  static svtkTypeUInt64 Encode(svtkIdType cellId, SVTKCellType type) noexcept
  {
    const size_t typeIndex = static_cast<size_t>(type);
    return ((static_cast<svtkTypeUInt64>(cellId) & CELLID_MASK) |
      (static_cast<svtkTypeUInt64>(TargetVarTable[typeIndex]) << 60));
  }

  TaggedCellId() noexcept = default;

  // Create a TaggedCellId from a cellId and cell type (e.g. SVTK_TRIANGLE).
  TaggedCellId(svtkIdType cellId, SVTKCellType cellType) noexcept : Value(Encode(cellId, cellType)) {}

  TaggedCellId(const TaggedCellId&) noexcept = default;
  TaggedCellId(TaggedCellId&&) noexcept = default;
  TaggedCellId& operator=(const TaggedCellId&) noexcept = default;
  TaggedCellId& operator=(TaggedCellId&&) noexcept = default;

  // Get an enum value describing the internal svtkCellArray target used to
  // store this cell.
  Target GetTarget() const noexcept { return static_cast<Target>(this->Value & TARGET_MASK); }

  // Get the SVTK cell type value (eg. SVTK_TRIANGLE) as a single byte.
  unsigned char GetCellType() const noexcept { return TypeTable[this->GetTypeIndex()]; }

  // Get the cell id used by the target svtkCellArray to store this cell.
  svtkIdType GetCellId() const noexcept { return static_cast<svtkIdType>(this->Value & CELLID_MASK); }

  // Update the cell id. Most useful with the CellMap::InsertNextCell(type)
  // signature.
  void SetCellId(svtkIdType cellId) noexcept
  {
    this->Value &= SHIFTED_TYPE_INDEX_MASK;
    this->Value |= (static_cast<svtkTypeUInt64>(cellId) & CELLID_MASK);
  }

  // Mark this cell as deleted.
  void MarkDeleted() noexcept { this->Value &= ~TYPE_VARIANT_MASK; }

  // Returns true if the cell has been deleted.
  bool IsDeleted() const noexcept { return (this->Value & TYPE_VARIANT_MASK) == 0; }

private:
  svtkTypeUInt64 Value;

  // These shouldn't be needed outside of this struct. You're probably looking
  // for GetCellType() instead.
  TypeVariant GetTypeVariant() const noexcept
  {
    return static_cast<TypeVariant>(this->Value & TYPE_VARIANT_MASK);
  }
  std::size_t GetTypeIndex() const noexcept { return static_cast<std::size_t>(this->Value >> 60); }
};

// Thin wrapper around a std::vector<TaggedCellId> to allow shallow copying, etc
class SVTKCOMMONDATAMODEL_EXPORT CellMap : public svtkObject
{
public:
  static CellMap* New();
  svtkTypeMacro(CellMap, svtkObject);

  static bool ValidateCellType(SVTKCellType cellType) noexcept
  {
    // 1-9 excluding 8 (SVTK_PIXEL):
    return cellType > 0 && cellType <= 10 && cellType != SVTK_PIXEL;
  }

  static bool ValidateCellId(svtkIdType cellId) noexcept
  {
    // is positive, won't truncate:
    return (
      (static_cast<svtkTypeUInt64>(cellId) & CELLID_MASK) == static_cast<svtkTypeUInt64>(cellId));
  }

  void DeepCopy(CellMap* other)
  {
    if (other)
    {
      this->Map = other->Map;
    }
    else
    {
      this->Map.clear();
    }
  }

  void SetCapacity(svtkIdType numCells) { this->Map.reserve(static_cast<std::size_t>(numCells)); }

  TaggedCellId& GetTag(svtkIdType cellId) { return this->Map[static_cast<std::size_t>(cellId)]; }

  const TaggedCellId& GetTag(svtkIdType cellId) const
  {
    return this->Map[static_cast<std::size_t>(cellId)];
  }

  // Caller must ValidateCellType first.
  void InsertNextCell(svtkIdType cellId, SVTKCellType cellType)
  {
    this->Map.emplace_back(cellId, cellType);
  }

  // Caller must ValidateCellType and ValidateCellId first.
  // useful for reusing the target lookup from cellType and then calling
  // TaggedCellId::SetCellId later.
  TaggedCellId& InsertNextCell(SVTKCellType cellType)
  {
    this->Map.emplace_back(svtkIdType(0), cellType);
    return this->Map.back();
  }

  svtkIdType GetNumberOfCells() const { return static_cast<svtkIdType>(this->Map.size()); }

  void Reset() { this->Map.clear(); }

  void Squeeze()
  {
    this->Map.shrink_to_fit(); // yaaaaay c++11
  }

  // In rounded-up kibibytes, as is SVTK's custom:
  unsigned long GetActualMemorySize() const
  {
    return static_cast<unsigned long>(sizeof(TaggedCellId) * this->Map.capacity() + 1023) / 1024;
  }

protected:
  CellMap();
  ~CellMap() override;

  std::vector<TaggedCellId> Map;

private:
  CellMap(const CellMap&) = delete;
  CellMap& operator=(const CellMap&) = delete;
};

} // end namespace svtkPolyData_detail

#endif // __SVTK_WRAP__
#endif // svtkPolyDataInternals.h

// SVTK-HeaderTest-Exclude: svtkPolyDataInternals.h
