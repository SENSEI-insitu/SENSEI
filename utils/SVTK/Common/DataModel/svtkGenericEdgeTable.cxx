/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGenericEdgeTable.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkGenericEdgeTable.h"
#include "svtkObjectFactory.h"

#include <cassert>
#include <vector>

svtkStandardNewMacro(svtkGenericEdgeTable);

static int PRIME_NUMBERS[] = { 1, 3, 7, 13, 31, 61, 127, 251, 509, 1021, 2039, 4093 };

// Description:
// Constructor with a scalar field of `size' doubles.
// \pre positive_number_of_components: size>0
svtkGenericEdgeTable::PointEntry::PointEntry(int size)
{
  assert("pre: positive_number_of_components" && size > 0);
  this->Reference = -10;

  this->Coord[0] = -100;
  this->Coord[1] = -100;
  this->Coord[2] = -100;
  this->Scalar = new double[size];
  this->numberOfComponents = size;
}

class svtkEdgeTablePoints
{
public:
  typedef std::vector<svtkGenericEdgeTable::PointEntry> VectorPointTableType;
  typedef std::vector<VectorPointTableType> PointTableType;

  void Resize(svtkIdType size);
  void LoadFactor();
  void DumpPoints();

  PointTableType PointVector;
  svtkIdType Modulo;
};

//-----------------------------------------------------------------------------
void svtkEdgeTablePoints::Resize(svtkIdType newSize)
{
  svtkIdType size = static_cast<svtkIdType>(PointVector.size());

  if (size <= newSize)
  {
    PointVector.resize(newSize);
    int index = static_cast<int>(log(static_cast<double>(newSize)) / log(2.));
    this->Modulo = PRIME_NUMBERS[index];
    // cout << "this->Modulo:" << newSize << "," << index << ","
    //     << this->Modulo << endl;
  }

  assert(static_cast<unsigned>(size) < PointVector.size()); // valid size
  // For now you are not supposed to use this method
  assert(0);
}

//-----------------------------------------------------------------------------
void svtkEdgeTablePoints::LoadFactor()
{
  svtkIdType numEntries = 0;
  svtkIdType numBins = 0;

  svtkIdType size = static_cast<svtkIdType>(PointVector.size());
  cerr << "EdgeTablePoints:\n";
  for (int i = 0; i < size; i++)
  {
    numEntries += static_cast<svtkIdType>(PointVector[i].size());
    if (!PointVector[i].empty())
      numBins++;
    cerr << PointVector[i].size() << ",";
  }
  cerr << "\n";
  cout << size << "," << numEntries << "," << numBins << "," << Modulo << "\n";
}

//-----------------------------------------------------------------------------
void svtkEdgeTablePoints::DumpPoints()
{
  svtkIdType size = static_cast<svtkIdType>(PointVector.size());
  for (int i = 0; i < size; i++)
  {
    VectorPointTableType v = PointVector[i];
    for (VectorPointTableType::iterator it = v.begin(); it != v.end(); ++it)
    {
      // PointEntry
      cout << "PointEntry: " << it->PointId << " " << it->Reference << ":(" << it->Coord[0] << ","
           << it->Coord[1] << "," << it->Coord[2] << ")" << endl;
    }
  }
}

//-----------------------------------------------------------------------------
class svtkEdgeTableEdge
{
public:
  typedef std::vector<svtkGenericEdgeTable::EdgeEntry> VectorEdgeTableType;
  typedef std::vector<VectorEdgeTableType> EdgeTableType;

  void Resize(svtkIdType size);
  void LoadFactor();
  void DumpEdges();

  EdgeTableType Vector;
  svtkIdType Modulo;
};

//-----------------------------------------------------------------------------
void svtkEdgeTableEdge::Resize(svtkIdType newSize)
{
  svtkIdType size = static_cast<svtkIdType>(Vector.size());

  if (size <= newSize)
  {
    Vector.resize(newSize);
    int index = static_cast<int>(log(static_cast<double>(newSize)) / log(2.));
    this->Modulo = PRIME_NUMBERS[index];
    cout << "this->Modulo:" << index << ":" << this->Modulo << endl;
  }

  // For now you are not supposed to use this method
  assert(0);
}

//-----------------------------------------------------------------------------
void svtkEdgeTableEdge::LoadFactor()
{
  svtkIdType numEntry = 0;
  svtkIdType numBins = 0;

  svtkIdType size = static_cast<svtkIdType>(Vector.size());
  cerr << "EdgeTableEdge:\n";
  for (int i = 0; i < size; i++)
  {
    VectorEdgeTableType v = Vector[i];
    numEntry += static_cast<svtkIdType>(v.size());
    if (!v.empty())
      numBins++;
  }
  cerr << "\n";
  cerr << size << "," << numEntry << "," << numBins << "," << Modulo << "\n";
}

//-----------------------------------------------------------------------------
void svtkEdgeTableEdge::DumpEdges()
{
  svtkIdType size = static_cast<svtkIdType>(Vector.size());
  for (int i = 0; i < size; i++)
  {
    VectorEdgeTableType v = Vector[i];
    for (VectorEdgeTableType::iterator it = v.begin(); it != v.end(); ++it)
    {
      // EdgeEntry
      cout << "EdgeEntry: (" << it->E1 << "," << it->E2 << ") " << it->Reference << ","
           << it->ToSplit << "," << it->PtId << endl;
    }
  }
}

//-----------------------------------------------------------------------------
static inline void OrderEdge(svtkIdType& e1, svtkIdType& e2)
{
  svtkIdType temp1 = e1;
  svtkIdType temp2 = e2;
  e1 = temp1 < temp2 ? temp1 : temp2;
  e2 = temp1 > temp2 ? temp1 : temp2;
}

//-----------------------------------------------------------------------------
// Instantiate object based on maximum point id.
svtkGenericEdgeTable::svtkGenericEdgeTable()
{
  this->EdgeTable = new svtkEdgeTableEdge;
  this->HashPoints = new svtkEdgeTablePoints;

  // Default to only one component
  this->NumberOfComponents = 1;

  // The whole problem is here to find the proper size for a descent hash table
  // Since we do not allow check our size as we go the hash table
  // Should be big enough from the beginning otherwise we'll lose the
  // constant time access
  // But on the other hand we do not want it to be too big for mem consumption
  // A compromise of 4093 was found fo be working in a lot of case
#if 1
  this->EdgeTable->Vector.resize(4093);
  this->EdgeTable->Modulo = 4093;
  this->HashPoints->PointVector.resize(4093); // 127
  this->HashPoints->Modulo = 4093;
#else
  this->EdgeTable->Vector.resize(509);
  this->EdgeTable->Modulo = 509;
  this->HashPoints->PointVector.resize(127);
  this->HashPoints->Modulo = 127;
#endif

  this->LastPointId = 0;
}

//-----------------------------------------------------------------------------
svtkGenericEdgeTable::~svtkGenericEdgeTable()
{
  delete this->EdgeTable;
  delete this->HashPoints;
}

//-----------------------------------------------------------------------------
// We assume the edge is not being split:
void svtkGenericEdgeTable::InsertEdge(svtkIdType e1, svtkIdType e2, svtkIdType cellId, int ref)
{
  svtkIdType ptId;
  this->InsertEdge(e1, e2, cellId, ref, 0, ptId);
}

//-----------------------------------------------------------------------------
// the edge is being split and we want the new ptId
void svtkGenericEdgeTable::InsertEdge(
  svtkIdType e1, svtkIdType e2, svtkIdType cellId, int ref, svtkIdType& ptId)
{
  this->InsertEdge(e1, e2, cellId, ref, 1, ptId);
}

//-----------------------------------------------------------------------------
void svtkGenericEdgeTable::InsertEdge(
  svtkIdType e1, svtkIdType e2, svtkIdType cellId, int ref, int toSplit, svtkIdType& ptId)
{
  if (e1 == e2)
  {
    svtkErrorMacro(<< "Not an edge:" << e1 << "," << e2);
  }
  assert("pre: not degenerated edge" && e1 != e2);

  // reorder so that e1 < e2;
  OrderEdge(e1, e2);

  svtkIdType pos = this->HashFunction(e1, e2);

  // Be careful with reference the equal is not overloaded
  svtkEdgeTableEdge::VectorEdgeTableType& vect = this->EdgeTable->Vector[pos];

  // Need to check size again
  // Here we just puch_back at the end,
  // the vector should not be very long and should not contain any empty spot.
  EdgeEntry newEntry;
  newEntry.E1 = e1;
  newEntry.E2 = e2;
  newEntry.Reference = ref;
  newEntry.ToSplit = toSplit;
  newEntry.CellId = cellId;

  if (newEntry.ToSplit)
  {
    newEntry.PtId = ptId = this->LastPointId++;
  }
  else
  {
    newEntry.PtId = ptId = -1;
  }

  //  svtkDebugMacro( << "Inserting Edge:" << e1 << "," << e2 << ": ref="  <<
  //    newEntry.Reference << ", cellId=" << cellId << ", split=" << toSplit <<
  //    ", ptId=" << newEntry.PtId );

  vect.push_back(newEntry);
}

//-----------------------------------------------------------------------------
// Try to remove an edge, in fact decrement the ref count
int svtkGenericEdgeTable::RemoveEdge(svtkIdType e1, svtkIdType e2)
{
  int ref = 0;
  // reorder so that e1 < e2;
  OrderEdge(e1, e2);

  // svtkDebugMacro( << "RemoveEdge:" << e1 << "," << e2 );

  svtkIdType pos = this->HashFunction(e1, e2);
  assert("check: valid range po" && static_cast<unsigned>(pos) < this->EdgeTable->Vector.size());

  // Need to check size first
  svtkEdgeTableEdge::VectorEdgeTableType& vect = this->EdgeTable->Vector[pos];

  int found = 0;

  svtkEdgeTableEdge::VectorEdgeTableType::iterator it;
  for (it = vect.begin(); it != vect.end();)
  {
    if (it->E1 == e1 && it->E2 == e2)
    {
      if (--it->Reference == 0)
      {
        // Ok this edge is about to be physically removed, remove also the point
        // is contained, if one:
        if (it->ToSplit)
        {
          assert("check: positive id" && it->PtId >= 0);

          this->RemovePoint(it->PtId);
        }
      }
      found = 1;
      ref = it->Reference;
    }

    if (it->E1 == e1 && it->E2 == e2 && it->Reference == 0)
    {
      it = vect.erase(it);
    }
    else
    {
      ++it;
    }
  }

  if (!found)
  {
    // We did not find any corresponding entry to erase, warn user
    svtkErrorMacro(<< "No entry were found in the hash table for edge:" << e1 << "," << e2);
    assert("check: not found" && 0);
  }

  return ref;
}

//-----------------------------------------------------------------------------
int svtkGenericEdgeTable::CheckEdge(svtkIdType e1, svtkIdType e2, svtkIdType& ptId)
{
  // reorder so that e1 < e2;
  OrderEdge(e1, e2);

  // svtkDebugMacro( << "Checking edge:" << e1 << "," << e2  );

  svtkIdType pos = this->HashFunction(e1, e2);

  if (static_cast<unsigned>(pos) >= this->EdgeTable->Vector.size())
  {
    svtkDebugMacro(<< "No entry were found in the hash table");
    return -1;
  }

  assert("check: valid range pos" && static_cast<unsigned>(pos) < this->EdgeTable->Vector.size());
  // Need to check size first
  const svtkEdgeTableEdge::VectorEdgeTableType& vect = this->EdgeTable->Vector[pos];
  for (auto it = vect.begin(); it != vect.end(); ++it)
  {
    if ((it->E1 == e1) && (it->E2 == e2))
    {
      ptId = it->PtId;
      return it->ToSplit;
    }
  }

  // We did not find any corresponding entry, warn user
  svtkDebugMacro(<< "No entry were found in the hash table");
  return -1;
}

//-----------------------------------------------------------------------------
int svtkGenericEdgeTable::IncrementEdgeReferenceCount(svtkIdType e1, svtkIdType e2, svtkIdType cellId)
{
  int index;
  // reorder so that e1 < e2;
  OrderEdge(e1, e2);

  // svtkDebugMacro( << "IncrementEdgeReferenceCount:" << e1 << "," << e2  );

  svtkIdType pos = this->HashFunction(e1, e2);
  assert("check: valid range pos" && static_cast<unsigned>(pos) < this->EdgeTable->Vector.size());

  // Need to check size first
  svtkEdgeTableEdge::VectorEdgeTableType& vect = this->EdgeTable->Vector[pos];

  int vectsize = static_cast<int>(vect.size());
  for (index = 0; index < vectsize; index++)
  {
    EdgeEntry& ent = vect[index];
    if ((ent.E1 == e1) && (ent.E2 == e2))
    {
      if (ent.CellId == cellId)
      {
        ent.Reference++;
        // svtkDebugMacro( << "Incrementing: (" << e1 << "," << e2 << ") " << ent.Reference );
      }
      else
      {
        // If cellIds are different it means we pass from one cell to another
        // therefore the first time we should not increment the ref count
        // as it has already been taken into account.
        ent.CellId = cellId;
      }
      return -1;
    }
  }

  // We did not find any corresponding entry, warn user
  svtkErrorMacro(<< "No entry were found in the hash table");
  return -1;
}

//-----------------------------------------------------------------------------
int svtkGenericEdgeTable::CheckEdgeReferenceCount(svtkIdType e1, svtkIdType e2)
{
  int index;
  OrderEdge(e1, e2); // reorder so that e1 < e2;

  // svtkDebugMacro( << "CheckEdgeReferenceCount:" << e1 << "," << e2  );

  svtkIdType pos = this->HashFunction(e1, e2);
  assert("check: valid range pos" && static_cast<unsigned>(pos) < this->EdgeTable->Vector.size());

  // Need to check size first
  svtkEdgeTableEdge::VectorEdgeTableType& vect = this->EdgeTable->Vector[pos];

  int vectsize = static_cast<int>(vect.size());
  for (index = 0; index < vectsize; index++)
  {
    EdgeEntry& ent = vect[index];
    if ((ent.E1 == e1) && (ent.E2 == e2))
    {
      assert("check: positive reference" && ent.Reference >= 0);
      return ent.Reference;
    }
  }

  // We did not find any corresponding entry, warn user
  svtkErrorMacro(<< "No entry were found in the hash table");
  return -1;
}

//-----------------------------------------------------------------------------
svtkIdType svtkGenericEdgeTable::HashFunction(svtkIdType e1, svtkIdType e2)
{
  return (e1 + e2) % this->EdgeTable->Modulo;
}

//-----------------------------------------------------------------------------
void svtkGenericEdgeTable::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//-----------------------------------------------------------------------------
void svtkGenericEdgeTable::Initialize(svtkIdType start)
{
  if (this->LastPointId)
  {
    // if different from zero then raise problem:
    svtkDebugMacro(<< "You are not supposed to initialize during algorithm");
    return;
  }

  this->LastPointId = start;
}

//-----------------------------------------------------------------------------
// Description:
// Return the total number of components for the point-centered attributes.
// \post positive_result: result>0
int svtkGenericEdgeTable::GetNumberOfComponents()
{
  return this->NumberOfComponents;
}

//-----------------------------------------------------------------------------
// Description:
// Set the total number of components for the point-centered attributes.
void svtkGenericEdgeTable::SetNumberOfComponents(int count)
{
  assert("pre: positive_count" && count > 0);
  this->NumberOfComponents = count;
}

//-----------------------------------------------------------------------------
svtkIdType svtkGenericEdgeTable::HashFunction(svtkIdType ptId)
{
  return ptId % this->HashPoints->Modulo;
}
//-----------------------------------------------------------------------------
// Check if point ptId exist in the hash table
int svtkGenericEdgeTable::CheckPoint(svtkIdType ptId)
{
  int index;
  svtkIdType pos = this->HashFunction(ptId);

  if (static_cast<unsigned>(pos) >= this->HashPoints->PointVector.size())
  {
    return 0;
  }

  assert(
    "check: valid range pos" && static_cast<unsigned>(pos) < this->HashPoints->PointVector.size());

  // Be careful with reference the equal is not overloaded
  svtkEdgeTablePoints::VectorPointTableType& vect = this->HashPoints->PointVector[pos];

  int vectsize = static_cast<int>(vect.size());
  for (index = 0; index < vectsize; index++)
  {
    if (vect[index].PointId == ptId)
    {
      return 1;
    }
  }

  if (index == vectsize)
  {
    // We did not find any corresponding entry
    return 0;
  }

  svtkErrorMacro(<< "Error, impossible case");
  return -1;
}

//-----------------------------------------------------------------------------
// Find point coordinate and scalar associated of point ptId
//
int svtkGenericEdgeTable::CheckPoint(svtkIdType ptId, double point[3], double* scalar)
{
  // pre: sizeof(scalar)==GetNumberOfComponents()
  int index;
  svtkIdType pos = this->HashFunction(ptId);
  assert(
    "check: valid range pos" && static_cast<unsigned>(pos) < this->HashPoints->PointVector.size());

  // Be careful with reference the equal is not overloaded
  svtkEdgeTablePoints::VectorPointTableType& vect = this->HashPoints->PointVector[pos];

  // Need to check size again

  int vectsize = static_cast<int>(vect.size());
  for (index = 0; index < vectsize; index++)
  {
    PointEntry& ent = vect[index];
    if (ent.PointId == ptId)
    {
      memcpy(point, ent.Coord, sizeof(double) * 3);
      memcpy(scalar, ent.Scalar, sizeof(double) * this->NumberOfComponents);
      return 1;
    }
  }

  if (index == vectsize)
  {
    // We did not find any corresponding entry, warn user
    svtkErrorMacro(<< "No entry were found in the hash table:" << ptId);
    return 0;
  }

  assert("check: TODO" && 0);
  return 1;
}

//-----------------------------------------------------------------------------
void svtkGenericEdgeTable::InsertPoint(svtkIdType ptId, double point[3])
{
  svtkIdType pos = this->HashFunction(ptId);

  // Need to check size first
  // this->HashPoints->Resize( pos );
  assert(
    "check: valid range pos" && static_cast<unsigned>(pos) < this->HashPoints->PointVector.size());

  // Be careful with reference the equal is not overloaded
  svtkEdgeTablePoints::VectorPointTableType& vect = this->HashPoints->PointVector[pos];

  // Need to check size again
  // Here we just puch_back at the end,
  // the vector should not be very long and should not contain any empty spot.
  PointEntry newEntry(this->NumberOfComponents); // = vect.back();
  newEntry.PointId = ptId;
  memcpy(newEntry.Coord, point, sizeof(double) * 3);
  newEntry.Reference = 1;

  //  svtkDebugMacro( << "Inserting Point:" << ptId << ":"
  //    << point[0] << "," << point[1] << "," << point[2] );
  vect.push_back(newEntry);
}

//-----------------------------------------------------------------------------
void svtkGenericEdgeTable::RemovePoint(svtkIdType ptId)
{
  int found = 0;
  svtkIdType pos = this->HashFunction(ptId);

  // Need to check size first
  assert(
    "check: valid range pos" && static_cast<unsigned>(pos) < this->HashPoints->PointVector.size());

  // Be careful with reference the equal is not overloaded
  svtkEdgeTablePoints::VectorPointTableType& vect = this->HashPoints->PointVector[pos];

  // svtkDebugMacro( << "Remove Point:" << ptId << ":" << vect.size() );

  svtkEdgeTablePoints::VectorPointTableType::iterator it;
  for (it = vect.begin(); it != vect.end();)
  {
    PointEntry& ent = *it;

    if (ent.PointId == ptId)
    {
      --ent.Reference;
      found = 1;
    }

    if (ent.PointId == ptId && ent.Reference == 0)
    {
      it = vect.erase(it);
    }
    else
    {
      ++it;
    }
  }

  if (!found)
  {
    // We did not find any corresponding entry, warn user
    svtkErrorMacro(<< "No entry were found in the hash table");
  }
}

//-----------------------------------------------------------------------------
void svtkGenericEdgeTable::InsertPointAndScalar(svtkIdType ptId, double pt[3], double* s)
{
  // sizeof(s)=this->NumberOfComponents
  svtkIdType pos = this->HashFunction(ptId);

  // Need to check size first
  // this->HashPoints->Resize( pos );
  /*if (!(static_cast<unsigned>(pos) < this->HashPoints->PointVector.size()))
  {
    int kk = 2;
    kk++;
  }*/

  // Be careful with reference the equal is not overloaded
  svtkEdgeTablePoints::VectorPointTableType& vect = this->HashPoints->PointVector[pos];

  // Please keep the following:
#if 0
  //Need to check size again
  //Here we just puch_back at the end,
  //the vector should not be very long and should not contain any empty spot.
  for (index=0; index<vect.size(); index++)
  {
    PointEntry ent = vect[index];
    if( ent.PointId == ptId )
    {
      //The point was already in the table:
      assert("check: same id" &&  ent.PointId == ptId );
      assert("check: same x" && ent.Coord[0] == pt[0]);
      assert("check: same y" && ent.Coord[1] == pt[1]);
      assert("check: same z" && ent.Coord[2] == pt[2]);
      assert("check: same x scalar" && ent.Scalar[0] == s[0]);
      assert("check: same y scalar" && ent.Scalar[1] == s[1]);
      assert("check: same z scalar" && ent.Scalar[2] == s[2]);

      svtkErrorMacro( << "The point was already in the table" ); //FIXME

      return;
    }
  }
#endif

  // We did not find any matching point thus insert it

  PointEntry newEntry(this->NumberOfComponents); // = vect.back();
  newEntry.PointId = ptId;
  memcpy(newEntry.Coord, pt, sizeof(double) * 3);
  memcpy(newEntry.Scalar, s, sizeof(double) * this->NumberOfComponents);
  newEntry.Reference = 1;

  //  svtkDebugMacro( << "InsertPointAndScalar:" << ptId << ":"
  //    << pt[0] << "," << pt[1] << "," << pt[2] << "->"
  //    << s[0] << "," << s[1] << "," << s[2] );

  vect.push_back(newEntry);
}

//-----------------------------------------------------------------------------
void svtkGenericEdgeTable::DumpTable()
{
  this->EdgeTable->DumpEdges();
  this->HashPoints->DumpPoints();
}

//-----------------------------------------------------------------------------
void svtkGenericEdgeTable::IncrementPointReferenceCount(svtkIdType ptId)
{
  unsigned int index;
  int found = 0;
  svtkIdType pos = this->HashFunction(ptId);

  // Need to check size first
  assert(
    "check: valid range pos" && static_cast<unsigned>(pos) < this->HashPoints->PointVector.size());

  // Be careful with reference the equal is not overloaded
  svtkEdgeTablePoints::VectorPointTableType& vect = this->HashPoints->PointVector[pos];

  // svtkDebugMacro(<< "IncrementPointReferenceCount:" << ptId << ":" << vect.size() );

  for (index = 0; index < vect.size(); index++)
  {
    PointEntry& ent = vect[index];
    if (ent.PointId == ptId)
    {
      ent.Reference++;
      found = 1;
    }
  }

  if (!found)
  {
    // We did not find any corresponding entry, warn user
    svtkErrorMacro(<< "No entry were found in the hash table");
  }
}

//-----------------------------------------------------------------------------
void svtkGenericEdgeTable::LoadFactor()
{
  svtkDebugMacro(<< "------ Begin LoadFactor ------- ");

  this->EdgeTable->LoadFactor();
  this->HashPoints->LoadFactor();
}
