/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDistributedGraphHelper.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*
 * Copyright (C) 2008 The Trustees of Indiana University.
 * Use, modification and distribution is subject to the Boost Software
 * License, Version 1.0. (See http://www.boost.org/LICENSE_1_0.txt)
 */
// .NAME svtkDistributedGraphHelper.cxx - distributed graph helper for svtkGraph
//
// .SECTION Description
// Attach a subclass of this helper to a svtkGraph to turn it into a distributed graph
#include "svtkDistributedGraphHelper.h"
#include "svtkGraph.h"
#include "svtkInformation.h"
#include "svtkInformationIntegerKey.h"
#include "svtkStdString.h"
#include "svtkVariant.h"

#include <cassert> // assert()
#include <climits> // CHAR_BIT

svtkInformationKeyMacro(svtkDistributedGraphHelper, DISTRIBUTEDVERTEXIDS, Integer);
svtkInformationKeyMacro(svtkDistributedGraphHelper, DISTRIBUTEDEDGEIDS, Integer);

//----------------------------------------------------------------------------
// class svtkDistributedGraphHelper
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
void svtkDistributedGraphHelper::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  int numProcs = this->Graph->GetInformation()->Get(svtkDataObject::DATA_NUMBER_OF_PIECES());
  int myRank = this->Graph->GetInformation()->Get(svtkDataObject::DATA_PIECE_NUMBER());
  os << indent << "Processor: " << myRank << " of " << numProcs << endl;
}

//----------------------------------------------------------------------------
svtkDistributedGraphHelper::svtkDistributedGraphHelper()
{
  this->Graph = nullptr;
  this->VertexDistribution = nullptr;
}

//----------------------------------------------------------------------------
svtkDistributedGraphHelper::~svtkDistributedGraphHelper() = default;

//----------------------------------------------------------------------------
svtkIdType svtkDistributedGraphHelper::GetVertexOwner(svtkIdType v) const
{
  svtkIdType owner = v;
  int numProcs = this->Graph->GetInformation()->Get(svtkDataObject::DATA_NUMBER_OF_PIECES());

  if (numProcs > 1)
  {
    // An alternative to this obfuscated code is to provide
    // an 'unsigned' equivalent to svtkIdType.  Could then safely
    // do a logical right-shift of bits, e.g.:
    //   owner = (svtkIdTypeUnsigned) v >> this->indexBits;
    if (v & this->signBitMask)
    {
      owner ^= this->signBitMask;               // remove sign bit
      svtkIdType tmp = owner >> this->indexBits; // so can right-shift
      owner = tmp | this->highBitShiftMask;     // and append sign bit back
    }
    else
    {
      owner = v >> this->indexBits;
    }
  }
  else // numProcs = 1
  {
    owner = 0;
  }

  return owner;
}

//----------------------------------------------------------------------------
svtkIdType svtkDistributedGraphHelper::GetVertexIndex(svtkIdType v) const
{
  svtkIdType index = v;
  int numProcs = this->Graph->GetInformation()->Get(svtkDataObject::DATA_NUMBER_OF_PIECES());

  if (numProcs > 1)
  {
    // Shift off the Owner bits.  (Would a mask be faster?)
    index = (v << this->procBits) >> this->procBits;
  }

  return index;
}

//----------------------------------------------------------------------------
svtkIdType svtkDistributedGraphHelper::GetEdgeOwner(svtkIdType e_id) const
{
  svtkIdType owner = e_id;
  int numProcs = this->Graph->GetInformation()->Get(svtkDataObject::DATA_NUMBER_OF_PIECES());

  if (numProcs > 1)
  {
    if (e_id & this->signBitMask)
    {
      owner ^= this->signBitMask;               // remove sign bit
      svtkIdType tmp = owner >> this->indexBits; // so can right-shift
      owner = tmp | this->highBitShiftMask;     // and append sign bit back
    }
    else
    {
      owner = e_id >> this->indexBits;
    }
  }
  else // numProcs = 1
  {
    owner = 0;
  }

  return owner;
}

//----------------------------------------------------------------------------
svtkIdType svtkDistributedGraphHelper::GetEdgeIndex(svtkIdType e_id) const
{
  svtkIdType index = e_id;
  int numProcs = this->Graph->GetInformation()->Get(svtkDataObject::DATA_NUMBER_OF_PIECES());

  if (numProcs > 1)
  {
    // Shift off the Owner bits.  (Would a mask be faster?)
    index = (e_id << this->procBits) >> this->procBits;
  }

  return index;
}

//----------------------------------------------------------------------------
svtkIdType svtkDistributedGraphHelper::MakeDistributedId(int owner, svtkIdType local)
{
  int numProcs = this->Graph->GetInformation()->Get(svtkDataObject::DATA_NUMBER_OF_PIECES());

  if (numProcs > 1)
  {
    assert(owner >= 0 && owner < numProcs);
    return (static_cast<svtkIdType>(owner) << this->indexBits) | local;
  }

  return local;
}

//----------------------------------------------------------------------------
void svtkDistributedGraphHelper::AttachToGraph(svtkGraph* graph)
{
  this->Graph = graph;

  // Some factors and masks to help speed up encoding/decoding {owner,index}
  int numProcs = this->Graph->GetInformation()->Get(svtkDataObject::DATA_NUMBER_OF_PIECES());
  int tmp = numProcs - 1;
  // The following is integer arith equiv of ceil(log2(numProcs)):
  int numProcBits = 0;
  while (tmp != 0)
  {
    tmp >>= 1;
    numProcBits++;
  }
  if (numProcs == 1)
    numProcBits = 1;

  this->signBitMask = SVTK_ID_MIN;
  this->highBitShiftMask = static_cast<svtkIdType>(1) << numProcBits;
  this->procBits = numProcBits + 1;
  this->indexBits = (sizeof(svtkIdType) * CHAR_BIT) - (numProcBits + 1);
}

//----------------------------------------------------------------------------
void svtkDistributedGraphHelper::SetVertexPedigreeIdDistribution(
  svtkVertexPedigreeIdDistribution Func, void* userData)
{
  this->VertexDistribution = Func;
  this->VertexDistributionUserData = userData;
}

//----------------------------------------------------------------------------
svtkIdType svtkDistributedGraphHelper::GetVertexOwnerByPedigreeId(const svtkVariant& pedigreeId)
{
  svtkIdType numProcs = this->Graph->GetInformation()->Get(svtkDataObject::DATA_NUMBER_OF_PIECES());
  if (this->VertexDistribution)
  {
    return (this->VertexDistribution(pedigreeId, this->VertexDistributionUserData) % numProcs);
  }

  // Hash the variant in a very lame way.
  double numericValue;
  svtkStdString stringValue;
  const unsigned char *charsStart, *charsEnd;
  if (pedigreeId.IsNumeric())
  {
    // Convert every numeric value into a double.
    numericValue = pedigreeId.ToDouble();

    // Hash the characters in the double.
    charsStart = reinterpret_cast<const unsigned char*>(&numericValue);
    charsEnd = charsStart + sizeof(double);
  }
  else if (pedigreeId.GetType() == SVTK_STRING)
  {
    stringValue = pedigreeId.ToString();
    charsStart = reinterpret_cast<const unsigned char*>(stringValue.c_str());
    charsEnd = charsStart + stringValue.size();
  }
  else
  {
    svtkErrorMacro("Cannot hash vertex pedigree ID of type " << pedigreeId.GetType());
    return 0;
  }

  unsigned long hash = 5381;
  for (; charsStart != charsEnd; ++charsStart)
  {
    hash = ((hash << 5) + hash) ^ *charsStart;
  }

  return hash % numProcs;
}
