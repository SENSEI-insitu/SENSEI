/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestMinimalStandardRandomSequence.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// .NAME
// .SECTION Description
// This program tests the svtkMinimalStandardRandomSequence class.
//
// Correctness test is described in first column, page 1195:
// A seed of 1 at step 1 should give a seed of 1043618065 at step 10001.
//
// ref: "Random Number Generators: Good Ones are Hard to Find,"
// by Stephen K. Park and Keith W. Miller in Communications of the ACM,
// 31, 10 (Oct. 1988) pp. 1192-1201.
// Code is at page 1195, "Integer version 2"

#include "svtkDebugLeaks.h"
#include "svtkMath.h"
#include "svtkMinimalStandardRandomSequence.h"

int TestMinimalStandardRandomSequence(int, char*[])
{
  svtkMinimalStandardRandomSequence* seq = svtkMinimalStandardRandomSequence::New();

  seq->SetSeedOnly(1);

  // Check seed has been set
  bool status = seq->GetSeed() == 1;

  if (status)
  {
    int i = 0;
    while (i < 10000)
    {
      //      cout << "i=" << i << " seed=" << seq->GetSeed()<< endl;
      seq->Next();
      ++i;
    }
    status = seq->GetSeed() == 1043618065;
    if (!status)
    {
      cout << "FAILED: seed is not 1043618065, it is " << seq->GetSeed() << endl;
    }
  }
  else
  {
    cout << "FAILED: seed is not 1, it is " << seq->GetSeed() << endl;
  }

  svtkMath::RandomSeed(1);
  int i = 0;
  while (i < 9997)
  {
    // cout << "i=" << i << " seed=" << svtkMath::GetSeed() << endl;
    svtkMath::Random();
    ++i;
  }
  status = svtkMath::GetSeed() == 1043618065;
  if (!status)
  {
    cout << "FAILED: static seed is not 1043618065, it is " << svtkMath::GetSeed() << endl;
  }

  seq->SetSeed(1);
  i = 0;
  while (i < 9997)
  {
    // cout << "i=" << i << " seed=" << svtkMath::GetSeed() << endl;
    seq->Next();
    ++i;
  }
  status = seq->GetSeed() == 1043618065;
  if (!status)
  {
    cout << "FAILED: seed auto is not 1043618065, it is " << seq->GetSeed() << endl;
  }
  seq->Delete();
  int result;

  if (status)
  {
    // passed.
    result = 0;
  }
  else
  {
    // failed.
    result = 1;
  }
  return result;
}
