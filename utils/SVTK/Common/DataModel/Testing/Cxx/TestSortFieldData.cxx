/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestSortFieldData.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkDoubleArray.h"
#include "svtkFieldData.h"
#include "svtkIntArray.h"
#include "svtkMath.h"
#include "svtkSortFieldData.h"
#include "svtkStringArray.h"
#include "svtkVariantArray.h"

#include <locale>  // C++ locale
#include <sstream> // to_string chokes some compilers

// A simple test sorts on one of the components of a 3-tuple array, and then
// orders all of the arrays based on the sort indices.
int TestSortFieldData(int, char*[])
{
  // Create the arrays
  svtkIntArray* iArray = svtkIntArray::New();
  iArray->SetName("Int Array");
  iArray->SetNumberOfComponents(3);
  iArray->SetNumberOfTuples(10);

  svtkStringArray* sArray = svtkStringArray::New();
  sArray->SetName("String Array");
  sArray->SetNumberOfComponents(1);
  sArray->SetNumberOfTuples(10);

  svtkDoubleArray* dArray = svtkDoubleArray::New();
  dArray->SetName("Double Array");
  dArray->SetNumberOfComponents(2);
  dArray->SetNumberOfTuples(10);

  svtkVariantArray* vArray = svtkVariantArray::New();
  vArray->SetName("Variant Array");
  vArray->SetNumberOfComponents(1);
  vArray->SetNumberOfTuples(10);

  // Populate the arrays. Mostly random numbers with a some values
  // set to compare against expected output.
  std::ostringstream ostr;
  ostr.imbue(std::locale::classic());
  int permute[] = { 3, 0, 9, 6, 7, 4, 5, 8, 2, 1 };
  for (int i = 0; i < 10; ++i)
  {
    iArray->SetComponent(i, 0, i);
    iArray->SetComponent(permute[i], 1, i);
    iArray->SetComponent(i, 2, static_cast<int>(svtkMath::Random(0, 100)));
    ostr.str(""); // clear it out
    ostr << i;
    sArray->SetValue(permute[i], ostr.str());
    dArray->SetComponent(i, 0, static_cast<double>(svtkMath::Random(-1, 1)));
    dArray->SetComponent(permute[i], 1, static_cast<double>(i));
    vArray->SetValue(permute[i], svtkVariant(ostr.str()));
  }

  // Create the field data
  svtkFieldData* fd = svtkFieldData::New();
  fd->AddArray(iArray);
  fd->AddArray(dArray);
  fd->AddArray(vArray);
  fd->AddArray(sArray);

  // Sort the data
  svtkIdType* idx = svtkSortFieldData::Sort(fd, "Int Array", 1, 1, 0);

  // Now test the result
  cout << "Ordering:\n\t( ";
  for (int i = 0; i < 10; ++i)
  {
    cout << idx[i] << " ";
    if (idx[i] != permute[i])
    {
      cout << "Bad sort order!\n";
      delete[] idx;
      return 1;
    }
  }
  delete[] idx;
  cout << ")";

  cout << "\n\nInteger Array (sorted by component==1):\n";
  for (int i = 0; i < 10; ++i)
  {
    cout << "\t(" << iArray->GetComponent(i, 0) << "," << iArray->GetComponent(i, 1) << ","
         << iArray->GetComponent(i, 2) << ")\n";
    if (iArray->GetComponent(i, 1) != i)
    {
      cout << "Bad sort order!\n";
      return 1;
    }
  }

  cout << "\nDouble Array:\n";
  for (int i = 0; i < 10; ++i)
  {
    cout << "\t(" << dArray->GetComponent(i, 0) << "," << dArray->GetComponent(i, 1) << ")\n";
    if (dArray->GetComponent(i, 1) != i)
    {
      cout << "Bad sort order!\n";
      return 1;
    }
  }

  cout << "\nString Array:\n\t( ";
  for (int i = 0; i < 10; ++i)
  {
    cout << sArray->GetValue(i) << " ";
    ostr.str(""); // clear it out
    ostr << i;
    if (sArray->GetValue(i) != ostr.str())
    {
      cout << "Bad sort order!\n";
      return 1;
    }
  }
  cout << ")\n";

  cout << "\nVariant Array:\n\t( ";
  for (int i = 0; i < 10; ++i)
  {
    cout << vArray->GetValue(i).ToString() << " ";
    ostr.str(""); // clear it out
    ostr << i;
    if (vArray->GetValue(i).ToString() != ostr.str())
    {
      cout << "Bad sort order!\n";
      return 1;
    }
  }
  cout << ")\n";

  // Clean up
  iArray->Delete();
  sArray->Delete();
  dArray->Delete();
  vArray->Delete();
  fd->Delete();

  return 0;
}
