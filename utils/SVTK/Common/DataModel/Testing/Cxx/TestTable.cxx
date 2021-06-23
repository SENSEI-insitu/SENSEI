/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestTable.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------*/
#include "svtkDoubleArray.h"
#include "svtkIntArray.h"
#include "svtkMath.h"
#include "svtkStringArray.h"
#include "svtkTable.h"
#include "svtkVariantArray.h"

#include <ctime>
#include <vector>
using namespace std;

void CheckEqual(svtkTable* table, vector<vector<double> >& stdTable)
{
  // Check sizes
  if (table->GetNumberOfRows() != static_cast<svtkIdType>(stdTable[0].size()))
  {
    cout << "Number of rows is incorrect (" << table->GetNumberOfRows() << " != " << stdTable.size()
         << ")" << endl;
    exit(1);
  }
  if (table->GetNumberOfColumns() != static_cast<svtkIdType>(stdTable.size()))
  {
    cout << "Number of columns is incorrect (" << table->GetNumberOfColumns()
         << " != " << stdTable.size() << ")" << endl;
    exit(1);
  }

  // Use GetValue() and GetValueByName() to check
  for (int i = 0; i < table->GetNumberOfRows(); i++)
  {
    for (int j = 0; j < table->GetNumberOfColumns(); j++)
    {
      double tableVal = table->GetValue(i, j).ToDouble();
      double stdTableVal = stdTable[j][i];
      if (stdTableVal && tableVal != stdTableVal)
      {
        cout << "Values not equal at row " << i << " column " << j << ": ";
        cout << "(" << tableVal << " != " << stdTableVal << ")" << endl;
        exit(1);
      }
    }
  }

  // Use GetColumn() and GetColumnByName() to check
  for (int j = 0; j < table->GetNumberOfColumns(); j++)
  {
    svtkAbstractArray* arr;
    if (svtkMath::Random() < 0.5)
    {
      arr = table->GetColumn(j);
    }
    else
    {
      arr = table->GetColumnByName(table->GetColumnName(j));
    }
    for (int i = 0; i < table->GetNumberOfRows(); i++)
    {
      double val;
      if (arr->IsA("svtkVariantArray"))
      {
        val = svtkArrayDownCast<svtkVariantArray>(arr)->GetValue(i).ToDouble();
      }
      else if (arr->IsA("svtkStringArray"))
      {
        svtkVariant v(svtkArrayDownCast<svtkStringArray>(arr)->GetValue(i));
        val = v.ToDouble();
      }
      else if (arr->IsA("svtkDataArray"))
      {
        val = svtkArrayDownCast<svtkDataArray>(arr)->GetTuple1(i);
      }
      else
      {
        cout << "Unknown array type" << endl;
        exit(1);
      }
      double stdTableVal = stdTable[j][i];
      if (stdTableVal && val != stdTableVal)
      {
        cout << "Values not equal at row " << i << " column " << j << ": ";
        cout << "(" << val << " != " << stdTableVal << ")" << endl;
        exit(1);
      }
    }
  }

  // Use GetRow() to check
  for (int i = 0; i < table->GetNumberOfRows(); i++)
  {
    svtkVariantArray* arr = table->GetRow(i);
    for (int j = 0; j < table->GetNumberOfColumns(); j++)
    {
      double val = arr->GetValue(j).ToDouble();
      double stdTableVal = stdTable[j][i];
      if (stdTableVal && val != stdTableVal)
      {
        cout << "Values not equal at row " << i << " column " << j << ": ";
        cout << "(" << val << " != " << stdTableVal << ")" << endl;
        exit(1);
      }
    }
  }
}

int TestTable(int, char*[])
{
  cout << "CTEST_FULL_OUTPUT" << endl;

  long seed = time(nullptr);
  cout << "Seed: " << seed << endl;
  svtkMath::RandomSeed(seed);

  // Make a table and a parallel vector of vectors
  svtkTable* table = svtkTable::New();
  vector<vector<double> > stdTable;

  int size = 100;
  double prob = 1.0 - 1.0 / size;
  double highProb = 1.0 - 1.0 / (size * size);

  cout << "Creating columns." << endl;
  svtkIdType columnId = 0;
  bool noColumns = true;
  while (noColumns || svtkMath::Random() < prob)
  {
    noColumns = false;

    stdTable.push_back(vector<double>());

    double r = svtkMath::Random();
    svtkVariant name(columnId);
    svtkAbstractArray* arr;
    if (r < 0.25)
    {
      arr = svtkIntArray::New();
      arr->SetName((name.ToString() + " (svtkIntArray)").c_str());
    }
    else if (r < 0.5)
    {
      arr = svtkDoubleArray::New();
      arr->SetName((name.ToString() + " (svtkDoubleArray)").c_str());
    }
    else if (r < 0.75)
    {
      arr = svtkStringArray::New();
      arr->SetName((name.ToString() + " (svtkStringArray)").c_str());
    }
    else
    {
      arr = svtkVariantArray::New();
      arr->SetName((name.ToString() + " (svtkVariantArray)").c_str());
    }
    table->AddColumn(arr);
    arr->Delete();
    columnId++;
  }

  cout << "Inserting empty rows." << endl;
  bool noRows = true;
  while (noRows || svtkMath::Random() < prob)
  {
    noRows = false;
    table->InsertNextBlankRow();
    for (unsigned int i = 0; i < stdTable.size(); i++)
    {
      stdTable[i].push_back(0.0);
    }
  }

  cout << "Inserting full rows." << endl;
  while (svtkMath::Random() < prob)
  {
    svtkVariantArray* rowArray = svtkVariantArray::New();
    for (svtkIdType j = 0; j < table->GetNumberOfColumns(); j++)
    {
      rowArray->InsertNextValue(svtkVariant(j));
      stdTable[j].push_back(j);
    }
    table->InsertNextRow(rowArray);
    rowArray->Delete();
  }

  cout << "Performing all kinds of inserts." << endl;
  int id = 0;
  while (svtkMath::Random() < highProb)
  {
    svtkIdType row = static_cast<svtkIdType>(svtkMath::Random(0, table->GetNumberOfRows()));
    svtkIdType col = static_cast<svtkIdType>(svtkMath::Random(0, table->GetNumberOfColumns()));
    svtkVariant v;
    if (svtkMath::Random() < 0.25)
    {
      svtkVariant temp(id);
      v = svtkVariant(temp.ToString());
    }
    else if (svtkMath::Random() < 0.5)
    {
      v = svtkVariant(id);
    }
    else
    {
      v = svtkVariant(static_cast<double>(id));
    }

    if (svtkMath::Random() < 0.5)
    {
      table->SetValue(row, col, v);
    }
    else
    {
      table->SetValueByName(row, table->GetColumnName(col), v);
    }
    stdTable[col][row] = id;

    id++;
  }

  cout << "Removing half of the rows." << endl;
  int numRowsToRemove = table->GetNumberOfRows() / 2;
  for (int i = 0; i < numRowsToRemove; i++)
  {
    svtkIdType row = static_cast<svtkIdType>(svtkMath::Random(0, table->GetNumberOfRows()));
    cout << "Removing row " << row << " from svtkTable with " << table->GetNumberOfRows() << " rows"
         << endl;
    table->RemoveRow(row);

    cout << "Removing row " << row << " from vector< vector<double> > with " << stdTable[0].size()
         << " rows " << endl;
    for (unsigned int j = 0; j < stdTable.size(); j++)
    {
      vector<double>::iterator rowIt = stdTable[j].begin() + row;
      stdTable[j].erase(rowIt);
    }
  }

  cout << "Removing half of the columns." << endl;
  int numColsToRemove = table->GetNumberOfColumns() / 2;
  for (int i = 0; i < numColsToRemove; i++)
  {
    svtkIdType col = static_cast<svtkIdType>(svtkMath::Random(0, table->GetNumberOfColumns()));
    if (svtkMath::Random() < 0.5)
    {
      table->RemoveColumn(col);
    }
    else
    {
      table->RemoveColumnByName(table->GetColumnName(col));
    }

    vector<vector<double> >::iterator colIt = stdTable.begin() + col;
    stdTable.erase(colIt);
  }

  cout << "svtkTable size: " << table->GetNumberOfRows() << "," << table->GetNumberOfColumns()
       << endl;
  cout << "vector<vector<double> > size: " << stdTable[0].size() << "," << stdTable.size() << endl;

  cout << "Checking that table matches expected table." << endl;
  CheckEqual(table, stdTable);

  table->Delete();
  return 0;
}
