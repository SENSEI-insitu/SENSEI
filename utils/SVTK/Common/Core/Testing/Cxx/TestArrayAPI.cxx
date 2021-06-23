/*=========================================================================

  Program:   Visualization Toolkit
  Module:    ArrayAPI.cxx

-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <svtkDenseArray.h>
#include <svtkSmartPointer.h>
#include <svtkSparseArray.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

#define test_expression(expression)                                                                \
  {                                                                                                \
    if (!(expression))                                                                             \
    {                                                                                              \
      std::ostringstream buffer;                                                                   \
      buffer << "Expression failed at line " << __LINE__ << ": " << #expression;                   \
      throw std::runtime_error(buffer.str());                                                      \
    }                                                                                              \
  }

int TestArrayAPI(int svtkNotUsed(argc), char* svtkNotUsed(argv)[])
{
  try
  {
    svtkSmartPointer<svtkArray> array;

    // Test to see that we can create every supported combination of storage- and value-type.
    std::vector<int> storage_types;
    storage_types.push_back(svtkArray::DENSE);
    storage_types.push_back(svtkArray::SPARSE);

    std::vector<int> value_types;
    value_types.push_back(SVTK_CHAR);
    value_types.push_back(SVTK_UNSIGNED_CHAR);
    value_types.push_back(SVTK_SHORT);
    value_types.push_back(SVTK_UNSIGNED_SHORT);
    value_types.push_back(SVTK_INT);
    value_types.push_back(SVTK_UNSIGNED_INT);
    value_types.push_back(SVTK_LONG);
    value_types.push_back(SVTK_UNSIGNED_LONG);
    value_types.push_back(SVTK_DOUBLE);
    value_types.push_back(SVTK_ID_TYPE);
    value_types.push_back(SVTK_STRING);
    value_types.push_back(SVTK_VARIANT);

    std::vector<svtkVariant> sample_values;
    sample_values.push_back(static_cast<char>(1));
    sample_values.push_back(static_cast<unsigned char>(2));
    sample_values.push_back(static_cast<short>(3));
    sample_values.push_back(static_cast<unsigned short>(4));
    sample_values.push_back(static_cast<int>(5));
    sample_values.push_back(static_cast<unsigned int>(6));
    sample_values.push_back(static_cast<long>(7));
    sample_values.push_back(static_cast<unsigned long>(8));
    sample_values.push_back(static_cast<double>(9.0));
    sample_values.push_back(static_cast<svtkIdType>(10));
    sample_values.push_back(svtkStdString("11"));
    sample_values.push_back(svtkVariant(12.0));

    for (std::vector<int>::const_iterator storage_type = storage_types.begin();
         storage_type != storage_types.end(); ++storage_type)
    {
      for (size_t value_type = 0; value_type != value_types.size(); ++value_type)
      {
        cerr << "creating array with storage type " << *storage_type << " and value type "
             << svtkImageScalarTypeNameMacro(value_types[value_type]) << endl;

        array.TakeReference(svtkArray::CreateArray(*storage_type, value_types[value_type]));
        test_expression(array);

        test_expression(array->GetName().empty());
        array->SetName("foo");
        test_expression(array->GetName() == "foo");

        array->Resize(10);
        array->SetVariantValue(5, sample_values[value_type]);
        test_expression(array->GetVariantValue(5).IsValid());
        test_expression(array->GetVariantValue(5) == sample_values[value_type]);
      }
    }

    // Do some spot-checking to see that the actual type matches what we expect ...
    array.TakeReference(svtkArray::CreateArray(svtkArray::DENSE, SVTK_DOUBLE));
    test_expression(svtkDenseArray<double>::SafeDownCast(array));

    array.TakeReference(svtkArray::CreateArray(svtkArray::SPARSE, SVTK_STRING));
    test_expression(svtkSparseArray<svtkStdString>::SafeDownCast(array));

    return 0;
  }
  catch (std::exception& e)
  {
    cerr << e.what() << endl;
    return 1;
  }
}
