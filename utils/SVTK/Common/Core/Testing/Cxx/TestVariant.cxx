/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestVariant.cxx

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

#include "svtkVariant.h"

int TestVariant(int, char*[])
{
  double value = 123456;
  const char* strValue = "123456";
  int errors = 0;
  int type[] = { SVTK_INT, SVTK_UNSIGNED_INT, SVTK_TYPE_INT64, SVTK_TYPE_UINT64, SVTK_FLOAT, SVTK_DOUBLE,
    SVTK_STRING };
  int numTypes = 7;

  for (int i = 0; i < numTypes; i++)
  {
    svtkVariant v;
    switch (type[i])
    {
      case SVTK_INT:
        v = static_cast<int>(value);
        break;
      case SVTK_UNSIGNED_INT:
        v = static_cast<unsigned int>(value);
        break;
      case SVTK_TYPE_INT64:
        v = static_cast<svtkTypeInt64>(value);
        break;
      case SVTK_TYPE_UINT64:
        v = static_cast<svtkTypeUInt64>(value);
        break;
      case SVTK_FLOAT:
        v = static_cast<float>(value);
        break;
      case SVTK_DOUBLE:
        v = static_cast<double>(value);
        break;
      case SVTK_STRING:
        v = strValue;
        break;
      default:
        continue;
    }
    cerr << "v = " << v << " (" << svtkImageScalarTypeNameMacro(type[i]) << ")\n";
    for (int j = 0; j < numTypes; j++)
    {
      svtkStdString str;
      switch (type[j])
      {
        case SVTK_INT:
        {
          int conv = v.ToInt();
          if (conv != static_cast<int>(value))
          {
            cerr << "conversion invalid (" << svtkImageScalarTypeNameMacro(type[i]) << " " << conv
                 << " != " << svtkImageScalarTypeNameMacro(type[j]) << " " << static_cast<int>(value)
                 << ")" << endl;
            errors++;
          }
          break;
        }
        case SVTK_UNSIGNED_INT:
        {
          unsigned int conv = v.ToUnsignedInt();
          if (conv != static_cast<unsigned int>(value))
          {
            cerr << "conversion invalid (" << svtkImageScalarTypeNameMacro(type[i]) << " " << conv
                 << " != " << svtkImageScalarTypeNameMacro(type[j]) << " "
                 << static_cast<unsigned int>(value) << ")" << endl;
            errors++;
          }
          break;
        }
        case SVTK_TYPE_INT64:
        {
          svtkTypeInt64 conv = v.ToTypeInt64();
          if (conv != static_cast<svtkTypeInt64>(value))
          {
            cerr << "conversion invalid (" << svtkImageScalarTypeNameMacro(type[i]) << " " << conv
                 << " != " << svtkImageScalarTypeNameMacro(type[j]) << " "
                 << static_cast<svtkTypeInt64>(value) << ")" << endl;
            errors++;
          }
          break;
        }
        case SVTK_TYPE_UINT64:
        {
          svtkTypeUInt64 conv = v.ToTypeUInt64();
          if (conv != static_cast<svtkTypeUInt64>(value))
          {
            cerr << "conversion invalid (" << svtkImageScalarTypeNameMacro(type[i]) << " " << conv
                 << " != " << svtkImageScalarTypeNameMacro(type[j]) << " "
                 << static_cast<svtkTypeUInt64>(value) << ")" << endl;
            errors++;
          }
          break;
        }
        case SVTK_FLOAT:
        {
          float conv = v.ToFloat();
          if (conv != static_cast<float>(value))
          {
            cerr << "conversion invalid (" << svtkImageScalarTypeNameMacro(type[i]) << " " << conv
                 << " != " << svtkImageScalarTypeNameMacro(type[j]) << " "
                 << static_cast<float>(value) << ")" << endl;
            errors++;
          }
          break;
        }
        case SVTK_DOUBLE:
        {
          double conv = v.ToDouble();
          if (conv != static_cast<double>(value))
          {
            cerr << "conversion invalid (" << svtkImageScalarTypeNameMacro(type[i]) << " " << conv
                 << " != " << svtkImageScalarTypeNameMacro(type[j]) << " "
                 << static_cast<double>(value) << ")" << endl;
            errors++;
          }
          break;
        }
        case SVTK_STRING:
        {
          svtkStdString conv = v.ToString();
          if (conv != svtkStdString(strValue))
          {
            cerr << "conversion invalid (" << svtkImageScalarTypeNameMacro(type[i]) << " " << conv
                 << " != " << svtkImageScalarTypeNameMacro(type[j]) << " " << strValue << ")"
                 << endl;
            errors++;
          }
          break;
        }
        default:
          continue;
      }
    }
  }

  svtkVariant flt(0.583f);
  svtkVariant dbl(0.583);
  svtkVariant str("0.583");
  if (!(flt == dbl) || flt < dbl || flt > dbl || !(str == dbl) || str < dbl || str > dbl ||
    !(flt == str) || flt < str || flt > str)
  {
    cerr << "Comparison of dissimilar-precision floats failed.\n";
    errors++;
  }

  return errors;
}
