/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkScaledSOADataArrayTemplateInstantiate.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// This file generates instantiations of svtkScaledSOADataArrayTemplate for the
// common data types. For AoS arrays, this is done in the more derived classes
// (e.g. svtkFloatArray.cxx.o contains the instantiation of
// svtkAOSDataArrayTemplate<float>), but since these aren't derived from in SVTK
// (yet), we instantiate them here.

#define SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATING
#include "svtkScaledSOADataArrayTemplate.txx"

SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(char);
SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(double);
SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(float);
SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(int);
SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(long);
SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(long long);
SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(short);
SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(signed char);
SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(unsigned char);
SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(unsigned int);
SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(unsigned long);
SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(unsigned long long);
SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(unsigned short);
