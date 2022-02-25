/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSOADataArrayTemplateInstantiate.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// This file generates instantiations of svtkSOADataArrayTemplate for the
// common data types. For AoS arrays, this is done in the more derived classes
// (e.g. svtkFloatArray.cxx.o contains the instantiation of
// svtkAOSDataArrayTemplate<float>), but since these aren't derived from in SVTK
// (yet), we instantiate them here.

#define SVTK_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATING
#include "svtkSOADataArrayTemplate.txx"

SVTK_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(char);
SVTK_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(double);
SVTK_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(float);
SVTK_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(int);
SVTK_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(long);
SVTK_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(long long);
SVTK_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(short);
SVTK_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(signed char);
SVTK_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(unsigned char);
SVTK_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(unsigned int);
SVTK_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(unsigned long);
SVTK_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(unsigned long long);
SVTK_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(unsigned short);
