/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayIteratorTemplateInstantiate.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#define svtkArrayIteratorTemplateInstantiate_cxx

#include "svtkArrayIteratorTemplate.txx"

svtkInstantiateTemplateMacro(template class SVTKCOMMONCORE_EXPORT svtkArrayIteratorTemplate);
template class SVTKCOMMONCORE_EXPORT svtkArrayIteratorTemplate<svtkStdString>;
template class SVTKCOMMONCORE_EXPORT svtkArrayIteratorTemplate<svtkUnicodeString>;
template class SVTKCOMMONCORE_EXPORT svtkArrayIteratorTemplate<svtkVariant>;
