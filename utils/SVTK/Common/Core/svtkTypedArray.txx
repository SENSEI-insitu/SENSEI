/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTypedArray.txx

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

#include "svtkVariantCast.h"
#include "svtkVariantCreate.h"

template <typename T>
void svtkTypedArray<T>::PrintSelf(ostream& os, svtkIndent indent)
{
  this->svtkTypedArray<T>::Superclass::PrintSelf(os, indent);
}

template <typename T>
svtkVariant svtkTypedArray<T>::GetVariantValue(const svtkArrayCoordinates& coordinates)
{
  return svtkVariantCreate<T>(this->GetValue(coordinates));
}

template <typename T>
svtkVariant svtkTypedArray<T>::GetVariantValueN(const SizeT n)
{
  return svtkVariantCreate<T>(this->GetValueN(n));
}

template <typename T>
void svtkTypedArray<T>::SetVariantValue(
  const svtkArrayCoordinates& coordinates, const svtkVariant& value)
{
  this->SetValue(coordinates, svtkVariantCast<T>(value));
}

template <typename T>
void svtkTypedArray<T>::SetVariantValueN(const SizeT n, const svtkVariant& value)
{
  this->SetValueN(n, svtkVariantCast<T>(value));
}

template <typename T>
void svtkTypedArray<T>::CopyValue(svtkArray* source, const svtkArrayCoordinates& source_coordinates,
  const svtkArrayCoordinates& target_coordinates)
{
  if (!source->IsA(this->GetClassName()))
  {
    svtkWarningMacro("source and target array data types do not match");
    return;
  }

  this->SetValue(
    target_coordinates, static_cast<svtkTypedArray<T>*>(source)->GetValue(source_coordinates));
}

template <typename T>
void svtkTypedArray<T>::CopyValue(
  svtkArray* source, const SizeT source_index, const svtkArrayCoordinates& target_coordinates)
{
  if (!source->IsA(this->GetClassName()))
  {
    svtkWarningMacro("source and target array data types do not match");
    return;
  }

  this->SetValue(
    target_coordinates, static_cast<svtkTypedArray<T>*>(source)->GetValueN(source_index));
}

template <typename T>
void svtkTypedArray<T>::CopyValue(
  svtkArray* source, const svtkArrayCoordinates& source_coordinates, const SizeT target_index)
{
  if (!source->IsA(this->GetClassName()))
  {
    svtkWarningMacro("source and target array data types do not match");
    return;
  }

  this->SetValueN(
    target_index, static_cast<svtkTypedArray<T>*>(source)->GetValue(source_coordinates));
}
