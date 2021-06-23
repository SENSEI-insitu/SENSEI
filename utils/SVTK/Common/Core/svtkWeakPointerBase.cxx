/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkWeakPointerBase.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkWeakPointerBase.h"

//----------------------------------------------------------------------------
class svtkWeakPointerBaseToObjectBaseFriendship
{
public:
  static void AddWeakPointer(svtkObjectBase* r, svtkWeakPointerBase* p);
  static void RemoveWeakPointer(svtkObjectBase* r, svtkWeakPointerBase* p) noexcept;
  static void ReplaceWeakPointer(
    svtkObjectBase* r, svtkWeakPointerBase* bad, svtkWeakPointerBase* good) noexcept;
};

//----------------------------------------------------------------------------
void svtkWeakPointerBaseToObjectBaseFriendship::AddWeakPointer(
  svtkObjectBase* r, svtkWeakPointerBase* p)
{
  if (r)
  {
    svtkWeakPointerBase** l = r->WeakPointers;
    if (l == nullptr)
    {
      // create a new list if none exists
      l = new svtkWeakPointerBase*[2];
      l[0] = p;
      l[1] = nullptr;
      r->WeakPointers = l;
    }
    else
    {
      size_t n = 0;
      while (l[n] != nullptr)
      {
        n++;
      }
      // if n+1 is a power of two, double the list size
      if ((n & (n + 1)) == 0)
      {
        svtkWeakPointerBase** t = l;
        l = new svtkWeakPointerBase*[(n + 1) * 2];
        for (size_t i = 0; i < n; i++)
        {
          l[i] = t[i];
        }
        delete[] t;
        r->WeakPointers = l;
      }
      // make sure list is null-terminated
      l[n++] = p;
      l[n] = nullptr;
    }
  }
}

//----------------------------------------------------------------------------
void svtkWeakPointerBaseToObjectBaseFriendship::RemoveWeakPointer(
  svtkObjectBase* r, svtkWeakPointerBase* p) noexcept
{
  if (r)
  {
    svtkWeakPointerBase** l = r->WeakPointers;
    if (l != nullptr)
    {
      size_t i = 0;
      while (l[i] != nullptr && l[i] != p)
      {
        i++;
      }
      while (l[i] != nullptr)
      {
        l[i] = l[i + 1];
        i++;
      }
      if (l[0] == nullptr)
      {
        delete[] l;
        r->WeakPointers = nullptr;
      }
    }
  }
}

//----------------------------------------------------------------------------
void svtkWeakPointerBaseToObjectBaseFriendship::ReplaceWeakPointer(
  svtkObjectBase* r, svtkWeakPointerBase* bad, svtkWeakPointerBase* good) noexcept
{
  if (r)
  {
    svtkWeakPointerBase** l = r->WeakPointers;
    if (l != nullptr)
    {
      for (; *l != nullptr; ++l)
      {
        if (*l == bad)
        {
          *l = good;
          break;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------
svtkWeakPointerBase::svtkWeakPointerBase(svtkObjectBase* r)
  : Object(r)
{
  svtkWeakPointerBaseToObjectBaseFriendship::AddWeakPointer(r, this);
}

//----------------------------------------------------------------------------
svtkWeakPointerBase::svtkWeakPointerBase(const svtkWeakPointerBase& r)
  : Object(r.Object)
{
  svtkWeakPointerBaseToObjectBaseFriendship::AddWeakPointer(r.Object, this);
}

//----------------------------------------------------------------------------
svtkWeakPointerBase::svtkWeakPointerBase(svtkWeakPointerBase&& r) noexcept : Object(r.Object)
{
  r.Object = nullptr;
  svtkWeakPointerBaseToObjectBaseFriendship::ReplaceWeakPointer(this->Object, &r, this);
}

//----------------------------------------------------------------------------
svtkWeakPointerBase::~svtkWeakPointerBase()
{
  svtkWeakPointerBaseToObjectBaseFriendship::RemoveWeakPointer(this->Object, this);

  this->Object = nullptr;
}

//----------------------------------------------------------------------------
svtkWeakPointerBase& svtkWeakPointerBase::operator=(svtkObjectBase* r)
{
  if (this->Object != r)
  {
    svtkWeakPointerBaseToObjectBaseFriendship::RemoveWeakPointer(this->Object, this);

    this->Object = r;

    svtkWeakPointerBaseToObjectBaseFriendship::AddWeakPointer(this->Object, this);
  }

  return *this;
}

//----------------------------------------------------------------------------
svtkWeakPointerBase& svtkWeakPointerBase::operator=(const svtkWeakPointerBase& r)
{
  if (this != &r)
  {
    if (this->Object != r.Object)
    {
      svtkWeakPointerBaseToObjectBaseFriendship::RemoveWeakPointer(this->Object, this);

      this->Object = r.Object;

      svtkWeakPointerBaseToObjectBaseFriendship::AddWeakPointer(this->Object, this);
    }
  }

  return *this;
}

//----------------------------------------------------------------------------
svtkWeakPointerBase& svtkWeakPointerBase::operator=(svtkWeakPointerBase&& r) noexcept
{
  if (this != &r)
  {
    if (this->Object != r.Object)
    {
      svtkWeakPointerBaseToObjectBaseFriendship::RemoveWeakPointer(this->Object, this);

      // WTB std::exchange
      this->Object = r.Object;
      r.Object = nullptr;

      svtkWeakPointerBaseToObjectBaseFriendship::ReplaceWeakPointer(this->Object, &r, this);
    }
  }

  return *this;
}

//----------------------------------------------------------------------------
ostream& operator<<(ostream& os, const svtkWeakPointerBase& p)
{
  // Just print the pointer value into the stream.
  return os << static_cast<void*>(p.GetPointer());
}
