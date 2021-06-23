/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSMPToolsInternal.h.in

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include <algorithm> //for std::sort()

namespace svtk
{
namespace detail
{
namespace smp
{
template <typename FunctorInternal>
void svtkSMPTools_Impl_For(
  svtkIdType first, svtkIdType last, svtkIdType grain,
  FunctorInternal& fi)
{
  svtkIdType n = last - first;
  if (!n)
  {
    return;
  }

  if (grain == 0 || grain >= n)
  {
    fi.Execute(first, last);
  }
  else
  {
    svtkIdType b = first;
    while (b < last)
    {
      svtkIdType e = b + grain;
      if (e > last)
      {
        e = last;
      }
      fi.Execute(b, e);
      b = e;
    }
  }
}

//--------------------------------------------------------------------------------
template<typename RandomAccessIterator>
void svtkSMPTools_Impl_Sort(RandomAccessIterator begin,
                                  RandomAccessIterator end)
{
  std::sort(begin, end);
}

//--------------------------------------------------------------------------------
template<typename RandomAccessIterator, typename Compare>
void svtkSMPTools_Impl_Sort(RandomAccessIterator begin,
                                  RandomAccessIterator end,
                                  Compare comp)
{
  std::sort(begin, end, comp);
}

}//namespace smp
}//namespace detail
}//namespace svtk
