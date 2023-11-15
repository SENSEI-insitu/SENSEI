/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPixelTransfer.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPixelTransfer
 * pixel extents
 *
 *
 * Class to handle non-contiguous data transfers of data described
 * by pixel extents within a process. For transferring data between
 * processes see svtkPPixelTransfer.
 *
 * @sa
 * svtkPixelExtent svtkPPixelTransfer
 */

#ifndef svtkPixelTransfer_h
#define svtkPixelTransfer_h

#include "svtkCommonDataModelModule.h" // for export
#include "svtkPixelExtent.h"           // for pixel extent
#include "svtkSetGet.h"                // for macros
#include <cstring>                    // for memcpy

class SVTKCOMMONDATAMODEL_EXPORT svtkPixelTransfer
{
public:
  svtkPixelTransfer() {}

  /**
   * for memory to memory transfers. Convenience api for working
   * with svtk type enum rather than c-data types and simple extents.
   */
  static int Blit(const svtkPixelExtent& ext, int nComps, int srcType, void* srcData, int destType,
    void* destData);

  /**
   * for memory to memory transfers. Convenience api for working
   * with svtk type enum rather than c-data types.
   */
  static int Blit(const svtkPixelExtent& srcWhole, const svtkPixelExtent& srcSubset,
    const svtkPixelExtent& destWhole, const svtkPixelExtent& destSubset, int nSrcComps, int srcType,
    void* srcData, int nDestComps, int destType, void* destData);

  /**
   * for local memory to memory transfers
   */
  template <typename SOURCE_TYPE, typename DEST_TYPE>
  static int Blit(const svtkPixelExtent& srcWhole, const svtkPixelExtent& srcSubset,
    const svtkPixelExtent& destWhole, const svtkPixelExtent& destSubset, int nSrcComps,
    SOURCE_TYPE* srcData, int nDestComps, DEST_TYPE* destData);

private:
  // distpatch helper for svtk data type enum
  template <typename SOURCE_TYPE>
  static int Blit(const svtkPixelExtent& srcWhole, const svtkPixelExtent& srcSubset,
    const svtkPixelExtent& destWhole, const svtkPixelExtent& destSubset, int nSrcComps,
    SOURCE_TYPE* srcData, int nDestComps, int destType, void* destData);
};

//-----------------------------------------------------------------------------
inline int svtkPixelTransfer::Blit(
  const svtkPixelExtent& ext, int nComps, int srcType, void* srcData, int destType, void* destData)
{
  return svtkPixelTransfer::Blit(
    ext, ext, ext, ext, nComps, srcType, srcData, nComps, destType, destData);
}

//-----------------------------------------------------------------------------
template <typename SOURCE_TYPE>
int svtkPixelTransfer::Blit(const svtkPixelExtent& srcWholeExt, const svtkPixelExtent& srcExt,
  const svtkPixelExtent& destWholeExt, const svtkPixelExtent& destExt, int nSrcComps,
  SOURCE_TYPE* srcData, int nDestComps, int destType, void* destData)
{
  // second layer of dispatch
  int iret = 0;
  switch (destType)
  {
    svtkTemplateMacro(iret = svtkPixelTransfer::Blit(srcWholeExt, srcExt, destWholeExt, destExt,
      nSrcComps, srcData, nDestComps, (SVTK_TT*)destData););
  }
  return iret;
}

//-----------------------------------------------------------------------------
template <typename SOURCE_TYPE, typename DEST_TYPE>
int svtkPixelTransfer::Blit(const svtkPixelExtent& srcWholeExt, const svtkPixelExtent& srcSubset,
  const svtkPixelExtent& destWholeExt, const svtkPixelExtent& destSubset, int nSrcComps,
  SOURCE_TYPE* srcData, int nDestComps, DEST_TYPE* destData)
{
  if ((srcData == nullptr) || (destData == nullptr))
  {
    return -1;
  }
  if ((srcWholeExt == srcSubset) && (destWholeExt == destSubset) && (nSrcComps == nDestComps))
  {
    // buffers are contiguous
    size_t n = srcWholeExt.Size() * nSrcComps;
    for (size_t i = 0; i < n; ++i)
    {
      destData[i] = static_cast<DEST_TYPE>(srcData[i]);
    }
  }
  else
  {
    // buffers are not contiguous
    int tmp[2];

    // get the dimensions of the arrays
    srcWholeExt.Size(tmp);
    int swnx = tmp[0];

    destWholeExt.Size(tmp);
    int dwnx = tmp[0];

    // move from logical extent to memory extent
    svtkPixelExtent srcExt(srcSubset);
    srcExt.Shift(srcWholeExt);

    svtkPixelExtent destExt(destSubset);
    destExt.Shift(destWholeExt);

    // get size of sub-set to copy (it's the same in src and dest)
    int nxny[2];
    srcExt.Size(nxny);

    // use smaller ncomps for loop index to avoid reading/writing
    // invalid mem
    int nCopyComps = nSrcComps < nDestComps ? nSrcComps : nDestComps;

    for (int j = 0; j < nxny[1]; ++j)
    {
      int sjj = swnx * (srcExt[2] + j) + srcExt[0];
      int djj = dwnx * (destExt[2] + j) + destExt[0];
      for (int i = 0; i < nxny[0]; ++i)
      {
        int sidx = nSrcComps * (sjj + i);
        int didx = nDestComps * (djj + i);
        // copy values from source
        for (int p = 0; p < nCopyComps; ++p)
        {
          destData[didx + p] = static_cast<DEST_TYPE>(srcData[sidx + p]);
        }
        // ensure all dest comps are initialized
        for (int p = nCopyComps; p < nDestComps; ++p)
        {
          destData[didx + p] = static_cast<DEST_TYPE>(0);
        }
      }
    }
  }
  return 0;
}

#endif
// SVTK-HeaderTest-Exclude: svtkPixelTransfer.h
