#include "svtkPixelTransfer.h"

//-----------------------------------------------------------------------------
int svtkPixelTransfer::Blit(const svtkPixelExtent& srcWholeExt, const svtkPixelExtent& srcExt,
  const svtkPixelExtent& destWholeExt, const svtkPixelExtent& destExt, int nSrcComps, int srcType,
  void* srcData, int nDestComps, int destType, void* destData)
{
  // first layer of dispatch
  int iret = 0;
  switch (srcType)
  {
    svtkTemplateMacro(iret = svtkPixelTransfer::Blit(srcWholeExt, srcExt, destWholeExt, destExt,
      nSrcComps, static_cast<SVTK_TT*>(srcData), nDestComps, destType, destData));
  }
  return iret;
}
