/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkScalarsToColors.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkScalarsToColors.h"

#include "svtkAbstractArray.h"
#include "svtkCharArray.h"
#include "svtkObjectFactory.h"
#include "svtkStringArray.h"
#include "svtkTemplateAliasMacro.h"
#include "svtkUnsignedCharArray.h"
#include "svtkVariantArray.h"

#include <map>

#include <cmath>

// A helper map for quick lookups of annotated values.
class svtkScalarsToColors::svtkInternalAnnotatedValueMap : public std::map<svtkVariant, svtkIdType>
{
};

svtkStandardNewMacro(svtkScalarsToColors);

//----------------------------------------------------------------------------
svtkScalarsToColors::svtkScalarsToColors()
{
  this->Alpha = 1.0;
  this->VectorComponent = 0;
  this->VectorSize = -1;
  this->VectorMode = svtkScalarsToColors::COMPONENT;

  // only used in this class, not used in subclasses
  this->InputRange[0] = 0.0;
  this->InputRange[1] = 255.0;

  // Annotated values, their annotations, and whether colors
  // should be indexed by annotated value.
  this->AnnotatedValues = nullptr;
  this->Annotations = nullptr;
  this->AnnotatedValueMap = new svtkInternalAnnotatedValueMap;
  this->IndexedLookup = 0;

  // obsolete, kept for backwards compatibility
  this->UseMagnitude = 0;
}

//----------------------------------------------------------------------------
svtkScalarsToColors::~svtkScalarsToColors()
{
  if (this->AnnotatedValues)
  {
    this->AnnotatedValues->UnRegister(this);
  }
  if (this->Annotations)
  {
    this->Annotations->UnRegister(this);
  }
  delete this->AnnotatedValueMap;
}

//----------------------------------------------------------------------------
// Description:
// Return true if all of the values defining the mapping have an opacity
// equal to 1. Default implementation return true.
int svtkScalarsToColors::IsOpaque()
{
  return 1;
}

//----------------------------------------------------------------------------
// Description:
// Return true if all of the values defining the mapping have an opacity
// equal to 1. Default implementation return true.
int svtkScalarsToColors::IsOpaque(svtkAbstractArray* scalars, int colorMode, int /*component*/)
{
  if (!scalars)
  {
    return this->IsOpaque();
  }

  int numberOfComponents = scalars->GetNumberOfComponents();

  svtkDataArray* dataArray = svtkArrayDownCast<svtkDataArray>(scalars);

  // map scalars through lookup table only if needed
  if ((colorMode == SVTK_COLOR_MODE_DEFAULT &&
        svtkArrayDownCast<svtkUnsignedCharArray>(dataArray) != nullptr) ||
    (colorMode == SVTK_COLOR_MODE_DIRECT_SCALARS && dataArray))
  {
    // we will be using the scalars directly, so look at the number of
    // components and the range
    if (numberOfComponents == 3 || numberOfComponents == 1)
    {
      return (this->Alpha >= 1.0 ? 1 : 0);
    }
    // otherwise look at the range of the alpha channel
    unsigned char opacity = 0;
    switch (scalars->GetDataType())
    {
      svtkTemplateMacro(svtkScalarsToColors::ColorToUChar(
        static_cast<SVTK_TT>(dataArray->GetRange(numberOfComponents - 1)[0]), &opacity));
    }
    return ((opacity == 255) ? 1 : 0);
  }

  return 1;
}

//----------------------------------------------------------------------------
void svtkScalarsToColors::SetVectorModeToComponent()
{
  this->SetVectorMode(svtkScalarsToColors::COMPONENT);
}

//----------------------------------------------------------------------------
void svtkScalarsToColors::SetVectorModeToMagnitude()
{
  this->SetVectorMode(svtkScalarsToColors::MAGNITUDE);
}

//----------------------------------------------------------------------------
void svtkScalarsToColors::SetVectorModeToRGBColors()
{
  this->SetVectorMode(svtkScalarsToColors::RGBCOLORS);
}

//----------------------------------------------------------------------------
// do not use SetMacro() because we do not want the table to rebuild.
void svtkScalarsToColors::SetAlpha(double alpha)
{
  this->Alpha = (alpha < 0.0 ? 0.0 : (alpha > 1.0 ? 1.0 : alpha));
}

//----------------------------------------------------------------------------
void svtkScalarsToColors::SetRange(double minval, double maxval)
{
  if (this->InputRange[0] != minval || this->InputRange[1] != maxval)
  {
    this->InputRange[0] = minval;
    this->InputRange[1] = maxval;
    this->Modified();
  }
}

//----------------------------------------------------------------------------
double* svtkScalarsToColors::GetRange()
{
  return this->InputRange;
}

//----------------------------------------------------------------------------
svtkIdType svtkScalarsToColors::GetNumberOfAvailableColors()
{
  // return total possible RGB colors
  return 256 * 256 * 256;
}

//----------------------------------------------------------------------------
void svtkScalarsToColors::DeepCopy(svtkScalarsToColors* obj)
{
  if (obj)
  {
    this->Alpha = obj->Alpha;
    this->VectorMode = obj->VectorMode;
    this->VectorComponent = obj->VectorComponent;
    this->VectorSize = obj->VectorSize;
    this->InputRange[0] = obj->InputRange[0];
    this->InputRange[1] = obj->InputRange[1];
    this->IndexedLookup = obj->IndexedLookup;
    if (obj->AnnotatedValues && obj->Annotations)
    {
      svtkAbstractArray* annValues =
        svtkAbstractArray::CreateArray(obj->AnnotatedValues->GetDataType());
      svtkStringArray* annotations = svtkStringArray::New();
      annValues->DeepCopy(obj->AnnotatedValues);
      annotations->DeepCopy(obj->Annotations);
      this->SetAnnotations(annValues, annotations);
      annValues->Delete();
      annotations->Delete();
    }
    else
    {
      this->SetAnnotations(nullptr, nullptr);
    }
  }
}

//----------------------------------------------------------------------------
inline void svtkScalarsToColorsComputeShiftScale(
  svtkScalarsToColors* self, double& shift, double& scale)
{
  static const double minscale = -1e17;
  static const double maxscale = 1e17;

  const double* range = self->GetRange();
  shift = -range[0];
  scale = range[1] - range[0];
  if (scale * scale > 1e-30)
  {
    scale = 1.0 / scale;
  }
  else
  {
    scale = (scale < 0.0 ? minscale : maxscale);
  }
}

//----------------------------------------------------------------------------
void svtkScalarsToColors::GetColor(double v, double rgb[3])
{
  static double minval = 0.0;
  static double maxval = 1.0;

  double shift, scale;
  svtkScalarsToColorsComputeShiftScale(this, shift, scale);

  double val = (v + shift) * scale;
  val = (val > minval ? val : minval);
  val = (val < maxval ? val : maxval);

  rgb[0] = val;
  rgb[1] = val;
  rgb[2] = val;
}

//----------------------------------------------------------------------------
double svtkScalarsToColors::GetOpacity(double svtkNotUsed(v))
{
  return 1.0;
}

//----------------------------------------------------------------------------
const unsigned char* svtkScalarsToColors::MapValue(double v)
{
  double rgb[3];

  this->GetColor(v, rgb);
  double alpha = this->GetOpacity(v);

  this->RGBABytes[0] = ColorToUChar(rgb[0]);
  this->RGBABytes[1] = ColorToUChar(rgb[1]);
  this->RGBABytes[2] = ColorToUChar(rgb[2]);
  this->RGBABytes[3] = ColorToUChar(alpha);

  return this->RGBABytes;
}

//----------------------------------------------------------------------------
svtkUnsignedCharArray* svtkScalarsToColors::MapScalars(
  svtkDataArray* scalars, int colorMode, int component, int outputFormat)
{
  return this->MapScalars(
    static_cast<svtkAbstractArray*>(scalars), colorMode, component, outputFormat);
}

//----------------------------------------------------------------------------
svtkUnsignedCharArray* svtkScalarsToColors::MapScalars(
  svtkAbstractArray* scalars, int colorMode, int component, int outputFormat)
{
  int numberOfComponents = scalars->GetNumberOfComponents();
  svtkUnsignedCharArray* newColors;

  svtkDataArray* dataArray = svtkArrayDownCast<svtkDataArray>(scalars);

  // map scalars through lookup table only if needed
  if ((colorMode == SVTK_COLOR_MODE_DEFAULT &&
        svtkArrayDownCast<svtkUnsignedCharArray>(dataArray) != nullptr) ||
    (colorMode == SVTK_COLOR_MODE_DIRECT_SCALARS && dataArray))
  {
    newColors = this->ConvertToRGBA(
      dataArray, scalars->GetNumberOfComponents(), dataArray->GetNumberOfTuples());
  }
  else
  {
    newColors = svtkUnsignedCharArray::New();
    newColors->SetNumberOfComponents(outputFormat);
    newColors->SetNumberOfTuples(scalars->GetNumberOfTuples());

    // If mapper did not specify a component, use the VectorMode
    if (component < 0 && numberOfComponents > 1)
    {
      this->MapVectorsThroughTable(scalars->GetVoidPointer(0), newColors->GetPointer(0),
        scalars->GetDataType(), scalars->GetNumberOfTuples(), scalars->GetNumberOfComponents(),
        outputFormat);
    }
    else
    {
      if (component < 0)
      {
        component = 0;
      }
      if (component >= numberOfComponents)
      {
        component = numberOfComponents - 1;
      }

      // Map the scalars to colors
      this->MapScalarsThroughTable(scalars->GetVoidPointer(component), newColors->GetPointer(0),
        scalars->GetDataType(), scalars->GetNumberOfTuples(), scalars->GetNumberOfComponents(),
        outputFormat);
    }
  }

  return newColors;
}

//----------------------------------------------------------------------------
// Map a set of vector values through the table
void svtkScalarsToColors::MapVectorsThroughTable(void* input, unsigned char* output, int scalarType,
  int numValues, int inComponents, int outputFormat, int vectorComponent, int vectorSize)
{
  if (outputFormat < SVTK_LUMINANCE || outputFormat > SVTK_RGBA)
  {
    svtkErrorMacro(<< "MapVectorsThroughTable: unrecognized color format");
    return;
  }

  int vectorMode = this->GetVectorMode();
  if (vectorMode == svtkScalarsToColors::COMPONENT)
  {
    // make sure vectorComponent is within allowed range
    if (vectorComponent == -1)
    {
      // if set to -1, use default value provided by table
      vectorComponent = this->GetVectorComponent();
    }
    if (vectorComponent < 0)
    {
      vectorComponent = 0;
    }
    if (vectorComponent >= inComponents)
    {
      vectorComponent = inComponents - 1;
    }
  }
  else
  {
    // make sure vectorSize is within allowed range
    if (vectorSize == -1)
    {
      // if set to -1, use default value provided by table
      vectorSize = this->GetVectorSize();
    }
    if (vectorSize <= 0)
    {
      vectorComponent = 0;
      vectorSize = inComponents;
    }
    else
    {
      if (vectorComponent < 0)
      {
        vectorComponent = 0;
      }
      if (vectorComponent >= inComponents)
      {
        vectorComponent = inComponents - 1;
      }
      if (vectorComponent + vectorSize > inComponents)
      {
        vectorSize = inComponents - vectorComponent;
      }
    }

    if (vectorMode == svtkScalarsToColors::MAGNITUDE && (inComponents == 1 || vectorSize == 1))
    {
      vectorMode = svtkScalarsToColors::COMPONENT;
    }
  }

  // increment input pointer to the first component to map
  if (vectorComponent > 0)
  {
    int scalarSize = svtkDataArray::GetDataTypeSize(scalarType);
    input = static_cast<unsigned char*>(input) + vectorComponent * scalarSize;
  }

  // map according to the current vector mode
  switch (vectorMode)
  {
    case svtkScalarsToColors::COMPONENT:
    {
      this->MapScalarsThroughTable(
        input, output, scalarType, numValues, inComponents, outputFormat);
    }
    break;

    case svtkScalarsToColors::MAGNITUDE:
    {
      // convert to magnitude in blocks of 300 values
      int inInc = svtkDataArray::GetDataTypeSize(scalarType) * inComponents;
      double magValues[300];
      int blockSize = 300;
      int numBlocks = (numValues + blockSize - 1) / blockSize;
      int lastBlockSize = numValues - blockSize * (numBlocks - 1);

      for (int i = 0; i < numBlocks; i++)
      {
        int numMagValues = ((i < numBlocks - 1) ? blockSize : lastBlockSize);
        this->MapVectorsToMagnitude(
          input, magValues, scalarType, numMagValues, inComponents, vectorSize);
        this->MapScalarsThroughTable(magValues, output, SVTK_DOUBLE, numMagValues, 1, outputFormat);
        input = static_cast<char*>(input) + numMagValues * inInc;
        output += numMagValues * outputFormat;
      }
    }
    break;

    case svtkScalarsToColors::RGBCOLORS:
    {
      this->MapColorsToColors(
        input, output, scalarType, numValues, inComponents, vectorSize, outputFormat);
    }
    break;
  }
}

//----------------------------------------------------------------------------
// Map a set of scalar values through the table
void svtkScalarsToColors::MapScalarsThroughTable(
  svtkDataArray* scalars, unsigned char* output, int outputFormat)
{
  if (outputFormat < SVTK_LUMINANCE || outputFormat > SVTK_RGBA)
  {
    svtkErrorMacro(<< "MapScalarsThroughTable: unrecognized color format");
    return;
  }

  this->MapScalarsThroughTable(scalars->GetVoidPointer(0), output, scalars->GetDataType(),
    scalars->GetNumberOfTuples(), scalars->GetNumberOfComponents(), outputFormat);
}

//----------------------------------------------------------------------------
// Color type converters in anonymous namespace
namespace
{

#define svtkScalarsToColorsLuminance(r, g, b) ((r)*0.30 + (g)*0.59 + (b)*0.11)

//----------------------------------------------------------------------------
void svtkScalarsToColorsLuminanceToLuminance(
  const unsigned char* inPtr, unsigned char* outPtr, svtkIdType count, int numComponents)
{
  do
  {
    *outPtr++ = *inPtr;
    inPtr += numComponents;
  } while (--count);
}

//----------------------------------------------------------------------------
void svtkScalarsToColorsLuminanceToRGB(
  const unsigned char* inPtr, unsigned char* outPtr, svtkIdType count, int numComponents)
{
  do
  {
    unsigned char l = *inPtr;
    outPtr[0] = l;
    outPtr[1] = l;
    outPtr[2] = l;
    inPtr += numComponents;
    outPtr += 3;
  } while (--count);
}

//----------------------------------------------------------------------------
void svtkScalarsToColorsRGBToLuminance(
  const unsigned char* inPtr, unsigned char* outPtr, svtkIdType count, int numComponents)
{
  do
  {
    unsigned char r = inPtr[0];
    unsigned char g = inPtr[1];
    unsigned char b = inPtr[2];
    *outPtr++ = static_cast<unsigned char>(svtkScalarsToColorsLuminance(r, g, b) + 0.5);
    inPtr += numComponents;
  } while (--count);
}

//----------------------------------------------------------------------------
void svtkScalarsToColorsRGBToRGB(
  const unsigned char* inPtr, unsigned char* outPtr, svtkIdType count, int numComponents)
{
  do
  {
    outPtr[0] = inPtr[0];
    outPtr[1] = inPtr[1];
    outPtr[2] = inPtr[2];
    inPtr += numComponents;
    outPtr += 3;
  } while (--count);
}

//----------------------------------------------------------------------------
void svtkScalarsToColorsLuminanceToLuminanceAlpha(const unsigned char* inPtr, unsigned char* outPtr,
  svtkIdType count, int numComponents, double alpha)
{
  unsigned char a = svtkScalarsToColors::ColorToUChar(alpha);

  do
  {
    outPtr[0] = inPtr[0];
    outPtr[1] = a;
    inPtr += numComponents;
    outPtr += 2;
  } while (--count);
}

//----------------------------------------------------------------------------
template <typename T>
void svtkScalarsToColorsLuminanceToRGBA(
  const T* inPtr, unsigned char* outPtr, svtkIdType count, int numComponents, double alpha)
{
  unsigned char a = svtkScalarsToColors::ColorToUChar(alpha);

  do
  {
    unsigned char l = svtkScalarsToColors::ColorToUChar(inPtr[0]);
    outPtr[0] = l;
    outPtr[1] = l;
    outPtr[2] = l;
    outPtr[3] = a;
    inPtr += numComponents;
    outPtr += 4;
  } while (--count);
}

//----------------------------------------------------------------------------
void svtkScalarsToColorsRGBToLuminanceAlpha(const unsigned char* inPtr, unsigned char* outPtr,
  svtkIdType count, int numComponents, double alpha)
{
  unsigned char a = svtkScalarsToColors::ColorToUChar(alpha);

  do
  {
    unsigned char r = inPtr[0];
    unsigned char g = inPtr[1];
    unsigned char b = inPtr[2];
    outPtr[0] = static_cast<unsigned char>(svtkScalarsToColorsLuminance(r, g, b) + 0.5);
    outPtr[1] = a;
    inPtr += numComponents;
    outPtr += 2;
  } while (--count);
}

//----------------------------------------------------------------------------
template <typename T>
void svtkScalarsToColorsRGBToRGBA(
  const T* inPtr, unsigned char* outPtr, svtkIdType count, int numComponents, double alpha)
{
  unsigned char a = svtkScalarsToColors::ColorToUChar(alpha);

  do
  {
    outPtr[0] = svtkScalarsToColors::ColorToUChar(inPtr[0]);
    outPtr[1] = svtkScalarsToColors::ColorToUChar(inPtr[1]);
    outPtr[2] = svtkScalarsToColors::ColorToUChar(inPtr[2]);
    outPtr[3] = a;
    inPtr += numComponents;
    outPtr += 4;
  } while (--count);
}

//----------------------------------------------------------------------------
void svtkScalarsToColorsLuminanceAlphaToLuminanceAlpha(const unsigned char* inPtr,
  unsigned char* outPtr, svtkIdType count, int numComponents, double alpha)
{
  if (alpha >= 1)
  {
    do
    {
      outPtr[0] = inPtr[0];
      outPtr[1] = inPtr[1];
      inPtr += numComponents;
      outPtr += 2;
    } while (--count);
  }
  else
  {
    do
    {
      outPtr[0] = inPtr[0];
      outPtr[1] = static_cast<unsigned char>(inPtr[1] * alpha + 0.5);
      inPtr += numComponents;
      outPtr += 2;
    } while (--count);
  }
}

//----------------------------------------------------------------------------
template <typename T>
void svtkScalarsToColorsLuminanceAlphaToRGBA(
  const T* inPtr, unsigned char* outPtr, svtkIdType count, int numComponents, double alpha)
{
  if (alpha >= 1)
  {
    do
    {
      unsigned char l = svtkScalarsToColors::ColorToUChar(inPtr[0]);
      unsigned char a = svtkScalarsToColors::ColorToUChar(inPtr[1]);
      outPtr[0] = l;
      outPtr[1] = l;
      outPtr[2] = l;
      outPtr[3] = a;
      inPtr += numComponents;
      outPtr += 4;
    } while (--count);
  }
  else
  {
    do
    {
      unsigned char l = svtkScalarsToColors::ColorToUChar(inPtr[0]);
      unsigned char a = svtkScalarsToColors::ColorToUChar(inPtr[1]);
      outPtr[0] = l;
      outPtr[1] = l;
      outPtr[2] = l;
      outPtr[3] = static_cast<unsigned char>(a * alpha + 0.5);
      inPtr += numComponents;
      outPtr += 4;
    } while (--count);
  }
}

//----------------------------------------------------------------------------
void svtkScalarsToColorsRGBAToLuminanceAlpha(const unsigned char* inPtr, unsigned char* outPtr,
  svtkIdType count, int numComponents, double alpha)
{
  do
  {
    unsigned char r = inPtr[0];
    unsigned char g = inPtr[1];
    unsigned char b = inPtr[2];
    unsigned char a = inPtr[3];
    outPtr[0] = static_cast<unsigned char>(svtkScalarsToColorsLuminance(r, g, b) + 0.5);
    outPtr[1] = static_cast<unsigned char>(a * alpha + 0.5);
    inPtr += numComponents;
    outPtr += 2;
  } while (--count);
}

//----------------------------------------------------------------------------
template <typename T>
void svtkScalarsToColorsRGBAToRGBA(
  const T* inPtr, unsigned char* outPtr, svtkIdType count, int numComponents, double alpha)
{
  if (alpha >= 1)
  {
    do
    {
      outPtr[0] = svtkScalarsToColors::ColorToUChar(inPtr[0]);
      outPtr[1] = svtkScalarsToColors::ColorToUChar(inPtr[1]);
      outPtr[2] = svtkScalarsToColors::ColorToUChar(inPtr[2]);
      outPtr[3] = svtkScalarsToColors::ColorToUChar(inPtr[3]);
      inPtr += numComponents;
      outPtr += 4;
    } while (--count);
  }
  else
  {
    do
    {
      outPtr[0] = svtkScalarsToColors::ColorToUChar(inPtr[0]);
      outPtr[1] = svtkScalarsToColors::ColorToUChar(inPtr[1]);
      outPtr[2] = svtkScalarsToColors::ColorToUChar(inPtr[2]);
      outPtr[3] = static_cast<unsigned char>(inPtr[3] * alpha + 0.5);
      inPtr += numComponents;
      outPtr += 4;
    } while (--count);
  }
}

//----------------------------------------------------------------------------
template <class T>
void svtkScalarsToColorsLuminanceToLuminance(const T* inPtr, unsigned char* outPtr, svtkIdType count,
  int numComponents, double shift, double scale)
{
  static const double minval = 0;
  static const double maxval = 255.0;

  do
  {
    double l = inPtr[0];
    l = (l + shift) * scale;
    l = (l > minval ? l : minval);
    l = (l < maxval ? l : maxval);
    l += 0.5;
    outPtr[0] = static_cast<unsigned char>(l);
    inPtr += numComponents;
    outPtr += 1;
  } while (--count);
}

//----------------------------------------------------------------------------
template <class T>
void svtkScalarsToColorsLuminanceToRGB(const T* inPtr, unsigned char* outPtr, svtkIdType count,
  int numComponents, double shift, double scale)
{
  static const double minval = 0;
  static const double maxval = 255.0;

  do
  {
    double l = inPtr[0];
    l = (l + shift) * scale;
    l = (l > minval ? l : minval);
    l = (l < maxval ? l : maxval);
    unsigned char lc = static_cast<unsigned char>(l + 0.5);
    outPtr[0] = lc;
    outPtr[1] = lc;
    outPtr[2] = lc;
    inPtr += numComponents;
    outPtr += 3;
  } while (--count);
}

//----------------------------------------------------------------------------
template <class T>
void svtkScalarsToColorsRGBToLuminance(const T* inPtr, unsigned char* outPtr, svtkIdType count,
  int numComponents, double shift, double scale)
{
  static const double minval = 0;
  static const double maxval = 255.0;

  do
  {
    double r = inPtr[0];
    double g = inPtr[1];
    double b = inPtr[2];
    r = (r + shift) * scale;
    g = (g + shift) * scale;
    b = (b + shift) * scale;
    r = (r > minval ? r : minval);
    r = (r < maxval ? r : maxval);
    g = (g > minval ? g : minval);
    g = (g < maxval ? g : maxval);
    b = (b > minval ? b : minval);
    b = (b < maxval ? b : maxval);
    double l = svtkScalarsToColorsLuminance(r, g, b) + 0.5;
    outPtr[0] = static_cast<unsigned char>(l);
    inPtr += numComponents;
    outPtr += 1;
  } while (--count);
}

//----------------------------------------------------------------------------
template <class T>
void svtkScalarsToColorsRGBToRGB(const T* inPtr, unsigned char* outPtr, svtkIdType count,
  int numComponents, double shift, double scale)
{
  static const double minval = 0;
  static const double maxval = 255.0;

  do
  {
    double r = inPtr[0];
    double g = inPtr[1];
    double b = inPtr[2];
    r = (r + shift) * scale;
    g = (g + shift) * scale;
    b = (b + shift) * scale;
    r = (r > minval ? r : minval);
    r = (r < maxval ? r : maxval);
    g = (g > minval ? g : minval);
    g = (g < maxval ? g : maxval);
    b = (b > minval ? b : minval);
    b = (b < maxval ? b : maxval);
    r += 0.5;
    g += 0.5;
    b += 0.5;
    outPtr[0] = static_cast<unsigned char>(r);
    outPtr[1] = static_cast<unsigned char>(g);
    outPtr[2] = static_cast<unsigned char>(b);
    inPtr += numComponents;
    outPtr += 3;
  } while (--count);
}

//----------------------------------------------------------------------------
template <class T>
void svtkScalarsToColorsLuminanceToLuminanceAlpha(const T* inPtr, unsigned char* outPtr,
  svtkIdType count, int numComponents, double shift, double scale, double alpha)
{
  unsigned char a = svtkScalarsToColors::ColorToUChar(alpha);
  static const double minval = 0;
  static const double maxval = 255.0;

  do
  {
    double l = inPtr[0];
    l = (l + shift) * scale;
    l = (l > minval ? l : minval);
    l = (l < maxval ? l : maxval);
    l += 0.5;
    outPtr[0] = static_cast<unsigned char>(l);
    outPtr[1] = a;
    inPtr += numComponents;
    outPtr += 2;
  } while (--count);
}

//----------------------------------------------------------------------------
template <class T>
void svtkScalarsToColorsLuminanceToRGBA(const T* inPtr, unsigned char* outPtr, svtkIdType count,
  int numComponents, double shift, double scale, double alpha)
{
  unsigned char a = svtkScalarsToColors::ColorToUChar(alpha);
  static const double minval = 0;
  static const double maxval = 255.0;

  do
  {
    double l = inPtr[0];
    l = (l + shift) * scale;
    l = (l > minval ? l : minval);
    l = (l < maxval ? l : maxval);
    unsigned char lc = static_cast<unsigned char>(l + 0.5);
    outPtr[0] = lc;
    outPtr[1] = lc;
    outPtr[2] = lc;
    outPtr[3] = a;
    inPtr += numComponents;
    outPtr += 4;
  } while (--count);
}

//----------------------------------------------------------------------------
template <class T>
void svtkScalarsToColorsRGBToLuminanceAlpha(const T* inPtr, unsigned char* outPtr, svtkIdType count,
  int numComponents, double shift, double scale, double alpha)
{
  unsigned char a = svtkScalarsToColors::ColorToUChar(alpha);
  static const double minval = 0;
  static const double maxval = 255.0;

  do
  {
    double r = inPtr[0];
    double g = inPtr[1];
    double b = inPtr[2];
    r = (r + shift) * scale;
    g = (g + shift) * scale;
    b = (b + shift) * scale;
    r = (r > minval ? r : minval);
    r = (r < maxval ? r : maxval);
    g = (g > minval ? g : minval);
    g = (g < maxval ? g : maxval);
    b = (b > minval ? b : minval);
    b = (b < maxval ? b : maxval);
    double l = svtkScalarsToColorsLuminance(r, g, b) + 0.5;
    outPtr[0] = static_cast<unsigned char>(l);
    outPtr[1] = a;
    inPtr += numComponents;
    outPtr += 2;
  } while (--count);
}

//----------------------------------------------------------------------------
template <class T>
void svtkScalarsToColorsRGBToRGBA(const T* inPtr, unsigned char* outPtr, svtkIdType count,
  int numComponents, double shift, double scale, double alpha)
{
  unsigned char a = svtkScalarsToColors::ColorToUChar(alpha);
  static const double minval = 0;
  static const double maxval = 255.0;

  do
  {
    double r = inPtr[0];
    double g = inPtr[1];
    double b = inPtr[2];
    r = (r + shift) * scale;
    g = (g + shift) * scale;
    b = (b + shift) * scale;
    r = (r > minval ? r : minval);
    r = (r < maxval ? r : maxval);
    g = (g > minval ? g : minval);
    g = (g < maxval ? g : maxval);
    b = (b > minval ? b : minval);
    b = (b < maxval ? b : maxval);
    r += 0.5;
    g += 0.5;
    b += 0.5;
    outPtr[0] = static_cast<unsigned char>(r);
    outPtr[1] = static_cast<unsigned char>(g);
    outPtr[2] = static_cast<unsigned char>(b);
    outPtr[3] = a;
    inPtr += numComponents;
    outPtr += 4;
  } while (--count);
}

//----------------------------------------------------------------------------
template <class T>
void svtkScalarsToColorsLuminanceAlphaToLuminanceAlpha(const T* inPtr, unsigned char* outPtr,
  svtkIdType count, int numComponents, double shift, double scale, double alpha)
{
  static const double minval = 0;
  static const double maxval = 255.0;

  do
  {
    double l = inPtr[0];
    double a = inPtr[1];
    l = (l + shift) * scale;
    a = (a + shift) * scale;
    l = (l > minval ? l : minval);
    l = (l < maxval ? l : maxval);
    a = (a > minval ? a : minval);
    a = (a < maxval ? a : maxval);
    l += 0.5;
    a = a * alpha + 0.5;
    outPtr[0] = static_cast<unsigned char>(l);
    outPtr[1] = static_cast<unsigned char>(a);
    inPtr += numComponents;
    outPtr += 2;
  } while (--count);
}

//----------------------------------------------------------------------------
template <class T>
void svtkScalarsToColorsLuminanceAlphaToRGBA(const T* inPtr, unsigned char* outPtr, svtkIdType count,
  int numComponents, double shift, double scale, double alpha)
{
  static const double minval = 0;
  static const double maxval = 255.0;

  do
  {
    double l = inPtr[0];
    double a = inPtr[1];
    l = (l + shift) * scale;
    a = (a + shift) * scale;
    l = (l > minval ? l : minval);
    l = (l < maxval ? l : maxval);
    a = (a > minval ? a : minval);
    a = (a < maxval ? a : maxval);
    unsigned char lc = static_cast<unsigned char>(l + 0.5);
    a = a * alpha + 0.5;
    outPtr[0] = lc;
    outPtr[1] = lc;
    outPtr[2] = lc;
    outPtr[3] = static_cast<unsigned char>(a);
    inPtr += numComponents;
    outPtr += 4;
  } while (--count);
}

//----------------------------------------------------------------------------
template <class T>
void svtkScalarsToColorsRGBAToLuminanceAlpha(const T* inPtr, unsigned char* outPtr, svtkIdType count,
  int numComponents, double shift, double scale, double alpha)
{
  static const double minval = 0;
  static const double maxval = 255.0;

  do
  {
    double r = inPtr[0];
    double g = inPtr[1];
    double b = inPtr[2];
    double a = inPtr[3];
    r = (r + shift) * scale;
    g = (g + shift) * scale;
    b = (b + shift) * scale;
    a = (a + shift) * scale;
    r = (r > minval ? r : minval);
    r = (r < maxval ? r : maxval);
    g = (g > minval ? g : minval);
    g = (g < maxval ? g : maxval);
    b = (b > minval ? b : minval);
    b = (b < maxval ? b : maxval);
    a = (a > minval ? a : minval);
    a = (a < maxval ? a : maxval);
    a = a * alpha + 0.5;
    double l = svtkScalarsToColorsLuminance(r, g, b) + 0.5;
    outPtr[0] = static_cast<unsigned char>(l);
    outPtr[1] = static_cast<unsigned char>(a);
    inPtr += numComponents;
    outPtr += 2;
  } while (--count);
}

//----------------------------------------------------------------------------
template <class T>
void svtkScalarsToColorsRGBAToRGBA(const T* inPtr, unsigned char* outPtr, svtkIdType count,
  int numComponents, double shift, double scale, double alpha)
{
  static const double minval = 0;
  static const double maxval = 255.0;

  do
  {
    double r = inPtr[0];
    double g = inPtr[1];
    double b = inPtr[2];
    double a = inPtr[3];
    r = (r + shift) * scale;
    g = (g + shift) * scale;
    b = (b + shift) * scale;
    a = (a + shift) * scale;
    r = (r > minval ? r : minval);
    r = (r < maxval ? r : maxval);
    g = (g > minval ? g : minval);
    g = (g < maxval ? g : maxval);
    b = (b > minval ? b : minval);
    b = (b < maxval ? b : maxval);
    a = (a > minval ? a : minval);
    a = (a < maxval ? a : maxval);
    r += 0.5;
    g += 0.5;
    b += 0.5;
    a = a * alpha + 0.5;
    outPtr[0] = static_cast<unsigned char>(r);
    outPtr[1] = static_cast<unsigned char>(g);
    outPtr[2] = static_cast<unsigned char>(b);
    outPtr[3] = static_cast<unsigned char>(a);
    inPtr += numComponents;
    outPtr += 4;
  } while (--count);
}

//----------------------------------------------------------------------------
unsigned char* svtkScalarsToColorsUnpackBits(void* inPtr, svtkIdType numValues)
{
  svtkIdType n = (numValues + 7) % 8;
  unsigned char* newPtr = new unsigned char[n];

  unsigned char* tmpPtr = newPtr;
  unsigned char* bitdata = static_cast<unsigned char*>(inPtr);
  for (svtkIdType i = 0; i < n; i += 8)
  {
    unsigned char b = *bitdata++;
    int j = 8;
    do
    {
      *tmpPtr++ = ((b >> (--j)) & 0x01);
    } while (j);
  }

  return newPtr;
}

// end anonymous namespace
}

//----------------------------------------------------------------------------
void svtkScalarsToColors::MapColorsToColors(void* inPtr, unsigned char* outPtr, int inputDataType,
  int numberOfTuples, int numberOfComponents, int inputFormat, int outputFormat)
{
  if (outputFormat < SVTK_LUMINANCE || outputFormat > SVTK_RGBA)
  {
    svtkErrorMacro(<< "MapScalarsToColors: unrecognized color format");
    return;
  }

  if (numberOfTuples <= 0)
  {
    return;
  }

  unsigned char* newPtr = nullptr;
  if (inputDataType == SVTK_BIT)
  {
    newPtr = svtkScalarsToColorsUnpackBits(inPtr, numberOfTuples * numberOfComponents);
    inPtr = newPtr;
    inputDataType = SVTK_UNSIGNED_CHAR;
  }

  if (inputFormat <= 0 || inputFormat > numberOfComponents)
  {
    inputFormat = numberOfComponents;
  }

  double shift, scale;
  svtkScalarsToColorsComputeShiftScale(this, shift, scale);
  scale *= 255.0;

  double alpha = this->Alpha;
  if (alpha < 0)
  {
    alpha = 0;
  }
  if (alpha > 1)
  {
    alpha = 1;
  }

  if (inputDataType == SVTK_UNSIGNED_CHAR && static_cast<int>(shift * scale + 0.5) == 0 &&
    static_cast<int>((255 + shift) * scale + 0.5) == 255)
  {
    if (outputFormat == SVTK_RGBA)
    {
      if (inputFormat == SVTK_LUMINANCE)
      {
        svtkScalarsToColorsLuminanceToRGBA(
          static_cast<unsigned char*>(inPtr), outPtr, numberOfTuples, numberOfComponents, alpha);
      }
      else if (inputFormat == SVTK_LUMINANCE_ALPHA)
      {
        svtkScalarsToColorsLuminanceAlphaToRGBA(
          static_cast<unsigned char*>(inPtr), outPtr, numberOfTuples, numberOfComponents, alpha);
      }
      else if (inputFormat == SVTK_RGB)
      {
        svtkScalarsToColorsRGBToRGBA(
          static_cast<unsigned char*>(inPtr), outPtr, numberOfTuples, numberOfComponents, alpha);
      }
      else
      {
        svtkScalarsToColorsRGBAToRGBA(
          static_cast<unsigned char*>(inPtr), outPtr, numberOfTuples, numberOfComponents, alpha);
      }
    }
    else if (outputFormat == SVTK_RGB)
    {
      if (inputFormat < SVTK_RGB)
      {
        svtkScalarsToColorsLuminanceToRGB(
          static_cast<unsigned char*>(inPtr), outPtr, numberOfTuples, numberOfComponents);
      }
      else
      {
        svtkScalarsToColorsRGBToRGB(
          static_cast<unsigned char*>(inPtr), outPtr, numberOfTuples, numberOfComponents);
      }
    }
    else if (outputFormat == SVTK_LUMINANCE_ALPHA)
    {
      if (inputFormat == SVTK_LUMINANCE)
      {
        svtkScalarsToColorsLuminanceToLuminanceAlpha(
          static_cast<unsigned char*>(inPtr), outPtr, numberOfTuples, numberOfComponents, alpha);
      }
      else if (inputFormat == SVTK_LUMINANCE_ALPHA)
      {
        svtkScalarsToColorsLuminanceAlphaToLuminanceAlpha(
          static_cast<unsigned char*>(inPtr), outPtr, numberOfTuples, numberOfComponents, alpha);
      }
      else if (inputFormat == SVTK_RGB)
      {
        svtkScalarsToColorsRGBToLuminanceAlpha(
          static_cast<unsigned char*>(inPtr), outPtr, numberOfTuples, numberOfComponents, alpha);
      }
      else
      {
        svtkScalarsToColorsRGBAToLuminanceAlpha(
          static_cast<unsigned char*>(inPtr), outPtr, numberOfTuples, numberOfComponents, alpha);
      }
    }
    else if (outputFormat == SVTK_LUMINANCE)
    {
      if (inputFormat < SVTK_RGB)
      {
        svtkScalarsToColorsLuminanceToLuminance(
          static_cast<unsigned char*>(inPtr), outPtr, numberOfTuples, numberOfComponents);
      }
      else
      {
        svtkScalarsToColorsRGBToLuminance(
          static_cast<unsigned char*>(inPtr), outPtr, numberOfTuples, numberOfComponents);
      }
    }
  }
  else
  {
    // must apply shift scale and/or do type conversion
    if (outputFormat == SVTK_RGBA)
    {
      if (inputFormat == SVTK_LUMINANCE)
      {
        switch (inputDataType)
        {
          svtkTemplateAliasMacro(svtkScalarsToColorsLuminanceToRGBA(static_cast<SVTK_TT*>(inPtr),
            outPtr, numberOfTuples, numberOfComponents, shift, scale, alpha));
        }
      }
      else if (inputFormat == SVTK_LUMINANCE_ALPHA)
      {
        switch (inputDataType)
        {
          svtkTemplateAliasMacro(svtkScalarsToColorsLuminanceAlphaToRGBA(static_cast<SVTK_TT*>(inPtr),
            outPtr, numberOfTuples, numberOfComponents, shift, scale, alpha));
        }
      }
      else if (inputFormat == SVTK_RGB)
      {
        switch (inputDataType)
        {
          svtkTemplateAliasMacro(svtkScalarsToColorsRGBToRGBA(static_cast<SVTK_TT*>(inPtr), outPtr,
            numberOfTuples, numberOfComponents, shift, scale, alpha));
        }
      }
      else
      {
        switch (inputDataType)
        {
          svtkTemplateAliasMacro(svtkScalarsToColorsRGBAToRGBA(static_cast<SVTK_TT*>(inPtr), outPtr,
            numberOfTuples, numberOfComponents, shift, scale, alpha));
        }
      }
    }
    else if (outputFormat == SVTK_RGB)
    {
      if (inputFormat < SVTK_RGB)
      {
        switch (inputDataType)
        {
          svtkTemplateAliasMacro(svtkScalarsToColorsLuminanceToRGB(
            static_cast<SVTK_TT*>(inPtr), outPtr, numberOfTuples, numberOfComponents, shift, scale));
        }
      }
      else
      {
        switch (inputDataType)
        {
          svtkTemplateAliasMacro(svtkScalarsToColorsRGBToRGB(
            static_cast<SVTK_TT*>(inPtr), outPtr, numberOfTuples, numberOfComponents, shift, scale));
        }
      }
    }
    else if (outputFormat == SVTK_LUMINANCE_ALPHA)
    {
      if (inputFormat == SVTK_LUMINANCE)
      {
        switch (inputDataType)
        {
          svtkTemplateAliasMacro(
            svtkScalarsToColorsLuminanceToLuminanceAlpha(static_cast<SVTK_TT*>(inPtr), outPtr,
              numberOfTuples, numberOfComponents, shift, scale, alpha));
        }
      }
      else if (inputFormat == SVTK_LUMINANCE_ALPHA)
      {
        switch (inputDataType)
        {
          svtkTemplateAliasMacro(
            svtkScalarsToColorsLuminanceAlphaToLuminanceAlpha(static_cast<SVTK_TT*>(inPtr), outPtr,
              numberOfTuples, numberOfComponents, shift, scale, alpha));
        }
      }
      else if (inputFormat == SVTK_RGB)
      {
        switch (inputDataType)
        {
          svtkTemplateAliasMacro(svtkScalarsToColorsRGBToLuminanceAlpha(static_cast<SVTK_TT*>(inPtr),
            outPtr, numberOfTuples, numberOfComponents, shift, scale, alpha));
        }
      }
      else
      {
        switch (inputDataType)
        {
          svtkTemplateAliasMacro(svtkScalarsToColorsRGBAToLuminanceAlpha(static_cast<SVTK_TT*>(inPtr),
            outPtr, numberOfTuples, numberOfComponents, shift, scale, alpha));
        }
      }
    }
    else if (outputFormat == SVTK_LUMINANCE)
    {
      if (inputFormat < SVTK_RGB)
      {
        switch (inputDataType)
        {
          svtkTemplateAliasMacro(svtkScalarsToColorsLuminanceToLuminance(
            static_cast<SVTK_TT*>(inPtr), outPtr, numberOfTuples, numberOfComponents, shift, scale));
        }
      }
      else
      {
        switch (inputDataType)
        {
          svtkTemplateAliasMacro(svtkScalarsToColorsRGBToLuminance(
            static_cast<SVTK_TT*>(inPtr), outPtr, numberOfTuples, numberOfComponents, shift, scale));
        }
      }
    }
  }

  delete[] newPtr;
}

//----------------------------------------------------------------------------
template <class T>
void svtkScalarsToColorsMapVectorsToMagnitude(
  const T* inPtr, double* outPtr, int numTuples, int vectorSize, int inInc)
{
  do
  {
    int n = vectorSize;
    double v = 0.0;
    do
    {
      double u = static_cast<double>(*inPtr++);
      v += u * u;
    } while (--n);
    *outPtr++ = sqrt(v);
    inPtr += inInc;
  } while (--numTuples);
}

//----------------------------------------------------------------------------
void svtkScalarsToColors::MapVectorsToMagnitude(void* inPtr, double* outPtr, int inputDataType,
  int numberOfTuples, int numberOfComponents, int vectorSize)
{
  if (numberOfTuples <= 0)
  {
    return;
  }

  unsigned char* newPtr = nullptr;
  if (inputDataType == SVTK_BIT)
  {
    newPtr = svtkScalarsToColorsUnpackBits(inPtr, numberOfTuples * numberOfComponents);
    inPtr = newPtr;
    inputDataType = SVTK_UNSIGNED_CHAR;
  }

  if (vectorSize <= 0 || vectorSize > numberOfComponents)
  {
    vectorSize = numberOfComponents;
  }
  int inInc = numberOfComponents - vectorSize;

  switch (inputDataType)
  {
    svtkTemplateAliasMacro(svtkScalarsToColorsMapVectorsToMagnitude(
      static_cast<SVTK_TT*>(inPtr), outPtr, numberOfTuples, vectorSize, inInc));
  }

  delete[] newPtr;
}

//----------------------------------------------------------------------------
void svtkScalarsToColors::MapScalarsThroughTable2(void* inPtr, unsigned char* outPtr,
  int inputDataType, int numberOfTuples, int numberOfComponents, int outputFormat)
{
  if (outputFormat < SVTK_LUMINANCE || outputFormat > SVTK_RGBA)
  {
    svtkErrorMacro(<< "MapScalarsThroughTable2: unrecognized color format");
    return;
  }

  if (numberOfTuples <= 0)
  {
    return;
  }

  unsigned char* newPtr = nullptr;
  if (inputDataType == SVTK_BIT)
  {
    newPtr = svtkScalarsToColorsUnpackBits(inPtr, numberOfTuples * numberOfComponents);
    inPtr = newPtr;
    inputDataType = SVTK_UNSIGNED_CHAR;
  }

  double shift, scale;
  svtkScalarsToColorsComputeShiftScale(this, shift, scale);
  scale *= 255.0;

  double alpha = this->Alpha;
  if (alpha < 0)
  {
    alpha = 0;
  }
  if (alpha > 1)
  {
    alpha = 1;
  }

  if (inputDataType == SVTK_UNSIGNED_CHAR && static_cast<int>(shift * scale + 0.5) == 0 &&
    static_cast<int>((255 + shift) * scale + 0.5) == 255)
  {
    if (outputFormat == SVTK_RGBA)
    {
      svtkScalarsToColorsLuminanceToRGBA(
        static_cast<unsigned char*>(inPtr), outPtr, numberOfTuples, numberOfComponents, alpha);
    }
    else if (outputFormat == SVTK_RGB)
    {
      svtkScalarsToColorsLuminanceToRGB(
        static_cast<unsigned char*>(inPtr), outPtr, numberOfTuples, numberOfComponents);
    }
    else if (outputFormat == SVTK_LUMINANCE_ALPHA)
    {
      svtkScalarsToColorsLuminanceToLuminanceAlpha(
        static_cast<unsigned char*>(inPtr), outPtr, numberOfTuples, numberOfComponents, alpha);
    }
    else if (outputFormat == SVTK_LUMINANCE)
    {
      svtkScalarsToColorsLuminanceToLuminance(
        static_cast<unsigned char*>(inPtr), outPtr, numberOfTuples, numberOfComponents);
    }
  }
  else
  {
    // must apply shift scale and/or do type conversion
    if (outputFormat == SVTK_RGBA)
    {
      switch (inputDataType)
      {
        svtkTemplateAliasMacro(svtkScalarsToColorsLuminanceToRGBA(static_cast<SVTK_TT*>(inPtr), outPtr,
          numberOfTuples, numberOfComponents, shift, scale, alpha));

        default:
          svtkErrorMacro(<< "MapScalarsThroughTable2: Unknown input data type");
          break;
      }
    }
    else if (outputFormat == SVTK_RGB)
    {
      switch (inputDataType)
      {
        svtkTemplateAliasMacro(svtkScalarsToColorsLuminanceToRGB(
          static_cast<SVTK_TT*>(inPtr), outPtr, numberOfTuples, numberOfComponents, shift, scale));

        default:
          svtkErrorMacro(<< "MapScalarsThroughTable2: Unknown input data type");
          break;
      }
    }
    else if (outputFormat == SVTK_LUMINANCE_ALPHA)
    {
      switch (inputDataType)
      {
        svtkTemplateAliasMacro(
          svtkScalarsToColorsLuminanceToLuminanceAlpha(static_cast<SVTK_TT*>(inPtr), outPtr,
            numberOfTuples, numberOfComponents, shift, scale, alpha));

        default:
          svtkErrorMacro(<< "MapScalarsThroughTable2: Unknown input data type");
          break;
      }
    }
    else if (outputFormat == SVTK_LUMINANCE)
    {
      switch (inputDataType)
      {
        svtkTemplateAliasMacro(svtkScalarsToColorsLuminanceToLuminance(
          static_cast<SVTK_TT*>(inPtr), outPtr, numberOfTuples, numberOfComponents, shift, scale));

        default:
          svtkErrorMacro(<< "MapScalarsThroughTable2: Unknown input data type");
          break;
      }
    }
  }

  delete[] newPtr;
}

// The callForAnyType is used to write generic code that works with any
// svtkDataArray derived types.
//
// This macro calls a template function (on the data type stored in the
// array).  Example usage:
//   callForAnyType(array, myFunc(static_cast<SVTK_TT*>(data), arg2));
// where 'array' is a svtkDataArray and
//       'data' could be: array->GetVoidPointer(0)
#define callForAnyType(array, call)                                                                \
  switch (array->GetDataType())                                                                    \
  {                                                                                                \
    svtkTemplateMacro(call);                                                                        \
  }

//----------------------------------------------------------------------------
svtkUnsignedCharArray* svtkScalarsToColors::ConvertToRGBA(
  svtkDataArray* colors, int numComp, int numTuples)
{
  if (svtkArrayDownCast<svtkCharArray>(colors) != nullptr)
  {
    svtkErrorMacro(<< "char type does not have enough values to hold a color");
    return nullptr;
  }

  if (numComp == 4 && this->Alpha >= 1.0 &&
    svtkArrayDownCast<svtkUnsignedCharArray>(colors) != nullptr)
  {
    svtkUnsignedCharArray* c = svtkArrayDownCast<svtkUnsignedCharArray>(colors);
    c->Register(this);
    return c;
  }

  svtkUnsignedCharArray* newColors = svtkUnsignedCharArray::New();
  newColors->SetNumberOfComponents(4);
  newColors->SetNumberOfTuples(numTuples);
  unsigned char* nptr = newColors->GetPointer(0);
  double alpha = this->Alpha;
  alpha = (alpha > 0 ? alpha : 0);
  alpha = (alpha < 1 ? alpha : 1);

  if (numTuples <= 0)
  {
    return newColors;
  }

  switch (numComp)
  {
    case 1:
      callForAnyType(colors,
        svtkScalarsToColorsLuminanceToRGBA(
          static_cast<SVTK_TT*>(colors->GetVoidPointer(0)), nptr, numTuples, numComp, alpha));
      break;

    case 2:
      callForAnyType(colors,
        svtkScalarsToColorsLuminanceAlphaToRGBA(
          static_cast<SVTK_TT*>(colors->GetVoidPointer(0)), nptr, numTuples, numComp, alpha));
      break;

    case 3:
      callForAnyType(colors,
        svtkScalarsToColorsRGBToRGBA(
          static_cast<SVTK_TT*>(colors->GetVoidPointer(0)), nptr, numTuples, numComp, alpha));
      break;

    case 4:
      callForAnyType(colors,
        svtkScalarsToColorsRGBAToRGBA(
          static_cast<SVTK_TT*>(colors->GetVoidPointer(0)), nptr, numTuples, numComp, alpha));
      break;

    default:
      svtkErrorMacro(<< "Cannot convert colors");
      return nullptr;
  }

  return newColors;
}

//----------------------------------------------------------------------------
void svtkScalarsToColors::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Alpha: " << this->Alpha << "\n";
  if (this->VectorMode == svtkScalarsToColors::MAGNITUDE)
  {
    os << indent << "VectorMode: Magnitude\n";
  }
  else if (this->VectorMode == svtkScalarsToColors::RGBCOLORS)
  {
    os << indent << "VectorMode: RGBColors\n";
  }
  else
  {
    os << indent << "VectorMode: Component\n";
  }
  os << indent << "VectorComponent: " << this->VectorComponent << "\n";
  os << indent << "VectorSize: " << this->VectorSize << "\n";
  os << indent << "IndexedLookup: " << (this->IndexedLookup ? "ON" : "OFF") << "\n";
  svtkIdType nv = this->GetNumberOfAnnotatedValues();
  os << indent << "AnnotatedValues: " << nv << (nv > 0 ? " entries:\n" : " entries.\n");
  svtkIndent i2(indent.GetNextIndent());
  for (svtkIdType i = 0; i < nv; ++i)
  {
    os << i2 << i << ": value: " << this->GetAnnotatedValue(i).ToString() << " note: \""
       << this->GetAnnotation(i) << "\"\n";
  }
}

//----------------------------------------------------------------------------
void svtkScalarsToColors::SetAnnotations(svtkAbstractArray* values, svtkStringArray* annotations)
{
  if ((values && !annotations) || (!values && annotations))
    return;

  if (values && annotations && values->GetNumberOfTuples() != annotations->GetNumberOfTuples())
  {
    svtkErrorMacro(<< "Values and annotations do not have the same number of tuples ("
                  << values->GetNumberOfTuples() << " and " << annotations->GetNumberOfTuples()
                  << ", respectively. Ignoring.");
    return;
  }

  if (this->AnnotatedValues && !values)
  {
    this->AnnotatedValues->Delete();
    this->AnnotatedValues = nullptr;
  }
  else if (values)
  { // Ensure arrays are of the same type before copying.
    if (this->AnnotatedValues)
    {
      if (this->AnnotatedValues->GetDataType() != values->GetDataType())
      {
        this->AnnotatedValues->Delete();
        this->AnnotatedValues = nullptr;
      }
    }
    if (!this->AnnotatedValues)
    {
      this->AnnotatedValues = svtkAbstractArray::CreateArray(values->GetDataType());
    }
  }
  bool sameVals = (values == this->AnnotatedValues);
  if (!sameVals && values)
  {
    this->AnnotatedValues->DeepCopy(values);
  }

  if (this->Annotations && !annotations)
  {
    this->Annotations->Delete();
    this->Annotations = nullptr;
  }
  else if (!this->Annotations && annotations)
  {
    this->Annotations = svtkStringArray::New();
  }
  bool sameText = (annotations == this->Annotations);
  if (!sameText)
  {
    this->Annotations->DeepCopy(annotations);
  }
  this->UpdateAnnotatedValueMap();
  this->Modified();
}

//----------------------------------------------------------------------------
svtkIdType svtkScalarsToColors::SetAnnotation(svtkVariant value, svtkStdString annotation)
{
  svtkIdType i = this->CheckForAnnotatedValue(value);
  bool modified = false;
  if (i >= 0)
  {
    if (this->Annotations->GetValue(i) != annotation)
    {
      this->Annotations->SetValue(i, annotation);
      modified = true;
    }
  }
  else
  {
    i = this->Annotations->InsertNextValue(annotation);
    this->AnnotatedValues->InsertVariantValue(i, value);
    modified = true;
  }
  if (modified)
  {
    this->UpdateAnnotatedValueMap();
    this->Modified();
  }
  return i;
}

//----------------------------------------------------------------------------
svtkIdType svtkScalarsToColors::SetAnnotation(svtkStdString value, svtkStdString annotation)
{
  bool valid;
  svtkVariant val(value);
  double x = val.ToDouble(&valid);
  if (valid)
  {
    return this->SetAnnotation(x, annotation);
  }
  return this->SetAnnotation(val, annotation);
}

//----------------------------------------------------------------------------
svtkIdType svtkScalarsToColors::GetNumberOfAnnotatedValues()
{
  return this->AnnotatedValues ? this->AnnotatedValues->GetNumberOfTuples() : 0;
}

//----------------------------------------------------------------------------
svtkVariant svtkScalarsToColors::GetAnnotatedValue(svtkIdType idx)
{
  if (!this->AnnotatedValues || idx < 0 || idx >= this->AnnotatedValues->GetNumberOfTuples())
  {
    svtkVariant invalid;
    return invalid;
  }
  return this->AnnotatedValues->GetVariantValue(idx);
}

//----------------------------------------------------------------------------
svtkStdString svtkScalarsToColors::GetAnnotation(svtkIdType idx)
{
  if (!this->Annotations)
  /* Don't check idx as Annotations->GetValue() does:
   * || idx < 0 || idx >= this->Annotations->GetNumberOfTuples())
   */
  {
    svtkStdString empty;
    return empty;
  }
  return this->Annotations->GetValue(idx);
}

//----------------------------------------------------------------------------
svtkIdType svtkScalarsToColors::GetAnnotatedValueIndex(svtkVariant val)
{
  return (this->AnnotatedValues ? this->CheckForAnnotatedValue(val) : -1);
}

//----------------------------------------------------------------------------
bool svtkScalarsToColors::RemoveAnnotation(svtkVariant value)
{
  svtkIdType i = this->CheckForAnnotatedValue(value);
  bool needToRemove = (i >= 0);
  if (needToRemove)
  {
    // Note that this is the number of values minus 1:
    svtkIdType na = this->AnnotatedValues->GetMaxId();
    for (; i < na; ++i)
    {
      this->AnnotatedValues->SetVariantValue(i, this->AnnotatedValues->GetVariantValue(i + 1));
      this->Annotations->SetValue(i, this->Annotations->GetValue(i + 1));
    }
    this->AnnotatedValues->Resize(na);
    this->Annotations->Resize(na);
    this->UpdateAnnotatedValueMap();
    this->Modified();
  }
  return needToRemove;
}

//----------------------------------------------------------------------------
void svtkScalarsToColors::ResetAnnotations()
{
  if (!this->Annotations)
  {
    svtkVariantArray* va = svtkVariantArray::New();
    svtkStringArray* sa = svtkStringArray::New();
    this->SetAnnotations(va, sa);
    va->Delete();
    sa->Delete();
  }
  this->AnnotatedValues->Reset();
  this->Annotations->Reset();
  this->AnnotatedValueMap->clear();
  this->Modified();
}

//----------------------------------------------------------------------------
void svtkScalarsToColors::GetAnnotationColor(const svtkVariant& val, double rgba[4])
{
  if (this->IndexedLookup)
  {
    svtkIdType i = this->GetAnnotatedValueIndex(val);
    this->GetIndexedColor(i, rgba);
  }
  else
  {
    this->GetColor(val.ToDouble(), rgba);
    rgba[3] = 1.;
  }
}

//----------------------------------------------------------------------------
svtkIdType svtkScalarsToColors::CheckForAnnotatedValue(svtkVariant value)
{
  if (!this->Annotations)
  {
    svtkVariantArray* va = svtkVariantArray::New();
    svtkStringArray* sa = svtkStringArray::New();
    this->SetAnnotations(va, sa);
    va->FastDelete();
    sa->FastDelete();
  }
  return this->GetAnnotatedValueIndexInternal(value);
}

//----------------------------------------------------------------------------
// An unsafe version of svtkScalarsToColors::CheckForAnnotatedValue for
// internal use (no pointer checks performed)
svtkIdType svtkScalarsToColors::GetAnnotatedValueIndexInternal(const svtkVariant& value)
{
  svtkInternalAnnotatedValueMap::iterator it = this->AnnotatedValueMap->find(value);
  svtkIdType nv = this->GetNumberOfAvailableColors();
  svtkIdType i = (it == this->AnnotatedValueMap->end() ? -1 : (nv ? it->second % nv : it->second));
  return i;
}

//----------------------------------------------------------------------------
void svtkScalarsToColors::GetIndexedColor(svtkIdType, double rgba[4])
{
  rgba[0] = rgba[1] = rgba[2] = rgba[3] = 0.;
}

//----------------------------------------------------------------------------
void svtkScalarsToColors::UpdateAnnotatedValueMap()
{
  this->AnnotatedValueMap->clear();

  svtkIdType na = this->AnnotatedValues ? this->AnnotatedValues->GetMaxId() + 1 : 0;
  for (svtkIdType i = 0; i < na; ++i)
  {
    (*this->AnnotatedValueMap)[this->AnnotatedValues->GetVariantValue(i)] = i;
  }
}
