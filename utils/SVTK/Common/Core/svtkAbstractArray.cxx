/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAbstractArray.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkAbstractArray.h"

#include "svtkBitArray.h"
#include "svtkCharArray.h"
#include "svtkDoubleArray.h"
#include "svtkFloatArray.h"
#include "svtkIdList.h"
#include "svtkIdTypeArray.h"
#include "svtkInformation.h"
#include "svtkInformationDoubleVectorKey.h"
#include "svtkInformationInformationVectorKey.h"
#include "svtkInformationIntegerKey.h"
#include "svtkInformationVariantVectorKey.h"
#include "svtkInformationVector.h"
#include "svtkIntArray.h"
#include "svtkLongArray.h"
#include "svtkLongLongArray.h"
#include "svtkMath.h"
#include "svtkMinimalStandardRandomSequence.h"
#include "svtkNew.h"
#include "svtkShortArray.h"
#include "svtkSignedCharArray.h"
#include "svtkStringArray.h"
#include "svtkUnicodeString.h" // for svtkSuperExtraExtendedTemplateMacro
#include "svtkUnicodeStringArray.h"
#include "svtkUnsignedCharArray.h"
#include "svtkUnsignedIntArray.h"
#include "svtkUnsignedLongArray.h"
#include "svtkUnsignedLongLongArray.h"
#include "svtkUnsignedShortArray.h"
#include "svtkVariantArray.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>

svtkInformationKeyMacro(svtkAbstractArray, GUI_HIDE, Integer);
svtkInformationKeyMacro(svtkAbstractArray, PER_COMPONENT, InformationVector);
svtkInformationKeyMacro(svtkAbstractArray, PER_FINITE_COMPONENT, InformationVector);
svtkInformationKeyMacro(svtkAbstractArray, DISCRETE_VALUES, VariantVector);
svtkInformationKeyRestrictedMacro(
  svtkAbstractArray, DISCRETE_VALUE_SAMPLE_PARAMETERS, DoubleVector, 2);

namespace
{
typedef std::vector<svtkStdString*> svtkInternalComponentNameBase;
}
class svtkAbstractArray::svtkInternalComponentNames : public svtkInternalComponentNameBase
{
};

//----------------------------------------------------------------------------
// Construct object with sane defaults.
svtkAbstractArray::svtkAbstractArray()
{
  this->Size = 0;
  this->MaxId = -1;
  this->NumberOfComponents = 1;
  this->Name = nullptr;
  this->RebuildArray = false;
  this->Information = nullptr;
  this->ComponentNames = nullptr;

  this->MaxDiscreteValues = svtkAbstractArray::MAX_DISCRETE_VALUES; // 32
}

//----------------------------------------------------------------------------
svtkAbstractArray::~svtkAbstractArray()
{
  if (this->ComponentNames)
  {
    for (unsigned int i = 0; i < this->ComponentNames->size(); ++i)
    {
      delete this->ComponentNames->at(i);
    }
    this->ComponentNames->clear();
    delete this->ComponentNames;
    this->ComponentNames = nullptr;
  }

  this->SetName(nullptr);
  this->SetInformation(nullptr);
}

//----------------------------------------------------------------------------
void svtkAbstractArray::SetComponentName(svtkIdType component, const char* name)
{
  if (component < 0 || name == nullptr)
  {
    return;
  }
  unsigned int index = static_cast<unsigned int>(component);
  if (this->ComponentNames == nullptr)
  {
    // delayed allocate
    this->ComponentNames = new svtkAbstractArray::svtkInternalComponentNames();
  }

  if (index == this->ComponentNames->size())
  {
    // the array isn't large enough, so we will resize
    this->ComponentNames->push_back(new svtkStdString(name));
    return;
  }
  else if (index > this->ComponentNames->size())
  {
    this->ComponentNames->resize(index + 1, nullptr);
  }

  // replace an existing element
  svtkStdString* compName = this->ComponentNames->at(index);
  if (!compName)
  {
    compName = new svtkStdString(name);
    this->ComponentNames->at(index) = compName;
  }
  else
  {
    compName->assign(name);
  }
}

//----------------------------------------------------------------------------
const char* svtkAbstractArray::GetComponentName(svtkIdType component) const
{
  unsigned int index = static_cast<unsigned int>(component);
  if (!this->ComponentNames || component < 0 || index >= this->ComponentNames->size())
  {
    // make sure we have valid vector
    return nullptr;
  }

  svtkStdString* compName = this->ComponentNames->at(index);
  return (compName) ? compName->c_str() : nullptr;
}

//----------------------------------------------------------------------------
bool svtkAbstractArray::HasAComponentName() const
{
  return (this->ComponentNames) ? (!this->ComponentNames->empty()) : 0;
}

//----------------------------------------------------------------------------
int svtkAbstractArray::CopyComponentNames(svtkAbstractArray* da)
{
  if (da && da != this && da->ComponentNames)
  {
    // clear the vector of the all data
    if (!this->ComponentNames)
    {
      this->ComponentNames = new svtkAbstractArray::svtkInternalComponentNames();
    }

    // copy the passed in components
    for (unsigned int i = 0; i < this->ComponentNames->size(); ++i)
    {
      delete this->ComponentNames->at(i);
    }
    this->ComponentNames->clear();
    this->ComponentNames->reserve(da->ComponentNames->size());
    const char* name;
    for (unsigned int i = 0; i < da->ComponentNames->size(); ++i)
    {
      name = da->GetComponentName(i);
      if (name)
      {
        this->SetComponentName(i, name);
      }
    }
    return 1;
  }
  return 0;
}

//----------------------------------------------------------------------------
bool svtkAbstractArray::SetNumberOfValues(svtkIdType numValues)
{
  svtkIdType numTuples = this->NumberOfComponents == 1
    ? numValues
    : (numValues + this->NumberOfComponents - 1) / this->NumberOfComponents;
  if (!this->Resize(numTuples))
  {
    return false;
  }
  this->MaxId = numValues - 1;
  return true;
}

//----------------------------------------------------------------------------
void svtkAbstractArray::SetInformation(svtkInformation* args)
{
  // Same as in svtkCxxSetObjectMacro, but no Modified() so that
  // this doesn't cause extra pipeline updates.
  svtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting Information to " << args);
  if (this->Information != args)
  {
    svtkInformation* tempSGMacroVar = this->Information;
    this->Information = args;
    if (this->Information != nullptr)
    {
      this->Information->Register(this);
    }
    if (tempSGMacroVar != nullptr)
    {
      tempSGMacroVar->UnRegister(this);
    }
  }
}

//----------------------------------------------------------------------------
void svtkAbstractArray::GetTuples(svtkIdList* tupleIds, svtkAbstractArray* aa)
{
  if (aa->GetNumberOfComponents() != this->GetNumberOfComponents())
  {
    svtkWarningMacro("Number of components for input and output do not match.");
    return;
  }
  // Here we give the slowest implementation. Subclasses can override
  // to use the knowledge about the data.
  svtkIdType num = tupleIds->GetNumberOfIds();
  for (svtkIdType i = 0; i < num; i++)
  {
    aa->SetTuple(i, tupleIds->GetId(i), this);
  }
}

//----------------------------------------------------------------------------
void svtkAbstractArray::GetTuples(svtkIdType p1, svtkIdType p2, svtkAbstractArray* aa)
{
  if (aa->GetNumberOfComponents() != this->GetNumberOfComponents())
  {
    svtkWarningMacro("Number of components for input and output do not match.");
    return;
  }

  // Here we give the slowest implementation. Subclasses can override
  // to use the knowledge about the data.
  svtkIdType num = p2 - p1 + 1;
  for (svtkIdType i = 0; i < num; i++)
  {
    aa->SetTuple(i, (p1 + i), this);
  }
}

//----------------------------------------------------------------------------
bool svtkAbstractArray::HasStandardMemoryLayout() const
{
  return true;
}

//----------------------------------------------------------------------------
void svtkAbstractArray::DeepCopy(svtkAbstractArray* da)
{
  if (!da || da == this)
  {
    return;
  }

  if (da->HasInformation())
  {
    this->CopyInformation(da->GetInformation(), /*deep=*/1);
  }
  else
  {
    this->SetInformation(nullptr);
  }

  this->SetName(da->Name);

  this->CopyComponentNames(da);
}

//----------------------------------------------------------------------------
void svtkAbstractArray::ExportToVoidPointer(void* dest)
{
  if (this->MaxId > 0 && this->GetDataTypeSize() > 0)
  {
    void* src = this->GetVoidPointer(0);
    memcpy(dest, src, ((this->MaxId + 1) * this->GetDataTypeSize()));
  }
}

//----------------------------------------------------------------------------
int svtkAbstractArray::CopyInformation(svtkInformation* infoFrom, int deep)
{
  // Copy all keys. NOTE: subclasses rely on this.
  svtkInformation* myInfo = this->GetInformation();
  myInfo->Copy(infoFrom, deep);

  // Remove any keys we own that are not to be copied here.
  // For now, remove per-component metadata.
  myInfo->Remove(PER_COMPONENT());
  myInfo->Remove(PER_FINITE_COMPONENT());
  myInfo->Remove(DISCRETE_VALUES());

  return 1;
}

//----------------------------------------------------------------------------
// call modified on superclass
void svtkAbstractArray::Modified()
{
  if (this->HasInformation())
  {
    svtkInformation* info = this->GetInformation();
    // Clear key-value pairs that are now out of date.
    info->Remove(PER_COMPONENT());
    info->Remove(PER_FINITE_COMPONENT());
  }
  this->Superclass::Modified();
}

//----------------------------------------------------------------------------
svtkInformation* svtkAbstractArray::GetInformation()
{
  if (!this->Information)
  {
    svtkInformation* info = svtkInformation::New();
    this->SetInformation(info);
    info->FastDelete();
  }
  return this->Information;
}

//----------------------------------------------------------------------------
template <class T>
int svtkAbstractArrayGetDataTypeSize(T*)
{
  return sizeof(T);
}

int svtkAbstractArray::GetDataTypeSize(int type)
{
  switch (type)
  {
    svtkTemplateMacro(return svtkAbstractArrayGetDataTypeSize(static_cast<SVTK_TT*>(nullptr)));

    case SVTK_BIT:
    case SVTK_STRING:
    case SVTK_UNICODE_STRING:
      return 0;

    default:
      svtkGenericWarningMacro(<< "Unsupported data type!");
  }

  return 1;
}

// ----------------------------------------------------------------------
svtkAbstractArray* svtkAbstractArray::CreateArray(int dataType)
{
  switch (dataType)
  {
    case SVTK_BIT:
      return svtkBitArray::New();

    case SVTK_CHAR:
      return svtkCharArray::New();

    case SVTK_SIGNED_CHAR:
      return svtkSignedCharArray::New();

    case SVTK_UNSIGNED_CHAR:
      return svtkUnsignedCharArray::New();

    case SVTK_SHORT:
      return svtkShortArray::New();

    case SVTK_UNSIGNED_SHORT:
      return svtkUnsignedShortArray::New();

    case SVTK_INT:
      return svtkIntArray::New();

    case SVTK_UNSIGNED_INT:
      return svtkUnsignedIntArray::New();

    case SVTK_LONG:
      return svtkLongArray::New();

    case SVTK_UNSIGNED_LONG:
      return svtkUnsignedLongArray::New();

    case SVTK_LONG_LONG:
      return svtkLongLongArray::New();

    case SVTK_UNSIGNED_LONG_LONG:
      return svtkUnsignedLongLongArray::New();

    case SVTK_FLOAT:
      return svtkFloatArray::New();

    case SVTK_DOUBLE:
      return svtkDoubleArray::New();

    case SVTK_ID_TYPE:
      return svtkIdTypeArray::New();

    case SVTK_STRING:
      return svtkStringArray::New();

    case SVTK_UNICODE_STRING:
      return svtkUnicodeStringArray::New();

    case SVTK_VARIANT:
      return svtkVariantArray::New();

    default:
      break;
  }

  svtkGenericWarningMacro("Unsupported data type: " << dataType << "! Setting to SVTK_DOUBLE");
  return svtkDoubleArray::New();
}

//---------------------------------------------------------------------------
template <typename T>
svtkVariant svtkAbstractArrayGetVariantValue(T* arr, svtkIdType index)
{
  return svtkVariant(arr[index]);
}

//----------------------------------------------------------------------------
svtkVariant svtkAbstractArray::GetVariantValue(svtkIdType valueIdx)
{
  svtkVariant val;
  switch (this->GetDataType())
  {
    svtkExtraExtendedTemplateMacro(val = svtkAbstractArrayGetVariantValue(
                                    static_cast<SVTK_TT*>(this->GetVoidPointer(0)), valueIdx));
  }
  return val;
}

//----------------------------------------------------------------------------
void svtkAbstractArray::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  const char* name = this->GetName();
  if (name)
  {
    os << indent << "Name: " << name << "\n";
  }
  else
  {
    os << indent << "Name: (none)\n";
  }
  os << indent << "Data type: " << this->GetDataTypeAsString() << "\n";
  os << indent << "Size: " << this->Size << "\n";
  os << indent << "MaxId: " << this->MaxId << "\n";
  os << indent << "NumberOfComponents: " << this->NumberOfComponents << endl;
  if (this->ComponentNames)
  {
    os << indent << "ComponentNames: " << endl;
    svtkIndent nextIndent = indent.GetNextIndent();
    for (unsigned int i = 0; i < this->ComponentNames->size(); ++i)
    {
      os << nextIndent << i << " : " << this->ComponentNames->at(i) << endl;
    }
  }
  os << indent << "Information: " << this->Information << endl;
  if (this->Information)
  {
    this->Information->PrintSelf(os, indent.GetNextIndent());
  }
}

//--------------------------------------------------------------------------
void svtkAbstractArray::GetProminentComponentValues(
  int comp, svtkVariantArray* values, double uncertainty, double minimumProminence)
{
  if (!values || comp < -1 || comp >= this->NumberOfComponents)
  {
    return;
  }

  values->Initialize();
  values->SetNumberOfComponents(comp < 0 ? this->NumberOfComponents : 1);

  bool justCreated = false;
  svtkInformation* info = this->GetInformation();
  const double* lastParams = info
    ? (info->Has(DISCRETE_VALUE_SAMPLE_PARAMETERS()) ? info->Get(DISCRETE_VALUE_SAMPLE_PARAMETERS())
                                                     : nullptr)
    : nullptr;
  if (comp >= 0 && info)
  {
    svtkInformationVector* infoVec = info->Get(PER_COMPONENT());
    if (!infoVec || infoVec->GetNumberOfInformationObjects() < this->NumberOfComponents)
    {
      infoVec = svtkInformationVector::New();
      infoVec->SetNumberOfInformationObjects(this->NumberOfComponents);
      info->Set(PER_COMPONENT(), infoVec);
      infoVec->FastDelete();
      justCreated = true;
    }
    info = infoVec->GetInformationObject(comp);
  }
  if (info)
  {
    // Any insane parameter values map to
    // deterministic, exhaustive enumeration of all
    // distinct values:
    if (uncertainty < 0. || uncertainty > 1.)
    {
      uncertainty = 0.;
    }
    if (minimumProminence < 0. || minimumProminence > 1.)
    {
      minimumProminence = 0.;
    }
    // Are parameter values requesting more certainty in reporting or
    // that less-prominent values be reported? If so, recompute.
    bool tighterParams = lastParams
      ? (lastParams[0] > uncertainty || lastParams[1] > minimumProminence ? true : false)
      : true;
    // Recompute discrete value set when the array has been
    // modified since the information was written.
    if (!info->Has(DISCRETE_VALUES()) || tighterParams || this->GetMTime() > info->GetMTime() ||
      justCreated)
    {
      this->UpdateDiscreteValueSet(uncertainty, minimumProminence);
    }
  }
  else
  {
    return;
  }

  svtkIdType len;
  const svtkVariant* vals = info->Get(DISCRETE_VALUES());
  if (vals != nullptr)
  {
    len = info->Length(DISCRETE_VALUES());
    values->SetNumberOfTuples(len / values->GetNumberOfComponents());
    for (svtkIdType i = 0; i < len; ++i)
    {
      values->SetVariantValue(i, vals[i]);
    }
  }
}

//-----------------------------------------------------------------------------
namespace
{
template <typename T>
bool AccumulateSampleValues(T* array, int nc, svtkIdType begin, svtkIdType end,
  std::vector<std::set<T> >& uniques, std::set<std::vector<T> >& tupleUniques,
  unsigned int maxDiscreteValues)
{
  // number of discrete components remaining (tracked during iteration):
  int ndc = nc;
  std::pair<typename std::set<T>::iterator, bool> result;
  std::pair<typename std::set<std::vector<T> >::iterator, bool> tresult;
  std::vector<T> tuple;
  tuple.resize(nc);
  // Here we iterate over the components and add to their respective lists
  // of previously encountered values -- as long as there are not too many
  // values already in the list. We also accumulate each component's value
  // into a svtkVariantArray named tuple, which is added to the list of
  // unique vectors -- again assuming it is not already too long.
  for (svtkIdType i = begin; i < end && ndc; ++i)
  {
    // First, attempt a per-component insert.
    for (int j = 0; j < nc; ++j)
    {
      if (uniques[j].size() > maxDiscreteValues)
        continue;
      T& val(array[i * nc + j]);
      tuple[j] = val;
      result = uniques[j].insert(val);
      if (result.second)
      {
        if (uniques[j].size() == maxDiscreteValues + 1)
        {
          --ndc;
        }
      }
    }
    // Now, as long as no component has exceeded maxDiscreteValues unique
    // values, it is worth seeing whether the tuple as a whole is unique:
    if (nc > 1 && ndc == nc)
    {
      tresult = tupleUniques.insert(tuple);
      (void)tresult; // nice to have when debugging.
    }
  }
  return ndc == 0;
}

//-----------------------------------------------------------------------------
template <typename U>
void SampleProminentValues(std::vector<std::vector<svtkVariant> >& uniques, svtkIdType maxId, int nc,
  svtkIdType nt, int blockSize, svtkIdType numberOfBlocks, U* ptr, unsigned int maxDiscreteValues)
{
  std::vector<std::set<U> > typeSpecificUniques;
  std::set<std::vector<U> > typeSpecificUniqueTuples;
  typeSpecificUniques.resize(nc);
  // I. Accumulate samples for all components plus the tuple,
  //    either for the full array or a random subset.
  if (numberOfBlocks * blockSize > maxId / 2)
  { // Awwww, just do the whole array already!
    AccumulateSampleValues(
      ptr, nc, 0, nt, typeSpecificUniques, typeSpecificUniqueTuples, maxDiscreteValues);
  }
  else
  { // Choose random blocks
    svtkNew<svtkMinimalStandardRandomSequence> seq;
    // test different blocks each time we're called:
    seq->SetSeed(static_cast<int>(seq->GetMTime()) ^ 0xdeadbeef);
    svtkIdType totalBlockCount = nt / blockSize + (nt % blockSize ? 1 : 0);
    std::set<svtkIdType> startTuples;
    // Sort the list of blocks we'll search to maintain cache coherence.
    for (int i = 0; i < numberOfBlocks; ++i, seq->Next())
    {
      svtkIdType startTuple = static_cast<svtkIdType>(seq->GetValue() * totalBlockCount) * blockSize;
      startTuples.insert(startTuple);
    }
    // Now iterate over the blocks, accumulating unique values and tuples.
    std::set<svtkIdType>::iterator blkIt;
    for (blkIt = startTuples.begin(); blkIt != startTuples.end(); ++blkIt)
    {
      svtkIdType startTuple = *blkIt;
      svtkIdType endTuple = startTuple + blockSize;
      endTuple = endTuple < nt ? endTuple : nt;
      bool endEarly = AccumulateSampleValues(ptr, nc, startTuple, endTuple, typeSpecificUniques,
        typeSpecificUniqueTuples, maxDiscreteValues);
      if (endEarly)
        break;
    }
  }

  // II. Convert type-specific sets of unique values into non-type-specific
  //     vectors of svtkVariants for storage in array information.

  // Handle per-component uniques first
  for (int i = 0; i < nc; ++i)
  {
    std::back_insert_iterator<std::vector<svtkVariant> > bi(uniques[i]);
    std::copy(typeSpecificUniques[i].begin(), typeSpecificUniques[i].end(), bi);
  }

  // Now squash any tuple-wide uniques into
  // the final entry of the outer vector.
  typename std::set<std::vector<U> >::iterator si;
  for (si = typeSpecificUniqueTuples.begin(); si != typeSpecificUniqueTuples.end(); ++si)
  {
    std::back_insert_iterator<std::vector<svtkVariant> > bi(uniques[nc]);
    std::copy(si->begin(), si->end(), bi);
  }
}
} // End anonymous namespace.

//-----------------------------------------------------------------------------
void svtkAbstractArray::UpdateDiscreteValueSet(double uncertainty, double minimumProminence)
{
  // For an array with T tuples and given uncertainty U and mininumum
  // prominence P, we sample N blocks of M tuples each, with
  // M*N = f(T; P, U) and f some sublinear function of T.
  // If every component plus all components taken together each have more than
  // MaxDiscreteValues distinct values, then we exit early.
  // M is chosen based on the number of bytes per tuple to maximize use of a
  // cache line (assuming a 64-byte cache line until kwsys::SystemInformation
  // or the like can provide a platform-independent way to query it).
  //
  // N is chosen to satisfy the requested uncertainty and prominence criteria
  // specified.
#define SVTK_CACHE_LINE_SIZE 64
#define SVTK_SAMPLE_FACTOR 5
  // I. Determine the granularity at which the array should be sampled.
  //int numberOfComponentsWithProminentValues = 0;
  int nc = this->NumberOfComponents;
  int blockSize = SVTK_CACHE_LINE_SIZE / (this->GetDataTypeSize() * nc);
  if (!blockSize)
  {
    blockSize = 4;
  }
  double logfac = 1.;
  svtkIdType nt = this->GetNumberOfTuples();
  svtkIdType numberOfSampleTuples = nt;
  if (this->MaxId > 0 && minimumProminence > 0.0)
  {
    logfac = -log(uncertainty * minimumProminence) / minimumProminence;
    if (logfac < 0)
    {
      logfac = -logfac;
    }
    if (!svtkMath::IsInf(logfac))
    {
      numberOfSampleTuples = static_cast<svtkIdType>(SVTK_SAMPLE_FACTOR * logfac);
    }
  }
  /*
  // Theoretically, we should discard values or tuples that recur fewer
  // than minFreq times in our sample, but in practice this involves
  // counting and communication that slow us down.
  svtkIdType minFreq = static_cast<svtkIdType>(
    numberOfSampleTuples * minimumProminence / 2);
    */
  svtkIdType numberOfBlocks =
    numberOfSampleTuples / blockSize + (numberOfSampleTuples % blockSize ? 1 : 0);
  if (static_cast<unsigned int>(numberOfBlocks * blockSize) < 2 * this->MaxDiscreteValues)
  {
    numberOfBlocks =
      2 * this->MaxDiscreteValues / blockSize + (2 * this->MaxDiscreteValues % blockSize ? 1 : 0);
  }
  // II. Sample the array.
  std::vector<std::vector<svtkVariant> > uniques(nc > 1 ? nc + 1 : nc);
  switch (this->GetDataType())
  {
    svtkSuperExtraExtendedTemplateMacro(
      SampleProminentValues(uniques, this->MaxId, nc, nt, blockSize, numberOfBlocks,
        static_cast<SVTK_TT*>(this->GetVoidPointer(0)), this->MaxDiscreteValues));
    default:
      svtkErrorMacro("Array type " << this->GetClassName() << " not supported.");
      break;
  }

  // III. Store the results in the array's svtkInformation.
  int c;
  svtkInformationVector* iv;
  for (c = 0; c < nc; ++c)
  {
    if (uniques[c].size() && uniques[c].size() <= this->MaxDiscreteValues)
    {
      //++numberOfComponentsWithProminentValues;
      iv = this->GetInformation()->Get(PER_COMPONENT());
      if (!iv)
      {
        svtkNew<svtkInformationVector> infoVec;
        infoVec->SetNumberOfInformationObjects(this->NumberOfComponents);
        this->GetInformation()->Set(PER_COMPONENT(), infoVec);
        iv = this->GetInformation()->Get(PER_COMPONENT());
      }
      iv->GetInformationObject(c)->Set(
        DISCRETE_VALUES(), &uniques[c][0], static_cast<int>(uniques[c].size()));
    }
    else
    {
      iv = this->GetInformation()->Get(PER_COMPONENT());
      if (iv)
      {
        iv->GetInformationObject(c)->Remove(DISCRETE_VALUES());
      }
    }
  }
  if (nc > 1 && uniques[nc].size() <= this->MaxDiscreteValues * nc)
  {
    //++numberOfComponentsWithProminentValues;
    this->GetInformation()->Set(
      DISCRETE_VALUES(), &uniques[nc][0], static_cast<int>(uniques[nc].size()));
  }
  else
  { // Remove the key
    this->GetInformation()->Remove(DISCRETE_VALUES());
  }

  // Always store the sample parameters; this lets us know not to
  // re-run the sampling algorithm.
  double params[2];
  params[0] = uncertainty;
  params[1] = minimumProminence;
  this->GetInformation()->Set(DISCRETE_VALUE_SAMPLE_PARAMETERS(), params, 2);
}
