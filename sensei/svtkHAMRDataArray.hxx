#ifndef svtkHAMRDataArray_hxx
#define svtkHAMRDataArray_hxx

#include <svtkCallbackCommand.h>
#include <svtkCommand.h>

/** Delete callback. In order to safely zero-copy data from a svtkHAMRDataArray
 * into a svtkAOSDataArrayTemplate the svtkAOSDataArrayTemplate needs to hold a
 * reference to svtkHAMRDataArray.  This reference can be released when the
 * svtkAOSDataArrayTemplate is deleted.  We will register this callback for the
 * delete event with SVTK. The clinet data will the the svtkHAMRDataArray
 * instance.
 */
static inline
void svtkHAMRDataArrayDelete(svtkObject *, unsigned long, void *clientData, void *)
{
    svtkObject *heldRef = (svtkObject*)clientData;
    heldRef->Delete();
}




// --------------------------------------------------------------------------
template<typename T>
svtkHAMRDataArray<T>::svtkHAMRDataArray() : Data(nullptr)
{
}

// --------------------------------------------------------------------------
template<typename T>
svtkHAMRDataArray<T>::~svtkHAMRDataArray()
{
  delete this->Data;
}

// --------------------------------------------------------------------------
template<typename T>
svtkHAMRDataArray<T> *svtkHAMRDataArray<T>::New()
{
  return new svtkHAMRDataArray<T>;
}

// --------------------------------------------------------------------------
template<typename T>
svtkHAMRDataArray<T> *svtkHAMRDataArray<T>::New(svtkDataArray *da)
{
  return svtkHAMRDataArray<T>::New(da, svtkAllocator::malloc,
           svtkStream(), svtkStreamMode::sync_host);
}

// --------------------------------------------------------------------------
template<typename T>
svtkHAMRDataArray<T> *svtkHAMRDataArray<T>::New(svtkDataArray *da,
  svtkAllocator alloc,  svtkStream stream, svtkStreamMode streamMode)
{
  if (!da)
  {
    std::cerr << __FILE__ << ":" << __LINE__ << std::endl
      << "ERROR: Can't copy construct a nullptr" << std::endl;
    return nullptr;
  }

  long numTups = da->GetNumberOfTuples();
  long numComps = da->GetNumberOfComponents();
  const char *name = da->GetName();

  switch (da->GetDataType())
  {
    svtkTemplateMacro(
      if (dynamic_cast<svtkHAMRDataArray<SVTK_TT>*>(da))
      {
        auto tDa = static_cast<svtkHAMRDataArray<SVTK_TT>*>(da);

        svtkHAMRDataArray<T> *inst = new svtkHAMRDataArray<T>;

        inst->Data = new hamr::buffer<T>(alloc, stream, streamMode, *tDa->Data);
        inst->MaxId = numTups - 1;
        inst->NumberOfComponents = numComps;
        inst->SetName(name);

        return inst;
      }
      else if (dynamic_cast<svtkAOSDataArrayTemplate<SVTK_TT>*>(da))
      {
        auto tDa = static_cast<svtkAOSDataArrayTemplate<SVTK_TT>*>(da);

        svtkHAMRDataArray<T> *inst = new svtkHAMRDataArray<T>;

        inst->Data = new hamr::buffer<T>(alloc, stream, streamMode, numTups*numComps);
        inst->Data->set(0, tDa->GetPointer(0), 0, numTups);
        inst->MaxId = numTups - 1;
        inst->NumberOfComponents = numComps;
        inst->SetName(name);

        return inst;
      }
    );
    default:
    {
      std::cerr << __FILE__ << ":" << __LINE__ << std::endl
        << "ERROR: Unsupported array type " << da->GetClassName() << std::endl;
    }
  }
  return nullptr;
}

// --------------------------------------------------------------------------
template<typename T>
svtkHAMRDataArray<T> *svtkHAMRDataArray<T>::New(const std::string &name,
  size_t numTuples, int numComps, svtkAllocator alloc, svtkStream stream,
  svtkStreamMode streamMode)
{
  assert(numComps > 0);

  svtkHAMRDataArray<T> *tmp = new svtkHAMRDataArray<T>;
  tmp->Data = new hamr::buffer<T>(alloc, stream, streamMode, numTuples*numComps);
  tmp->MaxId = numTuples - 1;
  tmp->NumberOfComponents = numComps;
  tmp->SetName(name.c_str());

  return tmp;
}

// --------------------------------------------------------------------------
template<typename T>
svtkHAMRDataArray<T> *svtkHAMRDataArray<T>::New(const std::string &name,
  size_t numTuples, int numComps, svtkAllocator alloc, svtkStream stream,
  svtkStreamMode streamMode, const T &initVal)
{
  assert(numComps > 0);

  svtkHAMRDataArray<T> *tmp = new svtkHAMRDataArray<T>;
  tmp->Data = new hamr::buffer<T>(alloc, stream, streamMode, numTuples*numComps, initVal);
  tmp->MaxId = numTuples - 1;
  tmp->NumberOfComponents = numComps;
  tmp->SetName(name.c_str());

  return tmp;
}

// --------------------------------------------------------------------------
template<typename T>
svtkHAMRDataArray<T> *svtkHAMRDataArray<T>::New(const std::string &name, T *data,
  size_t numTuples, int numComps, svtkAllocator alloc, svtkStream stream,
  svtkStreamMode streamMode, int owner, int take)
{
  assert(numComps > 0);

  svtkHAMRDataArray<T> *tmp = new svtkHAMRDataArray<T>;

  if (take)
  {
    tmp->Data = new hamr::buffer<T>(alloc, stream, streamMode,
                  numTuples*numComps, owner, data);
  }
  else
  {
    std::shared_ptr<T> dataPtr(data, [](void*){});

    tmp->Data = new hamr::buffer<T>(alloc, stream, streamMode,
                  numTuples*numComps, owner, dataPtr);
  }

  tmp->MaxId = numTuples - 1;
  tmp->NumberOfComponents = numComps;
  tmp->SetName(name.c_str());

  return tmp;
}

// --------------------------------------------------------------------------
template<typename T>
svtkHAMRDataArray<T> *svtkHAMRDataArray<T>::New(const std::string &name,
  const std::shared_ptr<T> &data, size_t numTuples, int numComps,
  svtkAllocator alloc, svtkStream stream, svtkStreamMode streamMode,
  int owner)
{
  assert(numComps > 0);

  svtkHAMRDataArray<T> *tmp = new svtkHAMRDataArray<T>;

  tmp->Data = new hamr::buffer<T>(alloc, stream, streamMode,
                numTuples*numComps, owner, data);

  tmp->MaxId = numTuples - 1;
  tmp->NumberOfComponents = numComps;
  tmp->SetName(name.c_str());

  return tmp;
}

// --------------------------------------------------------------------------
template<typename T>
template <typename deleter_t>
svtkHAMRDataArray<T> *svtkHAMRDataArray<T>::New(const std::string &name,
  T *data, size_t numTuples, int numComps, svtkAllocator alloc,
  svtkStream stream, svtkStreamMode streamMode, int owner, deleter_t deleter)
{
  assert(numComps > 0);

  svtkHAMRDataArray<T> *tmp = new svtkHAMRDataArray<T>;

  tmp->Data = new hamr::buffer<T>(alloc, stream, streamMode,
                numTuples*numComps, owner, data, deleter);

  tmp->MaxId = numTuples - 1;
  tmp->SetNumberOfComponents(numComps);
  tmp->SetName(name.c_str());

  return tmp;
}

// --------------------------------------------------------------------------
template<typename T>
void svtkHAMRDataArray<T>::SetData(T *data, size_t numTuples,
  int numComps, svtkAllocator alloc, svtkStream stream,
  svtkStreamMode streamMode, int owner)
{
  assert(numComps > 0);

  delete this->Data;

  this->Data = new hamr::buffer<T>(alloc, stream, streamMode,
                 numTuples*numComps, owner, data);

  this->MaxId = numTuples - 1;
  this->NumberOfComponents = numComps;
}

// --------------------------------------------------------------------------
template<typename T>
void svtkHAMRDataArray<T>::SetData(const std::shared_ptr<T> &data,
    size_t numTuples, int numComps, svtkAllocator alloc, svtkStream stream,
    svtkStreamMode streamMode, int owner)
{
  assert(numComps > 0);

  delete this->Data;

  this->Data = new hamr::buffer<T>(alloc, stream, streamMode,
                 numTuples*numComps, owner, data);

  this->MaxId = numTuples - 1;
  this->NumberOfComponents = numComps;
}

// --------------------------------------------------------------------------
template<typename T>
template <typename deleter_t>
void svtkHAMRDataArray<T>::SetData(T *data, size_t numTuples, int numComps,
  svtkAllocator alloc, svtkStream stream, svtkStreamMode streamMode,
  int owner, deleter_t deleter)
{
  assert(numComps > 0);

  delete this->Data;

  this->Data = new hamr::buffer<T>(alloc, stream, streamMode,
                 numTuples*numComps, owner, data, deleter);

  this->MaxId = numTuples - 1;
  this->SetNumberOfComponents(numComps);
}

// TODO - in the zero-copy case below the VTK array should hold a reference to this
#define AsSvtkDataArrayImpl(_cls_t, _cpp_t)                     \
  if (std::is_same<_cls_t, _cpp_t>::value)                      \
  {                                                             \
    int nComps = this->GetNumberOfComponents();                 \
    size_t nTups = this->GetNumberOfTuples();                   \
                                                                \
    svtkAOSDataArrayTemplate<_cpp_t> *tmp =                     \
      svtkAOSDataArrayTemplate<_cpp_t>::New();                  \
                                                                \
    tmp->SetNumberOfComponents(nComps);                         \
    tmp->SetName(this->GetName());                              \
                                                                \
    if  (this->Data->host_accessible())                          \
    {                                                           \
      /* zero-copy from CPU to VTK */                           \
      tmp->SetVoidArray(this->Data->data(), nTups, 1, 0);       \
    }                                                           \
    else                                                        \
    {                                                           \
      /* deep-copy from the GPU to VTK */                       \
      tmp->SetNumberOfTuples(nTups);                            \
      T *ptr = (T*)tmp->GetVoidPointer(0);                      \
      this->Data->get(0, ptr, 0, nTups*nComps);                 \
    }                                                           \
                                                                \
    return tmp;                                                 \
  }

// --------------------------------------------------------------------------
template<typename T>
svtkAOSDataArrayTemplate<T> *svtkHAMRDataArray<T>::AsSvtkAOSDataArray(int zeroCopy)
{
  int nComps = this->GetNumberOfComponents();
  size_t nTups = this->GetNumberOfTuples();

  svtkAOSDataArrayTemplate<T> *tmp =
    svtkAOSDataArrayTemplate<T>::New();

  tmp->SetNumberOfComponents(nComps);
  tmp->SetName(this->GetName());

  if  (zeroCopy && this->Data->host_accessible())
  {
    // zero-copy from HAMR to VTK can only occur when data is accessible on the
    // CPU since VTK arrays are CPU only
    tmp->SetVoidArray(this->Data->data(), nTups, 1, 0);

    // hold a reference to the HAMR data array.
    this->Register(nullptr);

    // register a callback to release the held reference when the SVTK array
    // signals it is finished
    svtkCallbackCommand *cc = svtkCallbackCommand::New();
    cc->SetCallback(svtkHAMRDataArrayDelete);
    cc->SetClientData(this);

    tmp->AddObserver(svtkCommand::DeleteEvent, cc);
    cc->Delete();
  }
  else
  {
    // the data is not accessible on the CPU or a deep-copy weas requested
    tmp->SetNumberOfTuples(nTups);
    T *ptr = (T*)tmp->GetVoidPointer(0);
    this->Data->get(0, ptr, 0, nTups*nComps);
  }

  return tmp;
}

// --------------------------------------------------------------------------
template<typename T>
svtkArrayIterator* svtkHAMRDataArray<T>::NewIterator()
{
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
  return nullptr;
}

// --------------------------------------------------------------------------
template<typename T>
int svtkHAMRDataArray<T>::GetDataType() const
{
  if (std::is_same<T, double>::value)
  {
    return SVTK_DOUBLE;
  }
  else if (std::is_same<T, float>::value)
  {
    return SVTK_FLOAT;
  }
  else if (std::is_same<T, long long>::value)
  {
    return SVTK_LONG_LONG;
  }
  else if (std::is_same<T, long>::value)
  {
    return SVTK_LONG;
  }
  else if (std::is_same<T, int>::value)
  {
    return SVTK_INT;
  }
  else if (std::is_same<T, short>::value)
  {
    return SVTK_SHORT;
  }
  else if (std::is_same<T, char>::value)
  {
    return SVTK_CHAR;
  }
  else if (std::is_same<T, unsigned long long>::value)
  {
    return SVTK_UNSIGNED_LONG_LONG;
  }
  else if (std::is_same<T, unsigned long>::value)
  {
    return SVTK_UNSIGNED_LONG;
  }
  else if (std::is_same<T, unsigned int>::value)
  {
    return SVTK_UNSIGNED_INT;
  }
  else if (std::is_same<T, unsigned short>::value)
  {
    return SVTK_UNSIGNED_SHORT;
  }
  else if (std::is_same<T, unsigned char>::value)
  {
    return SVTK_UNSIGNED_CHAR;
  }
  else
  {
    svtkErrorMacro("No SVTK type for T");
    abort();
    return -1;
  }
}

// --------------------------------------------------------------------------
template<typename T>
int svtkHAMRDataArray<T>::GetDataTypeSize() const
{
  return sizeof(T);
}

/*
// --------------------------------------------------------------------------
template<typename T>
bool svtkHAMRDataArray<T>::HasStandardMemoryLayout() const
{
  return false;
}
*/
/*
// --------------------------------------------------------------------------
template<typename T>
svtkTypeBool svtkHAMRDataArray<T>::Allocate(svtkIdType size, svtkIdType ext)
{
  (void) ext;
  if (this->Data)
  {
    delete this->Data;
  }

  this->Data = new hamr::buffer<T>(hamr::buffer_allocator::malloc, size);

  return true;
}
*/

// --------------------------------------------------------------------------
template<typename T>
svtkTypeBool svtkHAMRDataArray<T>::Resize(svtkIdType numTuples)
{
  this->Data->resize(numTuples*this->NumberOfComponents);
  return true;
}

/*
// --------------------------------------------------------------------------
template<typename T>
void svtkHAMRDataArray<T>::SetNumberOfComponents(int numComps)
{
  this->NumberOfComponents = numComps;
}
*/

// --------------------------------------------------------------------------
template<typename T>
void svtkHAMRDataArray<T>::SetNumberOfTuples(svtkIdType numTuples, svtkAllocator alloc)
{
  if (!this->Data)
  {
    this->Data = new hamr::buffer<T>(alloc, numTuples*this->NumberOfComponents);
  }
  else
  {
    this->Data->move(alloc);
    this->Data->resize(numTuples);
  }

  this->MaxId = numTuples - 1;
}

// --------------------------------------------------------------------------
template<typename T>
void svtkHAMRDataArray<T>::SetNumberOfTuples(svtkIdType numTuples)
{
  this->SetNumberOfTuples(numTuples, hamr::buffer_allocator::malloc);
}






// --------------------------------------------------------------------------
template <typename T>
svtkTypeBool svtkHAMRDataArray<T>::Allocate(svtkIdType numValues, svtkIdType ext)
{
  (void)numValues;
  (void)ext;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
  return false;
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::Initialize()
{
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

/*
// --------------------------------------------------------------------------
template <typename T>
int svtkHAMRDataArray<T>::GetDataType() const
{
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
  return 0;
}


// --------------------------------------------------------------------------
template <typename T>
int svtkHAMRDataArray<T>::GetDataTypeSize() const
{
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
  return 0;
}
*/

// --------------------------------------------------------------------------
template <typename T>
int svtkHAMRDataArray<T>::GetElementComponentSize() const
{
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
  return 0;
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::SetTuple(svtkIdType dstTupleIdx, svtkIdType srcTupleIdx, svtkAbstractArray* source)
{
  (void)dstTupleIdx;
  (void)srcTupleIdx;
  (void)source;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::InsertTuple(svtkIdType dstTupleIdx, svtkIdType srcTupleIdx, svtkAbstractArray* source)
{
  (void)dstTupleIdx;
  (void)srcTupleIdx;
  (void)source;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::InsertTuples(svtkIdList* dstIds, svtkIdList* srcIds, svtkAbstractArray* source)
{
  (void)dstIds;
  (void)srcIds;
  (void)source;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::InsertTuples(svtkIdType dstStart, svtkIdType n, svtkIdType srcStart, svtkAbstractArray* source)
{
  (void)dstStart;
  (void)n;
  (void)srcStart;
  (void)source;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
svtkIdType svtkHAMRDataArray<T>::InsertNextTuple(svtkIdType srcTupleIdx, svtkAbstractArray* source)
{
  (void)srcTupleIdx;
  (void)source;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::GetTuples(svtkIdList* tupleIds, svtkAbstractArray* output)
{
  (void)tupleIds;
  (void)output;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::GetTuples(svtkIdType p1, svtkIdType p2, svtkAbstractArray* output)
{
  (void)p1;
  (void)p2;
  (void)output;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
bool svtkHAMRDataArray<T>::HasStandardMemoryLayout() const
{
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
  return true;
}

// --------------------------------------------------------------------------
template <typename T>
void* svtkHAMRDataArray<T>::GetVoidPointer(svtkIdType valueIdx)
{
  /* could be handled in the case data is host accessible like this, but would
   * fail if that's not the case so we'll disallow this for now
  if (this->Data->host_accessible())
    return this->Data->data() + valueIdx;
  */
  (void)valueIdx;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
  return nullptr;
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::DeepCopy(svtkAbstractArray* da)
{
  (void)da;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::InterpolateTuple(svtkIdType dstTupleIdx, svtkIdList* ptIndices, svtkAbstractArray* source, double* weights)
{
  (void)dstTupleIdx;
  (void)ptIndices;
  (void)source;
  (void)weights;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::InterpolateTuple(svtkIdType dstTupleIdx, svtkIdType srcTupleIdx1, svtkAbstractArray* source1, svtkIdType srcTupleIdx2, svtkAbstractArray* source2, double t)
{
  (void)dstTupleIdx;
  (void)srcTupleIdx1;
  (void)source1;
  (void)srcTupleIdx2;
  (void)source2;
  (void)t;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::Squeeze()
{
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::SetVoidArray(void* array, svtkIdType size, int save)
{
  (void)array;
  (void)size;
  (void)save;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::SetVoidArray(void* array, svtkIdType size, int save, int deleteMethod)
{
  (void)array;
  (void)size;
  (void)save;
  (void)deleteMethod;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::SetArrayFreeFunction(void (*callback)(void*))
{
  (void)callback;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::ExportToVoidPointer(void* out_ptr)
{
  (void)out_ptr;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
unsigned long svtkHAMRDataArray<T>::GetActualMemorySize() const
{
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
  return 0;
}

// --------------------------------------------------------------------------
template <typename T>
int svtkHAMRDataArray<T>::IsNumeric() const
{
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
  return 0;
}

// --------------------------------------------------------------------------
template <typename T>
svtkIdType svtkHAMRDataArray<T>::LookupValue(svtkVariant value)
{
  (void)value;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::LookupValue(svtkVariant value, svtkIdList* valueIds)
{
  (void)value;
  (void)valueIds;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
svtkVariant svtkHAMRDataArray<T>::GetVariantValue(svtkIdType valueIdx)
{
  (void)valueIdx;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::InsertVariantValue(svtkIdType valueIdx, svtkVariant value)
{
  (void)valueIdx;
  (void)value;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::SetVariantValue(svtkIdType valueIdx, svtkVariant value)
{
  (void)valueIdx;
  (void)value;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::DataChanged()
{
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::ClearLookup()
{
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::GetProminentComponentValues(int comp, svtkVariantArray* values, double uncertainty, double minimumProminence)
{
  (void)comp;
  (void)values;
  (void)uncertainty;
  (void)minimumProminence;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
int svtkHAMRDataArray<T>::CopyInformation(svtkInformation* infoFrom, int deep)
{
  (void)infoFrom;
  (void)deep;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
  return 0;
}

// --------------------------------------------------------------------------
template <typename T>
double* svtkHAMRDataArray<T>::GetTuple(svtkIdType tupleIdx)
{
  (void)tupleIdx;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
  return nullptr;
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::GetTuple(svtkIdType tupleIdx, double* tuple)
{
  (void)tupleIdx;
  (void)tuple;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::SetTuple(svtkIdType tupleIdx, const float* tuple)
{
  (void)tupleIdx;
  (void)tuple;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::SetTuple(svtkIdType tupleIdx, const double* tuple)
{
  (void)tupleIdx;
  (void)tuple;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::InsertTuple(svtkIdType tupleIdx, const float* tuple)
{
  (void)tupleIdx;
  (void)tuple;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::InsertTuple(svtkIdType tupleIdx, const double* tuple)
{
  (void)tupleIdx;
  (void)tuple;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
svtkIdType svtkHAMRDataArray<T>::InsertNextTuple(const float* tuple)
{
  (void)tuple;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
  return 0;
}

// --------------------------------------------------------------------------
template <typename T>
svtkIdType svtkHAMRDataArray<T>::InsertNextTuple(const double* tuple)
{
  (void)tuple;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::RemoveTuple(svtkIdType tupleIdx)
{
  (void)tupleIdx;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::RemoveLastTuple()
{
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
double svtkHAMRDataArray<T>::GetComponent(svtkIdType tupleIdx, int compIdx)
{
  (void)tupleIdx;
  (void)compIdx;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
  return 0.;
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::SetComponent(svtkIdType tupleIdx, int compIdx, double value)
{
  (void)tupleIdx;
  (void)compIdx;
  (void)value;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::InsertComponent(svtkIdType tupleIdx, int compIdx, double value)
{
  (void)tupleIdx;
  (void)compIdx;
  (void)value;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::GetData( svtkIdType tupleMin, svtkIdType tupleMax, int compMin, int compMax, svtkDoubleArray* data)
{
  (void)tupleMin;
  (void)tupleMax;
  (void)compMin;
  (void)compMax;
  (void)data;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::DeepCopy(svtkDataArray* da)
{
  (void)da;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::ShallowCopy(svtkDataArray* other)
{
  (void)other;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::FillComponent(int compIdx, double value)
{
  (void)compIdx;
  (void)value;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::Fill(double value)
{
  (void)value;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::CopyComponent(int dstComponent, svtkDataArray* src, int srcComponent)
{
  (void)dstComponent;
  (void)src;
  (void)srcComponent;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
void* svtkHAMRDataArray<T>::WriteVoidPointer(svtkIdType valueIdx, svtkIdType numValues)
{
  (void)valueIdx;
  (void)numValues;
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
double svtkHAMRDataArray<T>::GetMaxNorm()
{
  svtkErrorMacro(<< "Method \"" <<  __FUNCTION__ << "\" not implemented.");
  abort();
}

// --------------------------------------------------------------------------
template <typename T>
int svtkHAMRDataArray<T>::GetArrayType() const
{
  return DataArray;
}

// ----------------------------------------------------------------------------
template <typename T>
void svtkHAMRDataArray<T>::PrintSelf(ostream& os, svtkIndent indent)
{
  (void) os;
  (void) indent;

  std::cerr << "this->Data = ";
  if (this->Data)
  {
    this->Data->print();
  }
  else
  {
    std::cerr << "nullptr";
  }
}

#endif
