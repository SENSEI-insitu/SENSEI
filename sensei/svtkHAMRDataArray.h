#ifndef svtkHAMRDataArray_h
#define svtkHAMRDataArray_h

#if defined(SENSEI_ENABLE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "hamr_buffer.h"
#include "hamr_buffer_allocator.h"
#include "hamr_buffer_transfer.h"
#include "hamr_stream.h"

#include "svtkDataArray.h"
#include "svtkAOSDataArrayTemplate.h"
#include "svtkCommonCoreModule.h"          // For export macro

#include <cstdint>
#include <string>

using svtkAllocator = hamr::buffer_allocator;
using svtkStreamMode = hamr::buffer_transfer;
using svtkStream = hamr::stream;


/// get the allocator type most suitable for the current build configuration
inline svtkAllocator GetDeviceAllocator() { return hamr::get_device_allocator(); }
inline svtkAllocator GetHostAllocator() { return hamr::get_host_allocator(); }
inline svtkAllocator GetCPUAllocator() { return hamr::buffer_allocator::malloc; }


template <typename T>
class SVTKCOMMONCORE_EXPORT svtkHAMRDataArray : public svtkDataArray
{
public:
  svtkTypeMacro(svtkHAMRDataArray, svtkDataArray);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /// allocate a new empty array
  static svtkHAMRDataArray *New();

  /// copy construct from the passed instance. this is a deep copy.
  static svtkHAMRDataArray *New(svtkDataArray *da);

  /// copy construct from the passed instance. this is a deep copy.
  static svtkHAMRDataArray *New(svtkDataArray *da, svtkAllocator alloc,
                              svtkStream stream, svtkStreamMode streamMode);

  /** zero-copy the passed data.
   * @param[in] name the name of the array
   * @param[in] data a pointer to the data
   * @param[in] numTuples the number of data tuples
   * @param[in] numComps the numper of components per tuple
   * @param[in] alloc an ::svtkAllocator instance declaring where the data resides
   * @param[in] stream an ::svtkStream instance providing an odering on operations
   * @param[in] streamMode an ::svtkStreamMode instance declaring synchronous behavior or not
   * @param[in] owner the device id where the data resides, or -1 for the CPU
   * @param[in] take if true the ponited to data will be released using the
   *                 deleter associated with the declared ::svtkAllocator alloc
   * @returns a new instance that must be deleted by the caller
   */
  static svtkHAMRDataArray *New(const std::string &name, T *data,
    size_t numTuples, int numComps, svtkAllocator alloc, svtkStream stream,
    svtkStreamMode streamMode, int owner, int take);

  /** zero-copy the passed data. This override gives one direct control over the
   * management and reference counting of the pointed to data.
   * @param[in] name the name of the array
   * @param[in] data a smart pointer that manages the pointed to data
   * @param[in] numTuples the number of data tuples
   * @param[in] numComps the numper of components per tuple
   * @param[in] alloc an ::svtkAllocator instance declaring where the data resides
   * @param[in] stream an ::svtkStream instance providing an odering on operations
   * @param[in] streamMode an ::svtkStreamMode instance declaring synchronous behavior or not
   * @param[in] owner the device id where the data resides, or -1 for the CPU
   * @returns a new instance that must be deleted by the caller
   */
  static svtkHAMRDataArray *New(const std::string &name,
    const std::shared_ptr<T> &data, size_t numTuples, int numComps,
    svtkAllocator alloc, svtkStream stream, svtkStreamMode streamMode,
    int owner);

  /** zero-copy the passed data. This override gives one direct control over the
   * method that is used to release the ponited to array.
   * @param[in] name the name of the array
   * @param[in] data a smart pointer that manages the pointed to data
   * @param[in] numTuples the number of data tuples
   * @param[in] numComps the numper of components per tuple
   * @param[in] alloc an ::svtkAllocator instance declaring where the data resides
   * @param[in] stream an ::svtkStream instance providing an odering on operations
   * @param[in] streamMode an ::svtkStreamMode instance declaring synchronous behavior or not
   * @param[in] owner the device id where the data resides, or -1 for the CPU
   * @returns a new instance that must be deleted by the caller
   */
  template <typename deleter_t>
  static svtkHAMRDataArray *New(const std::string &name, T *data, size_t numTuples,
    int numComps, svtkAllocator alloc, svtkStream stream, svtkStreamMode streamMode,
    int owner, deleter_t deleter);

  /** Allocate a new array of the specified size using the specified allocator
   * @param[in] name the name of the array
   * @param[in] numTuples the number of data tuples
   * @param[in] numComps the numper of components per tuple
   * @param[in] alloc an ::svtkAllocator instance declaring where the data resides
   * @param[in] stream an ::svtkStream instance providing an odering on operations
   * @param[in] streamMode an ::svtkStreamMode instance declaring synchronous behavior or not
   * @returns a new instance that must be deleted by the caller
   */
  static svtkHAMRDataArray *New(const std::string &name,
    size_t numTuples, int numComps, svtkAllocator alloc, svtkStream stream,
    svtkStreamMode streamMode);

  /** Allocate a new array of the specified size using the specified allocator
   * initialized to the specified value
   * @param[in] name the name of the array
   * @param[in] numTuples the number of data tuples
   * @param[in] numComps the numper of components per tuple
   * @param[in] alloc an ::svtkAllocator instance declaring where the data resides
   * @param[in] stream an ::svtkStream instance providing an odering on operations
   * @param[in] streamMode an ::svtkStreamMode instance declaring synchronous behavior or not
   * @param[in] initVal the value to initialize the contents to
   * @returns a new instance that must be deleted by the caller
   */
  static svtkHAMRDataArray *New(const std::string &name,
    size_t numTuples, int numComps, svtkAllocator alloc, svtkStream stream,
    svtkStreamMode streamMode, const T &initVal);

  /** Convert to an svtkAOSDataArrayTemplate instance. Because SVTK only
   * supports CPU based data, a deep-copy is made when this array is located on
   * the GPU. Otherwise the data is passed via zero-copy
   * @param[in] zeroCopy if true and the data resides on the CPU, the data is
   *                     passed to the new svtkAOSDataArrayTemplate instance by
   *                     zero-copy. Otehrwise a deep-copy is made.
   * @returns a new instance that must be deleted by the caller
   */
  svtkAOSDataArrayTemplate<T> *AsSvtkAOSDataArray(int zeroCopy);

  /** Sets or changes the allocator used to manage the menory, this may move
   * the data from one device to another
   */
  void SetAllocator(svtkAllocator alloc)
  {
    this->Data->move(alloc);
  }

  /// sets the stream. mode indicate synchronous behavior or not.
  void SetStream(const svtkStream &stream, svtkStreamMode &mode)
  {
    this->Data->set_stream(stream, mode);
  }

  /// @eturns the stream
  svtkStream &GetStream() { return this->Data->get_stream(); }
  const svtkStream &GetStream() const { return this->Data->get_stream(); }

  /* copy contents of the passed in data. allocator and owner tells the current
   * location of the passed data. Use ::Resize to allocate space.
   */
  //void CopyData(size_t destStart, T *srcData, size_t numTuples,
  //  svtkAllocator srcAlloc = svtkAllocator::malloc, int owner = -1);

  /* append the contents of the passed in data. allocator and owner tells the current
   * location of the passed data.
   */
  //void AppendData(T *srcData, size_t numTuples,
  //  svtkAllocator srcAlloc = svtkAllocator::malloc, int owner = -1);

  /** zero-copy the passed data. the allocator is used to tell where the data
   * resides. the callee (array instance) takes ownership of the pointer.
   */
  void SetData(T *data, size_t numTuples, int numComps, svtkAllocator alloc,
    svtkStream stream, svtkStreamMode streamMode, int owner);

  /** zero-copy the passed data. the allocator is used to tell where the data
   * resides.
   */
  void SetData(const std::shared_ptr<T> &data, size_t numTuples, int numComps,
    svtkAllocator alloc, svtkStream stream, svtkStreamMode, int owner);

  /** zero-copy the passed data. the allocator is used to tell where the data
   * resides the deleter will be called as void deleter(void *dataPtr) when the
   * data is no longer needed
   */
  template <typename deleter_t>
  void SetData(T *dataPtr, size_t numTuples, int numComps,
    svtkAllocator alloc, svtkStream stream, svtkStreamMode,
    int owner, deleter_t deleter);

  /// returns a pointer to the data that is safe to use on the CPU
  std::shared_ptr<const T> GetCPUAccessible() const { return this->Data->get_cpu_accessible(); }

  /// returns a pointer to the data that is safe for the compiled device
  std::shared_ptr<const T> GetDeviceAccessible() const { return this->Data->get_device_accessible(); }

  /// returns a pointer to the data that is safe to use with CUDA
  std::shared_ptr<const T> GetCUDAAccessible() const { return this->Data->get_cuda_accessible(); }

  /// returns a pointer to the data that is safe to use with HIP
  std::shared_ptr<const T> GetHIPAccessible() const { return this->Data->get_hip_accessible(); }

  /// returns a pointer to the data that is safe to use with OpenMP device off load
  std::shared_ptr<const T> GetOpenMPAccessible() const { return this->Data->get_openmp_accessible(); }

  /// return true if a pooniter to the data is safe to use on the CPU
  bool CPUAccessible() { return this->Data->cpu_accessible(); }

  /// return true if a pooniter to the data is safe to use with CUDA
  bool CUDAAccessible() { return this->Data->cuda_accessible(); }

  /// returns a pointer to the data that is safe to use with HIP
  bool HIPAccessible() { return this->Data->hip_accessible(); }

  /// return true if a pooniter to the data is safe to use with OpenMP device off load
  bool OpenMPAccessible() { return this->Data->openmp_accessible(); }

  /** fast access to the internally managed memory. Use this only when you know
   * where the data resides and will access it in that location. This method
   * saves the cost of a smart_ptr copy construct and the cost of the logic
   * that determines if a temporary is needed. For all other cases use
   * GetXAccessible to access the data.
   */
  T *GetData() { return this->Data->data(); }

  /** fast access to the internally managed memory. Use this only when you know
   * where the data resides and will access it in that location. This method
   * saves the cost of the logic determining if a temporary is needed. For all
   * other cases use GetXAccessible to access the data.
   */
  std::shared_ptr<T> GetDataPointer() { return this->Data->pointer(); }

  /// returns the number of values. this is the current size, not the capacity.
  svtkIdType GetNumberOfValues() const override { return this->Data->size(); }

  /// sets the current size and may change the capacity of the array
  void SetNumberOfTuples(svtkIdType numTuples) override;
  void SetNumberOfTuples(svtkIdType numTuples, svtkAllocator alloc);

  /// resize the container using the current allocator
  svtkTypeBool Resize(svtkIdType numTuples) override;

  /** @name not implemented
   * These methods are not impelemented. If called they will abort. If you find
   * yourself needing these then you are likely writing VTK code and should
   * convert the svtkHAMRDataArray into a svtkDataArray subclass.
   */
  ///@{
  svtkTypeBool Allocate(svtkIdType numValues, svtkIdType ext) override;

  void Initialize() override;

  int GetDataType() const override;

  int GetDataTypeSize() const override;

  int GetElementComponentSize() const override;

  void SetTuple(svtkIdType dstTupleIdx, svtkIdType srcTupleIdx, svtkAbstractArray* source) override;

  void InsertTuple(svtkIdType dstTupleIdx, svtkIdType srcTupleIdx, svtkAbstractArray* source) override;

  void InsertTuples(svtkIdList* dstIds, svtkIdList* srcIds, svtkAbstractArray* source) override;

  void InsertTuples(svtkIdType dstStart, svtkIdType n, svtkIdType srcStart, svtkAbstractArray* source) override;

  svtkIdType InsertNextTuple(svtkIdType srcTupleIdx, svtkAbstractArray* source) override;

  void GetTuples(svtkIdList* tupleIds, svtkAbstractArray* output) override;

  void GetTuples(svtkIdType p1, svtkIdType p2, svtkAbstractArray* output) override;

  bool HasStandardMemoryLayout() const override;

  void* GetVoidPointer(svtkIdType valueIdx) override;

  void DeepCopy(svtkAbstractArray* da) override;

  void InterpolateTuple(svtkIdType dstTupleIdx, svtkIdList* ptIndices, svtkAbstractArray* source, double* weights) override;

  void InterpolateTuple(svtkIdType dstTupleIdx, svtkIdType srcTupleIdx1, svtkAbstractArray* source1, svtkIdType srcTupleIdx2, svtkAbstractArray* source2, double t) override;

  void Squeeze() override;

  void SetVoidArray(void* svtkNotUsed(array), svtkIdType svtkNotUsed(size), int svtkNotUsed(save)) override;

  void SetVoidArray(void* ptr, svtkIdType size, int save, int deleteMethod) override;

  void SetArrayFreeFunction(void (*callback)(void*)) override;

  void ExportToVoidPointer(void* out_ptr) override;

  unsigned long GetActualMemorySize() const override;

  int IsNumeric() const override;

  SVTK_NEWINSTANCE svtkArrayIterator* NewIterator() override;

  svtkIdType LookupValue(svtkVariant value) override;
  void LookupValue(svtkVariant value, svtkIdList* valueIds) override;
  svtkVariant GetVariantValue(svtkIdType valueIdx) override;

  void InsertVariantValue(svtkIdType valueIdx, svtkVariant value) override;

  void SetVariantValue(svtkIdType valueIdx, svtkVariant value) override;

  void DataChanged() override;

  void ClearLookup() override;

  void GetProminentComponentValues(int comp, svtkVariantArray* values, double uncertainty = 1.e-6, double minimumProminence = 1.e-3) override;

  int CopyInformation(svtkInformation* infoFrom, int deep = 1) override;

  double* GetTuple(svtkIdType tupleIdx) override;

  void GetTuple(svtkIdType tupleIdx, double* tuple) override;

  void SetTuple(svtkIdType tupleIdx, const float* tuple) override;
  void SetTuple(svtkIdType tupleIdx, const double* tuple) override;

  void InsertTuple(svtkIdType tupleIdx, const float* tuple)  override;
  void InsertTuple(svtkIdType tupleIdx, const double* tuple)  override;

  svtkIdType InsertNextTuple(const float* tuple) override;
  svtkIdType InsertNextTuple(const double* tuple) override;

  void RemoveTuple(svtkIdType tupleIdx) override;
  void RemoveFirstTuple() override { this->RemoveTuple(0); }
  void RemoveLastTuple() override;

  double GetComponent(svtkIdType tupleIdx, int compIdx) override;

  void SetComponent(svtkIdType tupleIdx, int compIdx, double value) override;

  void InsertComponent(svtkIdType tupleIdx, int compIdx, double value) override;

  void GetData(svtkIdType tupleMin, svtkIdType tupleMax, int compMin, int compMax, svtkDoubleArray* data) override;

  void DeepCopy(svtkDataArray* da) override;

  void ShallowCopy(svtkDataArray* other) override;

  void FillComponent(int compIdx, double value) override;

  void Fill(double value) override;

  void CopyComponent(int dstComponent, svtkDataArray* src, int srcComponent) override;

  void* WriteVoidPointer(svtkIdType valueIdx, svtkIdType numValues) override;

  double GetMaxNorm() override;

  int GetArrayType() const override;
  //@}

//////////////////////////// from generic data array

  /** Get a pointer to the data. The data must already be on the CPU. Call
   * CPUAccessible() instead. */
  /*void* GetVoidPointer(svtkIdType valueIdx) override
  {
    if (!this->Data->cpu_accessible())
    {
      svtkErrroMacro("Accessing a device pointer on the CPU."
        " Call CpuAccessible() instead to move the data.")
      return nullptr;
    }
    return this->Data->data() + valueIdx;
  }*/


protected:
  svtkHAMRDataArray();
  ~svtkHAMRDataArray() override;

private:
  hamr::buffer<T> *Data;

private:
  svtkHAMRDataArray(const svtkHAMRDataArray&) = delete;
  void operator=(const svtkHAMRDataArray&) = delete;
};

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
           svtkStream(), svtkStreamMode::sync_cpu);
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
        inst->NumberOfComponnets = numComps;
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
    if  (this->Data->cpu_accessible())                          \
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
  (void) zeroCopy; // TODO : set the deleter
  AsSvtkDataArrayImpl(T, double)
  else AsSvtkDataArrayImpl(T, float)
  else AsSvtkDataArrayImpl(T, char)
  else AsSvtkDataArrayImpl(T, short)
  else AsSvtkDataArrayImpl(T, int)
  else AsSvtkDataArrayImpl(T, long)
  else AsSvtkDataArrayImpl(T, long long)
  else AsSvtkDataArrayImpl(T, unsigned char)
  else AsSvtkDataArrayImpl(T, unsigned short)
  else AsSvtkDataArrayImpl(T, unsigned int)
  else AsSvtkDataArrayImpl(T, unsigned long)
  else AsSvtkDataArrayImpl(T, unsigned long long)
  else
  {
    svtkErrorMacro("No SVTK type for T");
    abort();
  }
  return nullptr;
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

using svtkHAMRDoubleArray = svtkHAMRDataArray<double>;
using svtkHAMRFloatArray = svtkHAMRDataArray<float>;
using svtkHAMRCharArray = svtkHAMRDataArray<char>;
using svtkHAMRShortArray = svtkHAMRDataArray<short>;
using svtkHAMRIntArray = svtkHAMRDataArray<int>;
using svtkHAMRLongArray = svtkHAMRDataArray<long>;
using svtkHAMRLongLongArray = svtkHAMRDataArray<long long>;
using svtkHAMRUnsignedCharArray = svtkHAMRDataArray<unsigned char>;
using svtkHAMRUnsignedShortArray = svtkHAMRDataArray<unsigned short>;
using svtkHAMRUnsignedIntArray = svtkHAMRDataArray<unsigned int>;
using svtkHAMRUnsignedLongArray = svtkHAMRDataArray<unsigned long>;
using svtkHAMRUnsignedLongLongArray = svtkHAMRDataArray<unsigned long long>;

#endif
