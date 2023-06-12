#ifndef svtkHAMRDataArray_h
#define svtkHAMRDataArray_h

#include "senseiConfig.h"

#include "hamr_buffer.h"
#include "hamr_buffer_allocator.h"
#include "hamr_buffer_transfer.h"
#include "hamr_stream.h"

#include "svtkDataArray.h"
#include "svtkAOSDataArrayTemplate.h"
#include "svtkCommonCoreModule.h"

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
class SENSEI_EXPORT svtkHAMRDataArray : public svtkDataArray
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
   * @param[in] owner the device id where the data resides, or -1 for the host
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
   * @param[in] owner the device id where the data resides, or -1 for the host
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
   * @param[in] owner the device id where the data resides, or -1 for the host
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
   * supports host based data, a deep-copy is made when this array is located on
   * the GPU. Otherwise the data is passed via zero-copy
   * @param[in] zeroCopy if true and the data resides on the host, the data is
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

  /// synchronizes the internal stream
  void Synchronize() const { this->Data->synchronize(); }

  /// returns a pointer to the data that is safe to use on the host
  std::shared_ptr<const T> GetHostAccessible() const { return this->Data->get_host_accessible(); }

  /// returns a pointer to the data that is safe for the compiled device
  std::shared_ptr<const T> GetDeviceAccessible() const { return this->Data->get_device_accessible(); }

  /// returns a pointer to the data that is safe to use with CUDA
  std::shared_ptr<const T> GetCUDAAccessible() const { return this->Data->get_cuda_accessible(); }

  /// returns a pointer to the data that is safe to use with HIP
  std::shared_ptr<const T> GetHIPAccessible() const { return this->Data->get_hip_accessible(); }

  /// returns a pointer to the data that is safe to use with OpenMP device off load
  std::shared_ptr<const T> GetOpenMPAccessible() const { return this->Data->get_openmp_accessible(); }

  /// return true if a pooniter to the data is safe to use on the Host
  bool HostAccessible() { return this->Data->host_accessible(); }

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

protected:
  svtkHAMRDataArray();
  ~svtkHAMRDataArray() override;

private:
  hamr::buffer<T> *Data;

private:
  svtkHAMRDataArray(const svtkHAMRDataArray&) = delete;
  void operator=(const svtkHAMRDataArray&) = delete;

  template<typename U> friend class svtkHAMRDataArray;
};

#if !defined(SENSEI_SEPARATE_IMPL)
#include "svtkHAMRDataArray.hxx"
#endif

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
