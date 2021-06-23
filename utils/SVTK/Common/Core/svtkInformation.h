/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformation.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformation
 * @brief   Store svtkAlgorithm input/output information.
 *
 * svtkInformation represents information and/or data for one input or
 * one output of a svtkAlgorithm.  It maps from keys to values of
 * several data types.  Instances of this class are collected in
 * svtkInformationVector instances and passed to
 * svtkAlgorithm::ProcessRequest calls.  The information and
 * data referenced by the instance on a particular input or output
 * define the request made to the svtkAlgorithm instance.
 */

#ifndef svtkInformation_h
#define svtkInformation_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

#include <string> // for std::string compat

class svtkDataObject;
class svtkExecutive;
class svtkInformationDataObjectKey;
class svtkInformationDoubleKey;
class svtkInformationDoubleVectorKey;
class svtkInformationExecutivePortKey;
class svtkInformationExecutivePortVectorKey;
class svtkInformationIdTypeKey;
class svtkInformationInformationKey;
class svtkInformationInformationVectorKey;
class svtkInformationIntegerKey;
class svtkInformationIntegerPointerKey;
class svtkInformationIntegerVectorKey;
class svtkInformationInternals;
class svtkInformationKey;
class svtkInformationKeyToInformationFriendship;
class svtkInformationKeyVectorKey;
class svtkInformationObjectBaseKey;
class svtkInformationObjectBaseVectorKey;
class svtkInformationRequestKey;
class svtkInformationStringKey;
class svtkInformationStringVectorKey;
class svtkInformationUnsignedLongKey;
class svtkInformationVariantKey;
class svtkInformationVariantVectorKey;
class svtkInformationVector;
class svtkVariant;

class SVTKCOMMONCORE_EXPORT svtkInformation : public svtkObject
{
public:
  static svtkInformation* New();
  svtkTypeMacro(svtkInformation, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  void PrintKeys(ostream& os, svtkIndent indent);

  /**
   * Modified signature with no arguments that calls Modified
   * on svtkObject superclass.
   */
  void Modified() override;

  /**
   * Modified signature that takes an information key as an argument.
   * Sets the new MTime and invokes a modified event with the
   * information key as call data.
   */
  void Modified(svtkInformationKey* key);

  /**
   * Clear all information entries.
   */
  void Clear();

  /**
   * Return the number of keys in this information object (as would be returned
   * by iterating over the keys).
   */
  int GetNumberOfKeys();

  /**
   * Copy all information entries from the given svtkInformation
   * instance.  Any previously existing entries are removed.  If
   * deep==1, a deep copy of the information structure is performed (new
   * instances of any contained svtkInformation and svtkInformationVector
   * objects are created).
   */
  void Copy(svtkInformation* from, int deep = 0);

  /**
   * Append all information entries from the given svtkInformation
   * instance. If deep==1, a deep copy of the information structure is performed
   * (new instances of any contained svtkInformation and svtkInformationVector
   * objects are created).
   */
  void Append(svtkInformation* from, int deep = 0);

  //@{
  /**
   * Copy the key/value pair associated with the given key in the
   * given information object.  If deep=1, a deep copy of the information
   * structure is performed (new instances of any contained svtkInformation and
   * svtkInformationVector objects are created).
   */
  void CopyEntry(svtkInformation* from, svtkInformationKey* key, int deep = 0);
  void CopyEntry(svtkInformation* from, svtkInformationDataObjectKey* key, int deep = 0);
  void CopyEntry(svtkInformation* from, svtkInformationDoubleVectorKey* key, int deep = 0);
  void CopyEntry(svtkInformation* from, svtkInformationVariantKey* key, int deep = 0);
  void CopyEntry(svtkInformation* from, svtkInformationVariantVectorKey* key, int deep = 0);
  void CopyEntry(svtkInformation* from, svtkInformationInformationKey* key, int deep = 0);
  void CopyEntry(svtkInformation* from, svtkInformationInformationVectorKey* key, int deep = 0);
  void CopyEntry(svtkInformation* from, svtkInformationIntegerKey* key, int deep = 0);
  void CopyEntry(svtkInformation* from, svtkInformationIntegerVectorKey* key, int deep = 0);
  void CopyEntry(svtkInformation* from, svtkInformationObjectBaseVectorKey* key, int deep = 0);
  void CopyEntry(svtkInformation* from, svtkInformationRequestKey* key, int deep = 0);
  void CopyEntry(svtkInformation* from, svtkInformationStringKey* key, int deep = 0);
  void CopyEntry(svtkInformation* from, svtkInformationStringVectorKey* key, int deep = 0);
  void CopyEntry(svtkInformation* from, svtkInformationUnsignedLongKey* key, int deep = 0);
  //@}

  /**
   * Use the given key to lookup a list of other keys in the given
   * information object.  The key/value pairs associated with these
   * other keys will be copied.  If deep==1, a deep copy of the
   * information structure is performed.
   */
  void CopyEntries(svtkInformation* from, svtkInformationKeyVectorKey* key, int deep = 0);

  /**
   * Check whether the given key appears in this information object.
   */
  int Has(svtkInformationKey* key);

  /**
   * Remove the given key and its data from this information object.
   */
  void Remove(svtkInformationKey* key);

  //@{
  /**
   * Get/Set a request-valued entry.
   */
  void Set(svtkInformationRequestKey* key);
  void Remove(svtkInformationRequestKey* key);
  int Has(svtkInformationRequestKey* key);
  //@}

  //@{
  /**
   * Get/Set an integer-valued entry.
   */
  void Set(svtkInformationIntegerKey* key, int value);
  int Get(svtkInformationIntegerKey* key);
  void Remove(svtkInformationIntegerKey* key);
  int Has(svtkInformationIntegerKey* key);
  //@}

  //@{
  /**
   * Get/Set a svtkIdType-valued entry.
   */
  void Set(svtkInformationIdTypeKey* key, svtkIdType value);
  svtkIdType Get(svtkInformationIdTypeKey* key);
  void Remove(svtkInformationIdTypeKey* key);
  int Has(svtkInformationIdTypeKey* key);
  //@}

  //@{
  /**
   * Get/Set an double-valued entry.
   */
  void Set(svtkInformationDoubleKey* key, double value);
  double Get(svtkInformationDoubleKey* key);
  void Remove(svtkInformationDoubleKey* key);
  int Has(svtkInformationDoubleKey* key);
  //@}

  //@{
  /**
   * Get/Set an variant-valued entry.
   */
  void Set(svtkInformationVariantKey* key, const svtkVariant& value);
  const svtkVariant& Get(svtkInformationVariantKey* key);
  void Remove(svtkInformationVariantKey* key);
  int Has(svtkInformationVariantKey* key);
  //@}

  //@{
  /**
   * Get/Set an integer-vector-valued entry.
   */
  void Append(svtkInformationIntegerVectorKey* key, int value);
  void Set(svtkInformationIntegerVectorKey* key, const int* value, int length);
  void Set(svtkInformationIntegerVectorKey* key, int value1, int value2, int value3);
  void Set(svtkInformationIntegerVectorKey* key, int value1, int value2, int value3, int value4,
    int value5, int value6);
  int* Get(svtkInformationIntegerVectorKey* key);
  int Get(svtkInformationIntegerVectorKey* key, int idx);
  void Get(svtkInformationIntegerVectorKey* key, int* value);
  int Length(svtkInformationIntegerVectorKey* key);
  void Remove(svtkInformationIntegerVectorKey* key);
  int Has(svtkInformationIntegerVectorKey* key);
  //@}

  //@{
  /**
   * Get/Set a string-vector-valued entry.
   */
  void Append(svtkInformationStringVectorKey* key, const char* value);
  void Set(svtkInformationStringVectorKey* key, const char* value, int idx = 0);
  void Append(svtkInformationStringVectorKey* key, const std::string& value);
  void Set(svtkInformationStringVectorKey* key, const std::string& value, int idx = 0);
  const char* Get(svtkInformationStringVectorKey* key, int idx = 0);
  int Length(svtkInformationStringVectorKey* key);
  void Remove(svtkInformationStringVectorKey* key);
  int Has(svtkInformationStringVectorKey* key);
  //@}

  //@{
  /**
   * Get/Set an integer-pointer-valued entry.
   */
  void Set(svtkInformationIntegerPointerKey* key, int* value, int length);
  int* Get(svtkInformationIntegerPointerKey* key);
  void Get(svtkInformationIntegerPointerKey* key, int* value);
  int Length(svtkInformationIntegerPointerKey* key);
  void Remove(svtkInformationIntegerPointerKey* key);
  int Has(svtkInformationIntegerPointerKey* key);
  //@}

  //@{
  /**
   * Get/Set an unsigned-long-valued entry.
   */
  void Set(svtkInformationUnsignedLongKey* key, unsigned long value);
  unsigned long Get(svtkInformationUnsignedLongKey* key);
  void Remove(svtkInformationUnsignedLongKey* key);
  int Has(svtkInformationUnsignedLongKey* key);
  //@}

  //@{
  /**
   * Get/Set an double-vector-valued entry.
   */
  void Append(svtkInformationDoubleVectorKey* key, double value);
  void Set(svtkInformationDoubleVectorKey* key, const double* value, int length);
  void Set(svtkInformationDoubleVectorKey* key, double value1, double value2, double value3);
  void Set(svtkInformationDoubleVectorKey* key, double value1, double value2, double value3,
    double value4, double value5, double value6);
  double* Get(svtkInformationDoubleVectorKey* key);
  double Get(svtkInformationDoubleVectorKey* key, int idx);
  void Get(svtkInformationDoubleVectorKey* key, double* value);
  int Length(svtkInformationDoubleVectorKey* key);
  void Remove(svtkInformationDoubleVectorKey* key);
  int Has(svtkInformationDoubleVectorKey* key);
  //@}

  //@{
  /**
   * Get/Set an variant-vector-valued entry.
   */
  void Append(svtkInformationVariantVectorKey* key, const svtkVariant& value);
  void Set(svtkInformationVariantVectorKey* key, const svtkVariant* value, int length);
  void Set(svtkInformationVariantVectorKey* key, const svtkVariant& value1, const svtkVariant& value2,
    const svtkVariant& value3);
  void Set(svtkInformationVariantVectorKey* key, const svtkVariant& value1, const svtkVariant& value2,
    const svtkVariant& value3, const svtkVariant& value4, const svtkVariant& value5,
    const svtkVariant& value6);
  const svtkVariant* Get(svtkInformationVariantVectorKey* key);
  const svtkVariant& Get(svtkInformationVariantVectorKey* key, int idx);
  void Get(svtkInformationVariantVectorKey* key, svtkVariant* value);
  int Length(svtkInformationVariantVectorKey* key);
  void Remove(svtkInformationVariantVectorKey* key);
  int Has(svtkInformationVariantVectorKey* key);
  //@}

  //@{
  /**
   * Get/Set an InformationKey-vector-valued entry.
   */
  void Append(svtkInformationKeyVectorKey* key, svtkInformationKey* value);
  void AppendUnique(svtkInformationKeyVectorKey* key, svtkInformationKey* value);
  void Set(svtkInformationKeyVectorKey* key, svtkInformationKey* const* value, int length);
  void Remove(svtkInformationKeyVectorKey* key, svtkInformationKey* value);
  svtkInformationKey** Get(svtkInformationKeyVectorKey* key);
  svtkInformationKey* Get(svtkInformationKeyVectorKey* key, int idx);
  void Get(svtkInformationKeyVectorKey* key, svtkInformationKey** value);
  int Length(svtkInformationKeyVectorKey* key);
  void Remove(svtkInformationKeyVectorKey* key);
  int Has(svtkInformationKeyVectorKey* key);
  //@}

  // Provide extra overloads of this method to avoid requiring user
  // code to include the headers for these key types.  Avoid wrapping
  // them because the original method can be called from the wrappers
  // anyway and this causes a python help string to be too long.

  void Append(svtkInformationKeyVectorKey* key, svtkInformationDataObjectKey* value);
  void Append(svtkInformationKeyVectorKey* key, svtkInformationDoubleKey* value);
  void Append(svtkInformationKeyVectorKey* key, svtkInformationDoubleVectorKey* value);
  void Append(svtkInformationKeyVectorKey* key, svtkInformationInformationKey* value);
  void Append(svtkInformationKeyVectorKey* key, svtkInformationInformationVectorKey* value);
  void Append(svtkInformationKeyVectorKey* key, svtkInformationIntegerKey* value);
  void Append(svtkInformationKeyVectorKey* key, svtkInformationIntegerVectorKey* value);
  void Append(svtkInformationKeyVectorKey* key, svtkInformationStringKey* value);
  void Append(svtkInformationKeyVectorKey* key, svtkInformationStringVectorKey* value);
  void Append(svtkInformationKeyVectorKey* key, svtkInformationObjectBaseKey* value);
  void Append(svtkInformationKeyVectorKey* key, svtkInformationUnsignedLongKey* value);

  void AppendUnique(svtkInformationKeyVectorKey* key, svtkInformationDataObjectKey* value);
  void AppendUnique(svtkInformationKeyVectorKey* key, svtkInformationDoubleKey* value);
  void AppendUnique(svtkInformationKeyVectorKey* key, svtkInformationDoubleVectorKey* value);
  void AppendUnique(svtkInformationKeyVectorKey* key, svtkInformationInformationKey* value);
  void AppendUnique(svtkInformationKeyVectorKey* key, svtkInformationInformationVectorKey* value);
  void AppendUnique(svtkInformationKeyVectorKey* key, svtkInformationIntegerKey* value);
  void AppendUnique(svtkInformationKeyVectorKey* key, svtkInformationIntegerVectorKey* value);
  void AppendUnique(svtkInformationKeyVectorKey* key, svtkInformationStringKey* value);
  void AppendUnique(svtkInformationKeyVectorKey* key, svtkInformationStringVectorKey* value);
  void AppendUnique(svtkInformationKeyVectorKey* key, svtkInformationObjectBaseKey* value);
  void AppendUnique(svtkInformationKeyVectorKey* key, svtkInformationUnsignedLongKey* value);

  //@{
  /**
   * Get/Set a string-valued entry.
   */
  void Set(svtkInformationStringKey* key, const char*);
  void Set(svtkInformationStringKey* key, const std::string&);
  const char* Get(svtkInformationStringKey* key);
  void Remove(svtkInformationStringKey* key);
  int Has(svtkInformationStringKey* key);
  //@}

  //@{
  /**
   * Get/Set an entry storing another svtkInformation instance.
   */
  void Set(svtkInformationInformationKey* key, svtkInformation*);
  svtkInformation* Get(svtkInformationInformationKey* key);
  void Remove(svtkInformationInformationKey* key);
  int Has(svtkInformationInformationKey* key);
  //@}

  //@{
  /**
   * Get/Set an entry storing a svtkInformationVector instance.
   */
  void Set(svtkInformationInformationVectorKey* key, svtkInformationVector*);
  svtkInformationVector* Get(svtkInformationInformationVectorKey* key);
  void Remove(svtkInformationInformationVectorKey* key);
  int Has(svtkInformationInformationVectorKey* key);
  //@}

  //@{
  /**
   * Get/Set an entry storing a svtkObjectBase instance.
   */
  void Set(svtkInformationObjectBaseKey* key, svtkObjectBase*);
  svtkObjectBase* Get(svtkInformationObjectBaseKey* key);
  void Remove(svtkInformationObjectBaseKey* key);
  int Has(svtkInformationObjectBaseKey* key);
  //@}

  //@{
  /**
   * Manipulate a ObjectBaseVector entry.
   */
  void Append(svtkInformationObjectBaseVectorKey* key, svtkObjectBase* data);
  void Set(svtkInformationObjectBaseVectorKey* key, svtkObjectBase* value, int idx = 0);
  svtkObjectBase* Get(svtkInformationObjectBaseVectorKey* key, int idx = 0);
  int Length(svtkInformationObjectBaseVectorKey* key);
  void Remove(svtkInformationObjectBaseVectorKey* key);
  void Remove(svtkInformationObjectBaseVectorKey* key, svtkObjectBase* objectToRemove);
  void Remove(svtkInformationObjectBaseVectorKey* key, int indexToRemove);
  int Has(svtkInformationObjectBaseVectorKey* key);
  //@}

  //@{
  /**
   * Get/Set an entry storing a svtkDataObject instance.
   */
  void Set(svtkInformationDataObjectKey* key, svtkDataObject SVTK_WRAP_EXTERN*);
  svtkDataObject SVTK_WRAP_EXTERN* Get(svtkInformationDataObjectKey* key);
  void Remove(svtkInformationDataObjectKey* key);
  int Has(svtkInformationDataObjectKey* key);
  //@}

  //@{
  /**
   * Upcast the given key instance.
   */
  static svtkInformationKey* GetKey(svtkInformationDataObjectKey* key);
  static svtkInformationKey* GetKey(svtkInformationDoubleKey* key);
  static svtkInformationKey* GetKey(svtkInformationDoubleVectorKey* key);
  static svtkInformationKey* GetKey(svtkInformationInformationKey* key);
  static svtkInformationKey* GetKey(svtkInformationInformationVectorKey* key);
  static svtkInformationKey* GetKey(svtkInformationIntegerKey* key);
  static svtkInformationKey* GetKey(svtkInformationIntegerVectorKey* key);
  static svtkInformationKey* GetKey(svtkInformationRequestKey* key);
  static svtkInformationKey* GetKey(svtkInformationStringKey* key);
  static svtkInformationKey* GetKey(svtkInformationStringVectorKey* key);
  static svtkInformationKey* GetKey(svtkInformationKey* key);
  static svtkInformationKey* GetKey(svtkInformationUnsignedLongKey* key);
  static svtkInformationKey* GetKey(svtkInformationVariantKey* key);
  static svtkInformationKey* GetKey(svtkInformationVariantVectorKey* key);
  //@}

  //@{
  /**
   * Initiate garbage collection when a reference is removed.
   */
  void Register(svtkObjectBase* o) override;
  void UnRegister(svtkObjectBase* o) override;
  //@}

  //@{
  /**
   * Get/Set the Request ivar
   */
  void SetRequest(svtkInformationRequestKey* request);
  svtkInformationRequestKey* GetRequest();
  //@}

protected:
  svtkInformation();
  ~svtkInformation() override;

  // Get/Set a map entry directly through the svtkObjectBase instance
  // representing the value.  Used internally to manage the map.
  void SetAsObjectBase(svtkInformationKey* key, svtkObjectBase* value);
  const svtkObjectBase* GetAsObjectBase(const svtkInformationKey* key) const;
  svtkObjectBase* GetAsObjectBase(svtkInformationKey* key);

  // Internal implementation details.
  svtkInformationInternals* Internal;

  // Garbage collection support.
  void ReportReferences(svtkGarbageCollector*) override;

  // Report the object associated with the given key to the collector.
  void ReportAsObjectBase(svtkInformationKey* key, svtkGarbageCollector* collector);

private:
  friend class svtkInformationKeyToInformationFriendship;
  friend class svtkInformationIterator;

private:
  svtkInformation(const svtkInformation&) = delete;
  void operator=(const svtkInformation&) = delete;
  svtkInformationRequestKey* Request;
};

#endif
// SVTK-HeaderTest-Exclude: svtkInformation.h
