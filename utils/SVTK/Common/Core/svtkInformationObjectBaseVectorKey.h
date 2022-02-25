/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationObjectBaseVectorKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationObjectBaseVectorKey
 * @brief   Key for svtkObjectBase vector values.
 *
 * svtkInformationObjectBaseVectorKey is used to represent keys for double
 * vector values in svtkInformation.h. NOTE the interface in this key differs
 * from that in other similar keys because of our internal use of smart
 * pointers.
 */

#ifndef svtkInformationObjectBaseVectorKey_h
#define svtkInformationObjectBaseVectorKey_h

#include "svtkCommonCoreModule.h"            // For export macro
#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.
#include "svtkInformationKey.h"

class svtkInformationObjectBaseVectorValue;

class SVTKCOMMONCORE_EXPORT svtkInformationObjectBaseVectorKey : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationObjectBaseVectorKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@{
  /**
   * The name of the static instance and the class in which
   * it is defined(location) should be passed to the constructor.
   * Providing "requiredClass" name one can insure that only
   * objects of type "requiredClass" are stored in vectors
   * associated with the instance of this key type created.
   * These should be string literals as they are not copied.
   */
  svtkInformationObjectBaseVectorKey(
    const char* name, const char* location, const char* requiredClass = nullptr);
  //
  ~svtkInformationObjectBaseVectorKey() override;
  //@}

  /**
   * This method simply returns a new svtkInformationObjectBaseVectorKey, given a
   * name, location and optionally a required class (a classname to restrict
   * which class types can be set with this key). This method is provided
   * for wrappers. Use the constructor directly from C++ instead.
   */
  static svtkInformationObjectBaseVectorKey* MakeKey(
    const char* name, const char* location, const char* requiredClass = nullptr)
  {
    return new svtkInformationObjectBaseVectorKey(name, location, requiredClass);
  }

  /**
   * Clear the vector.
   */
  void Clear(svtkInformation* info);

  /**
   * Resize (extend) the vector to hold n objects. Any new elements
   * created will be null initialized.
   */
  void Resize(svtkInformation* info, int n);

  /**
   * Get the vector's length.
   */
  int Size(svtkInformation* info);
  int Length(svtkInformation* info) { return this->Size(info); }

  /**
   * Put the value on the back of the vector, with ref counting.
   */
  void Append(svtkInformation* info, svtkObjectBase* value);

  /**
   * Set element i of the vector to value. Resizes the vector
   * if needed.
   */
  void Set(svtkInformation* info, svtkObjectBase* value, int i);

  //@{
  /**
   * Remove all instances of val from the list. If using the indexed overload,
   * the object at the specified position is removed.
   */
  void Remove(svtkInformation* info, svtkObjectBase* val);
  void Remove(svtkInformation* info, int idx);
  using Superclass::Remove; // Don't hide base class methods
  //@}

  /**
   * Copy n values from the range in source defined by [from  from+n-1]
   * into the range in this vector defined by [to to+n-1]. Resizes
   * the vector if needed.
   */
  void SetRange(svtkInformation* info, svtkObjectBase** source, int from, int to, int n);

  /**
   * Copy n values from the range in this vector defined by [from  from+n-1]
   * into the range in the destination vector defined by [to to+n-1]. Up
   * to you to make sure the destination is big enough.
   */
  void GetRange(svtkInformation* info, svtkObjectBase** dest, int from, int to, int n);

  /**
   * Get the svtkObjectBase at a specific location in the vector.
   */
  svtkObjectBase* Get(svtkInformation* info, int idx);

  // _escription:
  // Get a pointer to the first svtkObjectBase in the vector. We are
  // uysing a vector of smart pointers so this is not easy to
  // implement.
  // svtkObjectBase **Get(svtkInformation* info);

  /**
   * Copy the entry associated with this key from one information
   * object to another.  If there is no entry in the first information
   * object for this key, the value is removed from the second.
   */
  void ShallowCopy(svtkInformation* from, svtkInformation* to) override;

  /**
   * Print the key's value in an information object to a stream.
   */
  void Print(ostream& os, svtkInformation* info) override;

protected:
  // The type required of all objects stored with this key.
  const char* RequiredClass;

private:
  /**
   * Used to create the underlying vector that will be associated
   * with this key.
   */
  void CreateObjectBase();
  /**
   * Check insures that if RequiredClass is set then the
   * type of aValue matches. return true if the two match.
   */
  bool ValidateDerivedType(svtkInformation* info, svtkObjectBase* aValue);
  /**
   * Get the vector associated with this key, if there is
   * none then associate a new vector with this key and return
   * that.
   */
  svtkInformationObjectBaseVectorValue* GetObjectBaseVector(svtkInformation* info);

  //
  svtkInformationObjectBaseVectorKey(const svtkInformationObjectBaseVectorKey&) = delete;
  void operator=(const svtkInformationObjectBaseVectorKey&) = delete;
};

#endif
