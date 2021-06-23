/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationQuadratureSchemeDefinitionVectorKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationQuadratureSchemeDefinitionVectorKey
 * @brief   Key for svtkQuadratureSchemeDefinition vector values.
 *
 * svtkInformationQuadratureSchemeDefinitionVectorKey is used to represent keys for double
 * vector values in svtkInformation.h. NOTE the interface in this key differs
 * from that in other similar keys because of our internal use of smart
 * pointers.
 */

#ifndef svtkInformationQuadratureSchemeDefinitionVectorKey_h
#define svtkInformationQuadratureSchemeDefinitionVectorKey_h

#include "svtkCommonDataModelModule.h"       // For export macro
#include "svtkCommonInformationKeyManager.h" // Manage instances of this type.
#include "svtkInformationKey.h"

class svtkInformationQuadratureSchemeDefinitionVectorValue;
class svtkXMLDataElement;
class svtkQuadratureSchemeDefinition;

class SVTKCOMMONDATAMODEL_EXPORT svtkInformationQuadratureSchemeDefinitionVectorKey
  : public svtkInformationKey
{
public:
  svtkTypeMacro(svtkInformationQuadratureSchemeDefinitionVectorKey, svtkInformationKey);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@{
  /**
   * The name of the static instance and the class in which
   * it is defined(location) should be passed to the constructor.
   */
  svtkInformationQuadratureSchemeDefinitionVectorKey(const char* name, const char* location);
  //
  ~svtkInformationQuadratureSchemeDefinitionVectorKey() override;
  //@}

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
   * Put the value on the back of the vector, with reference counting.
   */
  void Append(svtkInformation* info, svtkQuadratureSchemeDefinition* value);
  /**
   * Set element i of the vector to value. Resizes the vector
   * if needed.
   */
  void Set(svtkInformation* info, svtkQuadratureSchemeDefinition* value, int i);
  /**
   * Copy n values from the range in source defined by [from  from+n-1]
   * into the range in this vector defined by [to to+n-1]. Resizes
   * the vector if needed.
   */
  void SetRange(
    svtkInformation* info, svtkQuadratureSchemeDefinition** source, int from, int to, int n);

  /**
   * Copy n values from the range in this vector defined by [from  from+n-1]
   * into the range in the destination vector defined by [to to+n-1]. Up
   * to you to make sure the destination is big enough.
   */
  void GetRange(
    svtkInformation* info, svtkQuadratureSchemeDefinition** dest, int from, int to, int n);

  /**
   * Get the svtkQuadratureSchemeDefinition at a specific location in the vector.
   */
  svtkQuadratureSchemeDefinition* Get(svtkInformation* info, int idx);

  // _escription:
  // Get a pointer to the first svtkQuadratureSchemeDefinition in the vector. We are
  // uysing a vector of smart pointers so this is not easy to
  // implement.
  // svtkQuadratureSchemeDefinition **Get(svtkInformation* info);

  //@{
  /**
   * Copy the entry associated with this key from one information
   * object to another.  If there is no entry in the first information
   * object for this key, the value is removed from the second.
   */
  void ShallowCopy(svtkInformation* from, svtkInformation* to) override;
  void DeepCopy(svtkInformation* from, svtkInformation* to) override;
  //@}

  /**
   * Print the key's value in an information object to a stream.
   */
  void Print(ostream& os, svtkInformation* info) override;

  // note: I had wanted to make the following interface in svtkInformationKey
  // with a default implementation that did nothing. but we decided that
  // svtkInformationKey class is too important a class to add such an interface
  // without a thorough design review. we don't have budget for such a review.

  /**
   * Generate an XML representation of the object. Each
   * key/value pair will be nested in the resulting XML hierarchy.
   * The element passed in is assumed to be empty.
   */
  int SaveState(svtkInformation* info, svtkXMLDataElement* element);
  /**
   * Load key/value pairs from an XML state representation created
   * with SaveState. Duplicate keys will generate a fatal error.
   */
  int RestoreState(svtkInformation* info, svtkXMLDataElement* element);

private:
  /**
   * Used to create the underlying vector that will be associated
   * with this key.
   */
  void CreateQuadratureSchemeDefinition();
  /**
   * Get the vector associated with this key, if there is
   * none then associate a new vector with this key and return
   * that.
   */
  svtkInformationQuadratureSchemeDefinitionVectorValue* GetQuadratureSchemeDefinitionVector(
    svtkInformation* info);

  //
  svtkInformationQuadratureSchemeDefinitionVectorKey(
    const svtkInformationQuadratureSchemeDefinitionVectorKey&) = delete;
  void operator=(const svtkInformationQuadratureSchemeDefinitionVectorKey&) = delete;
};

#endif
