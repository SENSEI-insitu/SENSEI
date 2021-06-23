/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationVector.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationVector
 * @brief   Store zero or more svtkInformation instances.
 *
 *
 * svtkInformationVector stores a vector of zero or more svtkInformation
 * objects corresponding to the input or output information for a
 * svtkAlgorithm.  An instance of this class is passed to
 * svtkAlgorithm::ProcessRequest calls.
 */

#ifndef svtkInformationVector_h
#define svtkInformationVector_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

class svtkInformation;
class svtkInformationVectorInternals;

class SVTKCOMMONCORE_EXPORT svtkInformationVector : public svtkObject
{
public:
  static svtkInformationVector* New();
  svtkTypeMacro(svtkInformationVector, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Get/Set the number of information objects in the vector.  Setting
   * the number to larger than the current number will create empty
   * svtkInformation instances.  Setting the number to smaller than the
   * current number will remove entries from higher indices.
   */
  int GetNumberOfInformationObjects() { return this->NumberOfInformationObjects; }
  void SetNumberOfInformationObjects(int n);
  //@}

  //@{
  /**
   * Get/Set the svtkInformation instance stored at the given index in
   * the vector.  The vector will automatically expand to include the
   * index given if necessary.  Missing entries in-between will be
   * filled with empty svtkInformation instances.
   */
  void SetInformationObject(int index, svtkInformation* info);
  svtkInformation* GetInformationObject(int index);
  //@}

  //@{
  /**
   * Append/Remove an information object.
   */
  void Append(svtkInformation* info);
  void Remove(svtkInformation* info);
  void Remove(int idx);
  //@}

  //@{
  /**
   * Initiate garbage collection when a reference is removed.
   */
  void Register(svtkObjectBase* o) override;
  void UnRegister(svtkObjectBase* o) override;
  //@}

  /**
   * Copy all information entries from the given svtkInformation
   * instance.  Any previously existing entries are removed.  If
   * deep==1, a deep copy of the information structure is performed (new
   * instances of any contained svtkInformation and svtkInformationVector
   * objects are created).
   */
  void Copy(svtkInformationVector* from, int deep = 0);

protected:
  svtkInformationVector();
  ~svtkInformationVector() override;

  // Internal implementation details.
  svtkInformationVectorInternals* Internal;

  int NumberOfInformationObjects;

  // Garbage collection support.
  void ReportReferences(svtkGarbageCollector*) override;

private:
  svtkInformationVector(const svtkInformationVector&) = delete;
  void operator=(const svtkInformationVector&) = delete;
};

#endif
