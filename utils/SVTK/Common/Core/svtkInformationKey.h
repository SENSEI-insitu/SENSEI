/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationKey.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkInformationKey
 * @brief   Superclass for svtkInformation keys.
 *
 * svtkInformationKey is the superclass for all keys used to access the
 * map represented by svtkInformation.  The svtkInformation::Set and
 * svtkInformation::Get methods of svtkInformation are accessed by
 * information keys.  A key is a pointer to an instance of a subclass
 * of svtkInformationKey.  The type of the subclass determines the
 * overload of Set/Get that is selected.  This ensures that the type
 * of value stored in a svtkInformation instance corresponding to a
 * given key matches the type expected for that key.
 */

#ifndef svtkInformationKey_h
#define svtkInformationKey_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"           // Need svtkTypeMacro
#include "svtkObjectBase.h"

class svtkInformation;

class SVTKCOMMONCORE_EXPORT svtkInformationKey : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkInformationKey, svtkObjectBase);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Prevent normal svtkObject reference counting behavior.
   */
  void Register(svtkObjectBase*) override;

  /**
   * Prevent normal svtkObject reference counting behavior.
   */
  void UnRegister(svtkObjectBase*) override;

  /**
   * Get the name of the key.  This is not the type of the key, but
   * the name of the key instance.
   */
  const char* GetName();

  /**
   * Get the location of the key.  This is the name of the class in
   * which the key is defined.
   */
  const char* GetLocation();

  //@{
  /**
   * Key instances are static data that need to be created and
   * destroyed.  The constructor and destructor must be public.  The
   * name of the static instance and the class in which it is defined
   * should be passed to the constructor.  They must be string
   * literals because the strings are not copied.
   */
  svtkInformationKey(const char* name, const char* location);
  ~svtkInformationKey() override;
  //@}

  /**
   * Copy the entry associated with this key from one information
   * object to another.  If there is no entry in the first information
   * object for this key, the value is removed from the second.
   */
  virtual void ShallowCopy(svtkInformation* from, svtkInformation* to) = 0;

  /**
   * Duplicate (new instance created) the entry associated with this key from
   * one information object to another (new instances of any contained
   * svtkInformation and svtkInformationVector objects are created).
   * Default implementation simply calls ShallowCopy().
   */
  virtual void DeepCopy(svtkInformation* from, svtkInformation* to) { this->ShallowCopy(from, to); }

  /**
   * Check whether this key appears in the given information object.
   */
  virtual int Has(svtkInformation* info);

  /**
   * Remove this key from the given information object.
   */
  virtual void Remove(svtkInformation* info);

  /**
   * Report a reference this key has in the given information object.
   */
  virtual void Report(svtkInformation* info, svtkGarbageCollector* collector);

  //@{
  /**
   * Print the key's value in an information object to a stream.
   */
  void Print(svtkInformation* info);
  virtual void Print(ostream& os, svtkInformation* info);
  //@}

  /**
   * This function is only relevant when the pertaining key
   * is used in a SVTK pipeline. Specific keys that handle
   * pipeline data requests (for example, UPDATE_PIECE_NUMBER)
   * can overwrite this method to notify the pipeline that a
   * a filter should be (re-)executed because what is in
   * the current output is different that what is being requested
   * by the key. For example, DATA_PIECE_NUMBER != UPDATE_PIECE_NUMBER.
   */
  virtual bool NeedToExecute(
    svtkInformation* svtkNotUsed(pipelineInfo), svtkInformation* svtkNotUsed(dobjInfo))
  {
    return false;
  }

  /**
   * This function is only relevant when the pertaining key
   * is used in a SVTK pipeline. Specific keys that handle
   * pipeline data requests (for example, UPDATE_PIECE_NUMBER)
   * can overwrite this method to store in the data information
   * meta-data about the request that led to the current filter
   * execution. This meta-data can later be used to compare what
   * is being requested to decide whether the filter needs to
   * re-execute. For example, a filter may store the current
   * UPDATE_PIECE_NUMBER in the data object's information as
   * the DATA_PIECE_NUMBER. DATA_PIECE_NUMBER can later be compared
   * to a new UPDATA_PIECE_NUMBER to decide whether a filter should
   * re-execute.
   */
  virtual void StoreMetaData(svtkInformation* svtkNotUsed(request),
    svtkInformation* svtkNotUsed(pipelineInfo), svtkInformation* svtkNotUsed(dobjInfo))
  {
  }

  /**
   * This function is only relevant when the pertaining key
   * is used in a SVTK pipeline. By overwriting this method, a
   * key can decide if/how to copy itself downstream or upstream
   * during a particular pipeline pass. For example, meta-data keys
   * can copy themselves during REQUEST_INFORMATION whereas request
   * keys can copy themselves during REQUEST_UPDATE_EXTENT.
   */
  virtual void CopyDefaultInformation(svtkInformation* svtkNotUsed(request),
    svtkInformation* svtkNotUsed(fromInfo), svtkInformation* svtkNotUsed(toInfo))
  {
  }

protected:
  char* Name;
  char* Location;

#define svtkInformationKeySetStringMacro(name)                                                      \
  virtual void Set##name(const char* _arg)                                                         \
  {                                                                                                \
    if (this->name == nullptr && _arg == nullptr)                                                  \
    {                                                                                              \
      return;                                                                                      \
    }                                                                                              \
    if (this->name && _arg && (!strcmp(this->name, _arg)))                                         \
    {                                                                                              \
      return;                                                                                      \
    }                                                                                              \
    delete[] this->name;                                                                           \
    if (_arg)                                                                                      \
    {                                                                                              \
      size_t n = strlen(_arg) + 1;                                                                 \
      char* cp1 = new char[n];                                                                     \
      const char* cp2 = (_arg);                                                                    \
      this->name = cp1;                                                                            \
      do                                                                                           \
      {                                                                                            \
        *cp1++ = *cp2++;                                                                           \
      } while (--n);                                                                               \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
      this->name = nullptr;                                                                        \
    }                                                                                              \
  }

  svtkInformationKeySetStringMacro(Name);
  svtkInformationKeySetStringMacro(Location);

  // Set/Get the value associated with this key instance in the given
  // information object.
  void SetAsObjectBase(svtkInformation* info, svtkObjectBase* value);
  const svtkObjectBase* GetAsObjectBase(svtkInformation* info) const;
  svtkObjectBase* GetAsObjectBase(svtkInformation* info);

  // Report the object associated with this key instance in the given
  // information object to the collector.
  void ReportAsObjectBase(svtkInformation* info, svtkGarbageCollector* collector);

  // Helper for debug leaks support.
  void ConstructClass(const char*);

private:
  svtkInformationKey(const svtkInformationKey&) = delete;
  void operator=(const svtkInformationKey&) = delete;
};

// Macros to define an information key instance in a C++ source file.
// The corresponding method declaration must appear in the class
// definition in the header file.
#define svtkInformationKeyMacro(CLASS, NAME, type)                                                  \
  static svtkInformation##type##Key* CLASS##_##NAME = new svtkInformation##type##Key(#NAME, #CLASS); \
  svtkInformation##type##Key* CLASS::NAME() { return CLASS##_##NAME; }
#define svtkInformationKeySubclassMacro(CLASS, NAME, type, super)                                   \
  static svtkInformation##type##Key* CLASS##_##NAME = new svtkInformation##type##Key(#NAME, #CLASS); \
  svtkInformation##super##Key* CLASS::NAME() { return CLASS##_##NAME; }
#define svtkInformationKeyRestrictedMacro(CLASS, NAME, type, required)                              \
  static svtkInformation##type##Key* CLASS##_##NAME =                                               \
    new svtkInformation##type##Key(#NAME, #CLASS, required);                                        \
  svtkInformation##type##Key* CLASS::NAME() { return CLASS##_##NAME; }

#endif
