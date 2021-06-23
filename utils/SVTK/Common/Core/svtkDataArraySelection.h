/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataArraySelection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkDataArraySelection
 * @brief   Store on/off settings for data arrays for a svtkSource.
 *
 * svtkDataArraySelection can be used by svtkSource subclasses to store
 * on/off settings for whether each svtkDataArray in its input should
 * be passed in the source's output.  This is primarily intended to
 * allow file readers to configure what data arrays are read from the
 * file.
 */

#ifndef svtkDataArraySelection_h
#define svtkDataArraySelection_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

class svtkDataArraySelectionInternals;

class SVTKCOMMONCORE_EXPORT svtkDataArraySelection : public svtkObject
{
public:
  svtkTypeMacro(svtkDataArraySelection, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  static svtkDataArraySelection* New();

  /**
   * Enable the array with the given name.  Creates a new entry if
   * none exists.
   *
   * This method will call `this->Modified()` if the enable state for the
   * array changed.
   */
  void EnableArray(const char* name);

  /**
   * Disable the array with the given name.  Creates a new entry if
   * none exists.
   *
   * This method will call `this->Modified()` if the enable state for the
   * array changed.
   */
  void DisableArray(const char* name);

  /**
   * Return whether the array with the given name is enabled.  If
   * there is no entry, the array is assumed to be disabled.
   */
  int ArrayIsEnabled(const char* name) const;

  /**
   * Return whether the array with the given name exists.
   */
  int ArrayExists(const char* name) const;

  /**
   * Enable all arrays that currently have an entry.
   *
   * This method will call `this->Modified()` if the enable state for any of the known
   * arrays is changed.
   */
  void EnableAllArrays();

  /**
   * Disable all arrays that currently have an entry.
   *
   * This method will call `this->Modified()` if the enable state for any of the known
   * arrays is changed.
   */
  void DisableAllArrays();

  /**
   * Get the number of arrays that currently have an entry.
   */
  int GetNumberOfArrays() const;

  /**
   * Get the number of arrays that are enabled.
   */
  int GetNumberOfArraysEnabled() const;

  /**
   * Get the name of the array entry at the given index.
   */
  const char* GetArrayName(int index) const;

  /**
   * Get an index of the array with the given name.
   */
  int GetArrayIndex(const char* name) const;

  /**
   * Get the index of an array with the given name among those
   * that are enabled.  Returns -1 if the array is not enabled.
   */
  int GetEnabledArrayIndex(const char* name) const;

  /**
   * Get whether the array at the given index is enabled.
   */
  int GetArraySetting(int index) const;

  /**
   * Get whether the array is enabled/disable using its name.
   */
  int GetArraySetting(const char* name) const { return this->ArrayIsEnabled(name); }

  /**
   * Set array setting given the name. If the array doesn't exist, it will be
   * added.
   *
   * This method will call `this->Modified()` if the enable state for the
   * array changed.
   */
  void SetArraySetting(const char* name, int status);

  /**
   * Remove all array entries.
   *
   * This method will call `this->Modified()` if the arrays were cleared.
   */
  void RemoveAllArrays();

  /**
   * Add to the list of arrays that have entries.  For arrays that
   * already have entries, the settings are untouched.  For arrays
   * that don't already have an entry, they are assumed to be enabled
   * by default. The state can also be passed as the second argument.
   * This method should be called only by the filter owning this
   * object.
   *
   * This method **does not** call `this->Modified()`.
   *
   * Also note for arrays already known to this instance (i.e.
   * `this->ArrayExists(name) == true`, this method has no effect.
   */
  int AddArray(const char* name, bool state = true);

  /**
   * Remove an array setting given its index.
   *
   * This method **does not** call `this->Modified()`.
   */
  void RemoveArrayByIndex(int index);

  /**
   * Remove an array setting given its name.
   *
   * This method **does not** call `this->Modified()`.
   */
  void RemoveArrayByName(const char* name);

  //@{
  /**
   * Set the list of arrays that have entries.  For arrays that
   * already have entries, the settings are copied.  For arrays that
   * don't already have an entry, they are assigned the given default
   * status.  If no default status is given, it is assumed to be on.
   * There will be no more entries than the names given.  This method
   * should be called only by the filter owning this object.  The
   * signature with the default must have a different name due to a
   * bug in the Borland C++ 5.5 compiler.
   *
   * This method **does not** call `this->Modified()`.
   */
  void SetArrays(const char* const* names, int numArrays);
  void SetArraysWithDefault(const char* const* names, int numArrays, int defaultStatus);
  //@}

  /**
   * Copy the selections from the given svtkDataArraySelection instance.
   *
   * This method will call `this->Modified()` if the array selections changed.
   */
  void CopySelections(svtkDataArraySelection* selections);

  /**
   * Update `this` to include values from `other`. For arrays that don't
   * exist in `this` but exist in `other`, they will get added to `this` with
   * the same array setting as in `other`. Array settings for arrays already in
   * `this` are left unchanged.
   *
   * This method will call `this->Modified()` if the array selections changed.
   */
  void Union(svtkDataArraySelection* other);

  //@{
  /**
   * Get/Set enabled state for any unknown arrays. Default is 0 i.e. not
   * enabled. When set to 1, `ArrayIsEnabled` will return 1 for any
   * array not explicitly specified.
   */
  svtkSetMacro(UnknownArraySetting, int);
  svtkGetMacro(UnknownArraySetting, int);
  //@}
protected:
  svtkDataArraySelection();
  ~svtkDataArraySelection() override;

  // Internal implementation details.
  svtkDataArraySelectionInternals* Internal;

  int UnknownArraySetting;

private:
  svtkDataArraySelection(const svtkDataArraySelection&) = delete;
  void operator=(const svtkDataArraySelection&) = delete;
};

#endif
