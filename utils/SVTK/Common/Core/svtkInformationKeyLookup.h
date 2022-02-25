/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationKeyLookup.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkInformationKeyLookup
 * @brief   Find svtkInformationKeys from name and
 * location strings.
 */

#ifndef svtkInformationKeyLookup_h
#define svtkInformationKeyLookup_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

#include <map>     // For std::map
#include <string>  // For std::string
#include <utility> // For std::pair

class svtkInformationKey;

class SVTKCOMMONCORE_EXPORT svtkInformationKeyLookup : public svtkObject
{
public:
  static svtkInformationKeyLookup* New();
  svtkTypeMacro(svtkInformationKeyLookup, svtkObject);

  /**
   * Lists all known keys.
   */
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Find an information key from name and location strings. For example,
   * Find("GUI_HIDE", "svtkAbstractArray") returns svtkAbstractArray::GUI_HIDE.
   * Note that this class only knows about keys in modules that are currently
   * linked to the running executable.
   */
  static svtkInformationKey* Find(const std::string& name, const std::string& location);

protected:
  svtkInformationKeyLookup();
  ~svtkInformationKeyLookup() override;

  friend class svtkInformationKey;

  /**
   * Add a key to the KeyMap. This is done automatically in the
   * svtkInformationKey constructor.
   */
  static void RegisterKey(
    svtkInformationKey* key, const std::string& name, const std::string& location);

private:
  svtkInformationKeyLookup(const svtkInformationKeyLookup&) = delete;
  void operator=(const svtkInformationKeyLookup&) = delete;

  typedef std::pair<std::string, std::string> Identifier; // Location, Name
  typedef std::map<Identifier, svtkInformationKey*> KeyMap;

  // Using a static function / variable here to ensure static initialization
  // works as intended, since key objects are static, too.
  static KeyMap& Keys();
};

#endif // svtkInformationKeyLookup_h
