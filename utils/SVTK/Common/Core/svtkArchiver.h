/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArchiver.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkArchiver
 * @brief   Writes an archive
 *
 * svtkArchiver is a base class for constructing an archive. The default
 * implementation constructs a directory at the location of the ArchiveName
 * and populates it with files and directories as requested by Insert().
 * Classes that derive from svtkArchiver can customize the output using such
 * features as compression, in-memory serialization and third-party archival
 * tools.
 */

#ifndef svtkArchiver_h
#define svtkArchiver_h

#include "svtkCommonCoreModule.h" // For export macro

#include "svtkObject.h"

#include <ios> // For std::streamsize

class SVTKCOMMONCORE_EXPORT svtkArchiver : public svtkObject
{
public:
  static svtkArchiver* New();
  svtkTypeMacro(svtkArchiver, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Specify the name of the archive to generate.
   */
  svtkGetStringMacro(ArchiveName);
  svtkSetStringMacro(ArchiveName);
  //@}

  //@{
  /**
   * Open the arhive for writing.
   */
  virtual void OpenArchive();
  //@}

  //@{
  /**
   * Close the arhive.
   */
  virtual void CloseArchive();
  //@}

  //@{
  /**
   * Insert \p data of size \p size into the archive at \p relativePath.
   */
  virtual void InsertIntoArchive(
    const std::string& relativePath, const char* data, std::size_t size);
  //@}

  //@{
  /**
   * Checks if \p relativePath represents an entry in the archive.
   */
  virtual bool Contains(const std::string& relativePath);
  //@}

protected:
  svtkArchiver();
  ~svtkArchiver() override;

  char* ArchiveName;

private:
  svtkArchiver(const svtkArchiver&) = delete;
  void operator=(const svtkArchiver&) = delete;
};

#endif
