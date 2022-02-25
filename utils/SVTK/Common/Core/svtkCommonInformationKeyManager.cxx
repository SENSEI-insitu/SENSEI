/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCommonInformationKeyManager.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkCommonInformationKeyManager.h"

#include "svtkInformationKey.h"

#include <vector>

// Subclass vector so we can directly call constructor.  This works
// around problems on Borland C++.
struct svtkCommonInformationKeyManagerKeysType : public std::vector<svtkInformationKey*>
{
  typedef std::vector<svtkInformationKey*> Superclass;
  typedef Superclass::iterator iterator;
};

//----------------------------------------------------------------------------
// Must NOT be initialized.  Default initialization to zero is
// necessary.
static unsigned int svtkCommonInformationKeyManagerCount;
static svtkCommonInformationKeyManagerKeysType* svtkCommonInformationKeyManagerKeys;

//----------------------------------------------------------------------------
svtkCommonInformationKeyManager::svtkCommonInformationKeyManager()
{
  if (++svtkCommonInformationKeyManagerCount == 1)
  {
    svtkCommonInformationKeyManager::ClassInitialize();
  }
}

//----------------------------------------------------------------------------
svtkCommonInformationKeyManager::~svtkCommonInformationKeyManager()
{
  if (--svtkCommonInformationKeyManagerCount == 0)
  {
    svtkCommonInformationKeyManager::ClassFinalize();
  }
}

//----------------------------------------------------------------------------
void svtkCommonInformationKeyManager::Register(svtkInformationKey* key)
{
  // Register this instance for deletion by the singleton.
  svtkCommonInformationKeyManagerKeys->push_back(key);
}

//----------------------------------------------------------------------------
void svtkCommonInformationKeyManager::ClassInitialize()
{
  // Allocate the singleton storing pointers to information keys.
  // This must be a malloc/free pair instead of new/delete to work
  // around problems on MachO (Mac OS X) runtime systems that do lazy
  // symbol loading.  Calling operator new here causes static
  // initialization to occur in other translation units immediately,
  // which then may try to access the vector before it is set here.
  void* keys = malloc(sizeof(svtkCommonInformationKeyManagerKeysType));
  svtkCommonInformationKeyManagerKeys = new (keys) svtkCommonInformationKeyManagerKeysType;
}

//----------------------------------------------------------------------------
void svtkCommonInformationKeyManager::ClassFinalize()
{
  if (svtkCommonInformationKeyManagerKeys)
  {
    // Delete information keys.
    for (svtkCommonInformationKeyManagerKeysType::iterator i =
           svtkCommonInformationKeyManagerKeys->begin();
         i != svtkCommonInformationKeyManagerKeys->end(); ++i)
    {
      svtkInformationKey* key = *i;
      delete key;
    }

    // Delete the singleton storing pointers to information keys.  See
    // ClassInitialize above for why this is a free instead of a
    // delete.
    svtkCommonInformationKeyManagerKeys->~svtkCommonInformationKeyManagerKeysType();
    free(svtkCommonInformationKeyManagerKeys);
    svtkCommonInformationKeyManagerKeys = nullptr;
  }
}
