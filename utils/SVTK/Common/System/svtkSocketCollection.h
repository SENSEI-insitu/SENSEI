/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSocketCollection.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkSocketCollection
 * @brief    a collection for sockets.
 *
 * Apart from being svtkCollection subclass for sockets, this class
 * provides means to wait for activity on all the sockets in the
 * collection simultaneously.
 */

#ifndef svtkSocketCollection_h
#define svtkSocketCollection_h

#include "svtkCollection.h"
#include "svtkCommonSystemModule.h" // For export macro

class svtkSocket;
class SVTKCOMMONSYSTEM_EXPORT svtkSocketCollection : public svtkCollection
{
public:
  static svtkSocketCollection* New();
  svtkTypeMacro(svtkSocketCollection, svtkCollection);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // Add Socket to the collection.
  void AddItem(svtkSocket* soc);

  /**
   * Select all Connected sockets in the collection. If msec is specified,
   * it timesout after msec milliseconds on inactivity.
   * Returns 0 on timeout, -1 on error; 1 is a socket was selected.
   * The selected socket can be retrieved by GetLastSelectedSocket().
   */
  int SelectSockets(unsigned long msec = 0);

  /**
   * Returns the socket selected during the last SelectSockets(), if any.
   * nullptr otherwise.
   */
  svtkSocket* GetLastSelectedSocket() { return this->SelectedSocket; }

  //@{
  /**
   * Overridden to unset SelectedSocket.
   */
  void ReplaceItem(int i, svtkObject*);
  void RemoveItem(int i);
  void RemoveItem(svtkObject*);
  void RemoveAllItems();
  //@}

protected:
  svtkSocketCollection();
  ~svtkSocketCollection() override;

  svtkSocket* SelectedSocket;

private:
  // Hide the standard AddItem.
  void AddItem(svtkObject* o) { this->Superclass::AddItem(o); }

private:
  svtkSocketCollection(const svtkSocketCollection&) = delete;
  void operator=(const svtkSocketCollection&) = delete;
};

#endif
