/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSocketCollection.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkSocketCollection.h"

#include "svtkCollectionIterator.h"
#include "svtkObjectFactory.h"
#include "svtkSocket.h"

svtkStandardNewMacro(svtkSocketCollection);
//-----------------------------------------------------------------------------
svtkSocketCollection::svtkSocketCollection()
{
  this->SelectedSocket = nullptr;
}

//-----------------------------------------------------------------------------
svtkSocketCollection::~svtkSocketCollection() = default;

//-----------------------------------------------------------------------------
void svtkSocketCollection::AddItem(svtkSocket* soc)
{
  this->Superclass::AddItem(soc);
}

//-----------------------------------------------------------------------------
int svtkSocketCollection::SelectSockets(unsigned long msec /*=0*/)
{
  // clear last selected socket.
  this->SelectedSocket = nullptr;

  int max = this->GetNumberOfItems();
  if (max <= 0)
  {
    svtkErrorMacro("No sockets to select.");
    return -1;
  }

  int* socket_indices = new int[max];
  int* sockets_to_select = new int[max];
  int no_of_sockets = 0;

  svtkCollectionIterator* iter = this->NewIterator();

  int index = 0;
  for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem(), index++)
  {
    svtkSocket* soc = svtkSocket::SafeDownCast(iter->GetCurrentObject());
    if (!soc->GetConnected())
    {
      // skip not-connected sockets.
      continue;
    }
    int sockfd = soc->GetSocketDescriptor();
    sockets_to_select[no_of_sockets] = sockfd;
    socket_indices[no_of_sockets] = index;
    no_of_sockets++;
  }

  if (no_of_sockets == 0)
  {
    svtkErrorMacro("No alive sockets!");
    delete[] sockets_to_select;
    delete[] socket_indices;
    return -1;
  }
  int res = svtkSocket::SelectSockets(sockets_to_select, no_of_sockets, msec, &index);
  int actual_index = -1;
  if (index != -1)
  {
    actual_index = socket_indices[index];
  }

  iter->Delete();
  delete[] sockets_to_select;
  delete[] socket_indices;

  if (res <= 0 || index == -1)
  {
    return res;
  }

  this->SelectedSocket = svtkSocket::SafeDownCast(this->GetItemAsObject(actual_index));
  return 1;
}

//-----------------------------------------------------------------------------
void svtkSocketCollection::RemoveItem(svtkObject* a)
{
  if (this->SelectedSocket && this->SelectedSocket == a)
  {
    this->SelectedSocket = nullptr;
  }
  this->Superclass::RemoveItem(a);
}

//-----------------------------------------------------------------------------
void svtkSocketCollection::RemoveItem(int i)
{
  if (this->SelectedSocket && this->GetItemAsObject(i) == this->SelectedSocket)
  {
    this->SelectedSocket = nullptr;
  }
  this->Superclass::RemoveItem(i);
}

//-----------------------------------------------------------------------------
void svtkSocketCollection::ReplaceItem(int i, svtkObject* a)
{
  if (this->SelectedSocket && this->GetItemAsObject(i) == this->SelectedSocket)
  {
    this->SelectedSocket = nullptr;
  }
  this->Superclass::ReplaceItem(i, a);
}

//-----------------------------------------------------------------------------
void svtkSocketCollection::RemoveAllItems()
{
  this->SelectedSocket = nullptr;
  this->Superclass::RemoveAllItems();
}

//-----------------------------------------------------------------------------
void svtkSocketCollection::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
