/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkServerSocket.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkServerSocket.h"

#include "svtkClientSocket.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkServerSocket);
//-----------------------------------------------------------------------------
svtkServerSocket::svtkServerSocket() = default;

//-----------------------------------------------------------------------------
svtkServerSocket::~svtkServerSocket() = default;

//-----------------------------------------------------------------------------
int svtkServerSocket::GetServerPort()
{
  if (!this->GetConnected())
  {
    return 0;
  }
  return this->GetPort(this->SocketDescriptor);
}

//-----------------------------------------------------------------------------
int svtkServerSocket::CreateServer(int port)
{
  if (this->SocketDescriptor != -1)
  {
    svtkWarningMacro("Server Socket already exists. Closing old socket.");
    this->CloseSocket(this->SocketDescriptor);
    this->SocketDescriptor = -1;
  }
  this->SocketDescriptor = this->CreateSocket();
  if (this->SocketDescriptor < 0)
  {
    return -1;
  }
  if (this->BindSocket(this->SocketDescriptor, port) != 0 ||
    this->Listen(this->SocketDescriptor) != 0)
  {
    // failed to bind or listen.
    this->CloseSocket(this->SocketDescriptor);
    this->SocketDescriptor = -1;
    return -1;
  }
  // Success.
  return 0;
}

//-----------------------------------------------------------------------------
svtkClientSocket* svtkServerSocket::WaitForConnection(unsigned long msec /*=0*/)
{
  if (this->SocketDescriptor < 0)
  {
    svtkErrorMacro("Server Socket not created yet!");
    return nullptr;
  }

  int ret = this->SelectSocket(this->SocketDescriptor, msec);
  if (ret == 0)
  {
    // Timed out.
    return nullptr;
  }
  if (ret == -1)
  {
    svtkErrorMacro("Error selecting socket.");
    return nullptr;
  }
  int clientsock = this->Accept(this->SocketDescriptor);
  if (clientsock == -1)
  {
    svtkErrorMacro("Failed to accept the socket.");
    return nullptr;
  }
  // Create a new svtkClientSocket and return it.
  svtkClientSocket* cs = svtkClientSocket::New();
  cs->SocketDescriptor = clientsock;
  cs->SetConnectingSide(false);
  return cs;
}

//-----------------------------------------------------------------------------
void svtkServerSocket::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
