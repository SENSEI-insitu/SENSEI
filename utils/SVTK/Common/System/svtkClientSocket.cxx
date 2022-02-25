/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkClientSocket.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkClientSocket.h"

#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkClientSocket);
//-----------------------------------------------------------------------------
svtkClientSocket::svtkClientSocket()
{
  this->ConnectingSide = false;
}

//-----------------------------------------------------------------------------
svtkClientSocket::~svtkClientSocket() = default;

//-----------------------------------------------------------------------------
int svtkClientSocket::ConnectToServer(const char* hostName, int port)
{
  if (this->SocketDescriptor != -1)
  {
    svtkWarningMacro("Client connection already exists. Closing it.");
    this->CloseSocket(this->SocketDescriptor);
    this->SocketDescriptor = -1;
  }

  this->SocketDescriptor = this->CreateSocket();
  if (this->SocketDescriptor == -1)
  {
    svtkErrorMacro("Failed to create socket.");
    return -1;
  }

  if (this->Connect(this->SocketDescriptor, hostName, port) == -1)
  {
    this->CloseSocket(this->SocketDescriptor);
    this->SocketDescriptor = -1;

    svtkErrorMacro("Failed to connect to server " << hostName << ":" << port);
    return -1;
  }

  this->ConnectingSide = true;
  return 0;
}

//-----------------------------------------------------------------------------
void svtkClientSocket::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "ConnectingSide: " << this->ConnectingSide << endl;
}
