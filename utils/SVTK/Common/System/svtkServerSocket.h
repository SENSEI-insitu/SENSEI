/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkServerSocket.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkServerSocket
 * @brief   Encapsulate a socket that accepts connections.
 *
 *
 */

#ifndef svtkServerSocket_h
#define svtkServerSocket_h

#include "svtkCommonSystemModule.h" // For export macro
#include "svtkSocket.h"

class svtkClientSocket;
class SVTKCOMMONSYSTEM_EXPORT svtkServerSocket : public svtkSocket
{
public:
  static svtkServerSocket* New();
  svtkTypeMacro(svtkServerSocket, svtkSocket);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Creates a server socket at a given port and binds to it.
   * Returns -1 on error. 0 on success.
   */
  int CreateServer(int port);

  /**
   * Waits for a connection. When a connection is received
   * a new svtkClientSocket object is created and returned.
   * Returns nullptr on timeout.
   */
  svtkClientSocket* WaitForConnection(unsigned long msec = 0);

  /**
   * Returns the port on which the server is running.
   */
  int GetServerPort();

protected:
  svtkServerSocket();
  ~svtkServerSocket() override;

private:
  svtkServerSocket(const svtkServerSocket&) = delete;
  void operator=(const svtkServerSocket&) = delete;
};

#endif
