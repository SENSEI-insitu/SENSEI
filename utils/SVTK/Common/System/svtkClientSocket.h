/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkClientSocket.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkClientSocket
 * @brief   Encapsulates a client socket.
 */

#ifndef svtkClientSocket_h
#define svtkClientSocket_h

#include "svtkCommonSystemModule.h" // For export macro
#include "svtkSocket.h"
class svtkServerSocket;

class SVTKCOMMONSYSTEM_EXPORT svtkClientSocket : public svtkSocket
{
public:
  static svtkClientSocket* New();
  svtkTypeMacro(svtkClientSocket, svtkSocket);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Connects to host. Returns 0 on success, -1 on error.
   */
  int ConnectToServer(const char* hostname, int port);

  //@{
  /**
   * Returns if the socket is on the connecting side (the side that requests a
   * ConnectToServer() or on the connected side (the side that was waiting for
   * the client to connect). This is used to disambiguate the two ends of a socket
   * connection.
   */
  svtkGetMacro(ConnectingSide, bool);
  //@}

protected:
  svtkClientSocket();
  ~svtkClientSocket() override;

  svtkSetMacro(ConnectingSide, bool);
  bool ConnectingSide;
  friend class svtkServerSocket;

private:
  svtkClientSocket(const svtkClientSocket&) = delete;
  void operator=(const svtkClientSocket&) = delete;
};

#endif
