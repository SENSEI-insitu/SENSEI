/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSocket.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkSocket.h"

#include "svtkObjectFactory.h"

// The SVTK_SOCKET_FAKE_API definition is given to the compiler
// command line by CMakeLists.txt if there is no real sockets
// interface available.  When this macro is defined we simply make
// every method return failure.
//
// Perhaps we should add a method to query at runtime whether a real
// sockets interface is available.

#ifndef SVTK_SOCKET_FAKE_API
#if defined(_WIN32) && !defined(__CYGWIN__)
#define SVTK_WINDOWS_FULL
#include "svtkWindows.h"
#else
#include <arpa/inet.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#endif

#if defined(_WIN32) && !defined(__CYGWIN__)

// TODO : document why we restrict to v1.1
#define WSA_VERSION MAKEWORD(1, 1)

#define svtkCloseSocketMacro(sock) (closesocket(sock))
#define svtkErrnoMacro (WSAGetLastError())
#define svtkStrerrorMacro(_num) (wsaStrerror(_num))
#define svtkSocketErrorIdMacro(_id) (WSA##_id)
#define svtkSocketErrorReturnMacro (SOCKET_ERROR)

#else

#define svtkCloseSocketMacro(sock) (close(sock))
#define svtkErrnoMacro (errno)
#define svtkStrerrorMacro(_num) (strerror(_num))
#define svtkSocketErrorIdMacro(_id) (_id)
#define svtkSocketErrorReturnMacro (-1)

#endif

// This macro wraps a system function call(_call),
// restarting the call in case it was interrupted
// by a signal (EINTR).
#define svtkRestartInterruptedSystemCallMacro(_call, _ret)                                          \
  do                                                                                               \
  {                                                                                                \
    (_ret) = (_call);                                                                              \
  } while (                                                                                        \
    ((_ret) == svtkSocketErrorReturnMacro) && (svtkErrnoMacro == svtkSocketErrorIdMacro(EINTR)));

// use when _str may be a null pointer but _fallback is not.
#define svtkSafeStrMacro(_str, _fallback) ((_str) ? (_str) : (_fallback))

// convert error number to string and report via svtkErrorMacro.
#define svtkSocketErrorMacro(_eno, _message)                                                        \
  svtkErrorMacro(<< (_message) << " " << svtkSafeStrMacro(svtkStrerrorMacro(_eno), "unknown error")   \
                << ".");

// convert error number to string and report via svtkGenericWarningMacro
#define svtkSocketGenericErrorMacro(_message)                                                       \
  svtkGenericWarningMacro(<< (_message) << " "                                                      \
                         << svtkSafeStrMacro(svtkStrerrorMacro(svtkErrnoMacro), "unknown error")      \
                         << ".");

// on windows strerror doesn't handle socket error codes
#if defined(_WIN32) && !defined(__CYGWIN__)
static const char* wsaStrerror(int wsaeid)
{
  static char buf[256] = { '\0' };
  int ok;
  ok = FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, 0, wsaeid, 0, buf, 256, 0);
  if (!ok)
  {
    return 0;
  }
  return buf;
}
#endif

//-----------------------------------------------------------------------------
svtkSocket::svtkSocket()
{
  this->SocketDescriptor = -1;
}

//-----------------------------------------------------------------------------
svtkSocket::~svtkSocket()
{
  if (this->SocketDescriptor != -1)
  {
    this->CloseSocket(this->SocketDescriptor);
    this->SocketDescriptor = -1;
  }
}

//-----------------------------------------------------------------------------
int svtkSocket::CreateSocket()
{
#ifndef SVTK_SOCKET_FAKE_API
  int sock;
  svtkRestartInterruptedSystemCallMacro(socket(AF_INET, SOCK_STREAM, 0), sock);
  if (sock == svtkSocketErrorReturnMacro)
  {
    svtkSocketErrorMacro(svtkErrnoMacro, "Socket error in call to socket.");
    return -1;
  }

  // Elimate windows 0.2 second delay sending (buffering) data.
  int on = 1;
  int iErr;
  svtkRestartInterruptedSystemCallMacro(
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (char*)&on, sizeof(on)), iErr);
  if (iErr == svtkSocketErrorReturnMacro)
  {
    svtkSocketErrorMacro(svtkErrnoMacro, "Socket error in call to setsockopt.");
    return -1;
  }

  return sock;
#else
  return -1;
#endif
}

//-----------------------------------------------------------------------------
void svtkSocket::CloseSocket()
{
  this->CloseSocket(this->SocketDescriptor);
  this->SocketDescriptor = -1;
}

//-----------------------------------------------------------------------------
int svtkSocket::BindSocket(int socketdescriptor, int port)
{
#ifndef SVTK_SOCKET_FAKE_API
  struct sockaddr_in server;

  server.sin_family = AF_INET;
  server.sin_addr.s_addr = INADDR_ANY;
  server.sin_port = htons(port);
  // Allow the socket to be bound to an address that is already in use
  int opt = 1;
  int iErr = ~svtkSocketErrorReturnMacro;
#ifdef _WIN32
  svtkRestartInterruptedSystemCallMacro(
    setsockopt(socketdescriptor, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(int)), iErr);
#elif defined(SVTK_HAVE_SO_REUSEADDR)
  svtkRestartInterruptedSystemCallMacro(
    setsockopt(socketdescriptor, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(int)), iErr);
#endif
  if (iErr == svtkSocketErrorReturnMacro)
  {
    svtkSocketErrorMacro(svtkErrnoMacro, "Socket error in call to setsockopt.");
    return -1;
  }

  svtkRestartInterruptedSystemCallMacro(
    bind(socketdescriptor, reinterpret_cast<sockaddr*>(&server), sizeof(server)), iErr);
  if (iErr == svtkSocketErrorReturnMacro)
  {
    svtkSocketErrorMacro(svtkErrnoMacro, "Socket error in call to bind.");
    return -1;
  }

  return 0;
#else
  static_cast<void>(socketdescriptor);
  static_cast<void>(port);
  return -1;
#endif
}

//-----------------------------------------------------------------------------
int svtkSocket::Accept(int socketdescriptor)
{
#ifndef SVTK_SOCKET_FAKE_API
  if (socketdescriptor < 0)
  {
    svtkErrorMacro("Invalid descriptor.");
    return -1;
  }

  int newDescriptor;
  svtkRestartInterruptedSystemCallMacro(accept(socketdescriptor, nullptr, nullptr), newDescriptor);
  if (newDescriptor == svtkSocketErrorReturnMacro)
  {
    svtkSocketErrorMacro(svtkErrnoMacro, "Socket error in call to accept.");
    return -1;
  }

  return newDescriptor;
#else
  static_cast<void>(socketdescriptor);
  return -1;
#endif
}

//-----------------------------------------------------------------------------
int svtkSocket::Listen(int socketdescriptor)
{
#ifndef SVTK_SOCKET_FAKE_API
  if (socketdescriptor < 0)
  {
    svtkErrorMacro("Invalid descriptor.");
    return -1;
  }

  int iErr;
  svtkRestartInterruptedSystemCallMacro(listen(socketdescriptor, 1), iErr);
  if (iErr == svtkSocketErrorReturnMacro)
  {
    svtkSocketErrorMacro(svtkErrnoMacro, "Socket error in call to listen.");
    return -1;
  }

  return 0;
#else
  static_cast<void>(socketdescriptor);
  return -1;
#endif
}

//-----------------------------------------------------------------------------
int svtkSocket::SelectSocket(int socketdescriptor, unsigned long msec)
{
#ifndef SVTK_SOCKET_FAKE_API
  if (socketdescriptor < 0)
  {
    svtkErrorMacro("Invalid descriptor.");
    return -1;
  }

  fd_set rset;
  int res;
  do
  {
    struct timeval tval;
    struct timeval* tvalptr = nullptr;
    if (msec > 0)
    {
      tval.tv_sec = msec / 1000;
      tval.tv_usec = (msec % 1000) * 1000;
      tvalptr = &tval;
    }

    FD_ZERO(&rset);
    FD_SET(socketdescriptor, &rset);

    // block until socket is readable.
    res = select(socketdescriptor + 1, &rset, nullptr, nullptr, tvalptr);
  } while ((res == svtkSocketErrorReturnMacro) && (svtkErrnoMacro == svtkSocketErrorIdMacro(EINTR)));

  if (res == 0)
  {
    // time out
    return 0;
  }
  else if (res == svtkSocketErrorReturnMacro)
  {
    // error in the call
    svtkSocketErrorMacro(svtkErrnoMacro, "Socket error in call to select.");
    return -1;
  }
  else if (!FD_ISSET(socketdescriptor, &rset))
  {
    svtkErrorMacro("Socket error in select. Descriptor not selected.");
    return -1;
  }

  // NOTE: not checking for pending errors,these will be handled
  // in the next call to read/recv

  // The indicated socket has some activity on it.
  return 1;
#else
  static_cast<void>(socketdescriptor);
  static_cast<void>(msec);
  return -1;
#endif
}

//-----------------------------------------------------------------------------
int svtkSocket::SelectSockets(
  const int* sockets_to_select, int size, unsigned long msec, int* selected_index)
{
#ifndef SVTK_SOCKET_FAKE_API

  *selected_index = -1;

  if (size < 0)
  {
    svtkGenericWarningMacro("Can't select fewer than 0.");
    return -1;
  }

  fd_set rset;
  int res = -1;
  do
  {
    struct timeval tval;
    struct timeval* tvalptr = nullptr;
    if (msec > 0)
    {
      tval.tv_sec = msec / 1000;
      tval.tv_usec = (msec % 1000) * 1000;
      tvalptr = &tval;
    }

    FD_ZERO(&rset);
    int max_fd = -1;
    for (int i = 0; i < size; i++)
    {
      FD_SET(sockets_to_select[i], &rset);
      max_fd = (sockets_to_select[i] > max_fd ? sockets_to_select[i] : max_fd);
    }

    // block until one socket is ready to read.
    res = select(max_fd + 1, &rset, nullptr, nullptr, tvalptr);
  } while ((res == svtkSocketErrorReturnMacro) && (svtkErrnoMacro == svtkSocketErrorIdMacro(EINTR)));

  if (res == 0)
  {
    // time out
    return 0;
  }
  else if (res == svtkSocketErrorReturnMacro)
  {
    // error in the call
    svtkSocketGenericErrorMacro("Socket error in call to select.");
    return -1;
  }

  // find the first socket which has some activity.
  for (int i = 0; i < size; i++)
  {
    if (FD_ISSET(sockets_to_select[i], &rset))
    {
      // NOTE: not checking for pending errors, these
      // will be handled in the next call to read/recv

      *selected_index = i;
      return 1;
    }
  }

  // no activity on any of the sockets
  svtkGenericWarningMacro("Socket error in select. No descriptor selected.");
  return -1;
#else
  static_cast<void>(sockets_to_select);
  static_cast<void>(size);
  static_cast<void>(msec);
  static_cast<void>(selected_index);
  return -1;
#endif
}

//-----------------------------------------------------------------------------
int svtkSocket::Connect(int socketdescriptor, const char* hostName, int port)
{
#ifndef SVTK_SOCKET_FAKE_API
  if (socketdescriptor < 0)
  {
    svtkErrorMacro("Invalid descriptor.");
    return -1;
  }

  struct hostent* hp;
  hp = gethostbyname(hostName);
  if (!hp)
  {
    unsigned long addr = inet_addr(hostName);
    hp = gethostbyaddr((char*)&addr, sizeof(addr), AF_INET);
  }
  if (!hp)
  {
    svtkErrorMacro("Unknown host: " << hostName);
    return -1;
  }

  struct sockaddr_in name;
  name.sin_family = AF_INET;
  memcpy(&name.sin_addr, hp->h_addr, hp->h_length);
  name.sin_port = htons(port);

  int iErr = connect(socketdescriptor, reinterpret_cast<sockaddr*>(&name), sizeof(name));
  if ((iErr == svtkSocketErrorReturnMacro) && (svtkErrnoMacro == svtkSocketErrorIdMacro(EINTR)))
  {
    // Restarting an interrupted connect call only works on linux,
    // other unix require a call to select which blocks until the
    // connection is complete.
    // See Stevens 2d ed, 15.4 p413, "interrupted connect"
    iErr = this->SelectSocket(socketdescriptor, 0);
    if (iErr == -1)
    {
      // SelectSocket doesn't test for pending errors.
      int pendingErr;
#if defined(SVTK_HAVE_GETSOCKNAME_WITH_SOCKLEN_T)
      socklen_t pendingErrLen = sizeof(pendingErr);
#else
      int pendingErrLen = sizeof(pendingErr);
#endif

      svtkRestartInterruptedSystemCallMacro(
        getsockopt(socketdescriptor, SOL_SOCKET, SO_ERROR, (char*)&pendingErr, &pendingErrLen),
        iErr);
      if (iErr == svtkSocketErrorReturnMacro)
      {
        svtkSocketErrorMacro(svtkErrnoMacro, "Socket error in call to getsockopt.");
        return -1;
      }
      else if (pendingErr)
      {
        svtkSocketErrorMacro(pendingErr, "Socket error pending from call to connect.");
        return -1;
      }
    }
  }
  else if (iErr == svtkSocketErrorReturnMacro)
  {
    svtkSocketErrorMacro(svtkErrnoMacro, "Socket error in call to connect.");
    return -1;
  }

  return 0;
#else
  static_cast<void>(socketdescriptor);
  static_cast<void>(hostName);
  static_cast<void>(port);
  return -1;
#endif
}

//-----------------------------------------------------------------------------
int svtkSocket::GetPort(int sock)
{
#ifndef SVTK_SOCKET_FAKE_API
  struct sockaddr_in sockinfo;
  memset(&sockinfo, 0, sizeof(sockinfo));
#if defined(SVTK_HAVE_GETSOCKNAME_WITH_SOCKLEN_T)
  socklen_t sizebuf = sizeof(sockinfo);
#else
  int sizebuf = sizeof(sockinfo);
#endif

  int iErr;
  svtkRestartInterruptedSystemCallMacro(
    getsockname(sock, reinterpret_cast<sockaddr*>(&sockinfo), &sizebuf), iErr);
  if (iErr == svtkSocketErrorReturnMacro)
  {
    svtkSocketErrorMacro(svtkErrnoMacro, "Socket error in call to getsockname.");
    return 0;
  }
  return ntohs(sockinfo.sin_port);
#else
  static_cast<void>(sock);
  return -1;
#endif
}

//-----------------------------------------------------------------------------
void svtkSocket::CloseSocket(int socketdescriptor)
{
#ifndef SVTK_SOCKET_FAKE_API
  if (socketdescriptor < 0)
  {
    svtkErrorMacro("Invalid descriptor.");
    return;
  }
  int iErr;
  svtkRestartInterruptedSystemCallMacro(svtkCloseSocketMacro(socketdescriptor), iErr);
  if (iErr == svtkSocketErrorReturnMacro)
  {
    svtkSocketErrorMacro(svtkErrnoMacro, "Socket error in call to close/closesocket.");
  }
#else
  static_cast<void>(socketdescriptor);
  return;
#endif
}

//-----------------------------------------------------------------------------
int svtkSocket::Send(const void* data, int length)
{
#ifndef SVTK_SOCKET_FAKE_API
  if (!this->GetConnected())
  {
    svtkErrorMacro("Not connected.");
    return 0;
  }
  if (length == 0)
  {
    // nothing to send.
    return 1;
  }
  const char* buffer = reinterpret_cast<const char*>(data);
  int total = 0;
  do
  {
    int flags = 0;
    int nSent;
    svtkRestartInterruptedSystemCallMacro(
      send(this->SocketDescriptor, buffer + total, length - total, flags), nSent);
    if (nSent == svtkSocketErrorReturnMacro)
    {
      svtkSocketErrorMacro(svtkErrnoMacro, "Socket error in call to send.");
      return 0;
    }
    total += nSent;
  } while (total < length);

  return 1;
#else
  static_cast<void>(data);
  static_cast<void>(length);
  return 0;
#endif
}

//-----------------------------------------------------------------------------
int svtkSocket::Receive(void* data, int length, int readFully /*=1*/)
{
#ifndef SVTK_SOCKET_FAKE_API
  if (!this->GetConnected())
  {
    svtkErrorMacro("Not connected.");
    return 0;
  }

#if defined(_WIN32) && !defined(__CYGWIN__)
  int tries = 0;
#endif

  char* buffer = reinterpret_cast<char*>(data);
  int total = 0;
  do
  {
    int nRecvd;
    svtkRestartInterruptedSystemCallMacro(
      recv(this->SocketDescriptor, buffer + total, length - total, 0), nRecvd);

    if (nRecvd == 0)
    {
      // peer shut down
      return 0;
    }

#if defined(_WIN32) && !defined(__CYGWIN__)
    if ((nRecvd == svtkSocketErrorReturnMacro) && (WSAGetLastError() == WSAENOBUFS))
    {
      // On long messages, Windows recv sometimes fails with WSAENOBUFS, but
      // will work if you try again.
      if ((tries++ < 1000))
      {
        Sleep(1);
        continue;
      }
      svtkSocketErrorMacro(svtkErrnoMacro, "Socket error in call to recv.");
      return 0;
    }
#endif

    total += nRecvd;
  } while (readFully && (total < length));

  return total;
#else
  static_cast<void>(data);
  static_cast<void>(length);
  static_cast<void>(readFully);
  return 0;
#endif
}

//-----------------------------------------------------------------------------
void svtkSocket::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "SocketDescriptor: " << this->SocketDescriptor << endl;
}
