/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkWin32OutputWindow.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkWin32OutputWindow.h"

#include "svtkLogger.h"
#include "svtkObjectFactory.h"
#include "svtkWindows.h"

svtkStandardNewMacro(svtkWin32OutputWindow);

HWND svtkWin32OutputWindowOutputWindow = 0;

//----------------------------------------------------------------------------
LRESULT APIENTRY svtkWin32OutputWindowWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
  switch (message)
  {
    case WM_SIZE:
    {
      int w = LOWORD(lParam); // width of client area
      int h = HIWORD(lParam); // height of client area

      MoveWindow(svtkWin32OutputWindowOutputWindow, 0, 0, w, h, true);
    }
    break;
    case WM_DESTROY:
      svtkWin32OutputWindowOutputWindow = nullptr;
      svtkObject::GlobalWarningDisplayOff();
      break;
    case WM_CREATE:
      break;
  }
  return DefWindowProc(hWnd, message, wParam, lParam);
}

//----------------------------------------------------------------------------
svtkWin32OutputWindow::svtkWin32OutputWindow()
{
  // Default to sending output to stderr/cerr when running a dashboard
  // and logging is not enabled.
  if (getenv("DART_TEST_FROM_DART") || getenv("DASHBOARD_TEST_FROM_CTEST"))
  {
    this->SetDisplayModeToDefault();
  }
  else
  {
    this->SetDisplayModeToNever();
  }
}

//----------------------------------------------------------------------------
svtkWin32OutputWindow::~svtkWin32OutputWindow() {}

//----------------------------------------------------------------------------
// Display text in the window, and translate the \n to \r\n.
//
void svtkWin32OutputWindow::DisplayText(const char* someText)
{
  if (!someText)
  {
    return;
  }
  if (this->PromptUser)
  {
    this->PromptText(someText);
    return;
  }

  const auto streamtype = this->GetDisplayStream(this->GetCurrentMessageType());

  // Create a buffer big enough to hold the entire text
  char* buffer = new char[strlen(someText) + 1];
  // Start at the beginning
  const char* NewLinePos = someText;
  while (NewLinePos)
  {
    int len = 0;
    // Find the next new line in text
    NewLinePos = strchr(someText, '\n');
    // if no new line is found then just add the text
    if (NewLinePos == 0)
    {
      svtkWin32OutputWindow::AddText(someText);
      OutputDebugString(someText);
      switch (streamtype)
      {
        case StreamType::StdOutput:
          cout << someText;
          break;
        case StreamType::StdError:
          cerr << someText;
          break;
        default:
          break;
      }
    }
    // if a new line is found copy it to the buffer
    // and add the buffer with a control new line
    else
    {
      len = NewLinePos - someText;
      strncpy(buffer, someText, len);
      buffer[len] = 0;
      someText = NewLinePos + 1;
      svtkWin32OutputWindow::AddText(buffer);
      svtkWin32OutputWindow::AddText("\r\n");
      OutputDebugString(buffer);
      OutputDebugString("\r\n");
      switch (streamtype)
      {
        case StreamType::StdOutput:
          cout << buffer;
          cout << "\r\n";
          break;
        case StreamType::StdError:
          cerr << buffer;
          cerr << "\r\n";
          break;
        default:
          break;
      }
    }
  }
  delete[] buffer;
}

//----------------------------------------------------------------------------
// Add some text to the EDIT control.
//
void svtkWin32OutputWindow::AddText(const char* someText)
{
  if (!Initialize() || (strlen(someText) == 0))
  {
    return;
  }

#ifdef UNICODE
  // move to the end of the text area
  SendMessageW(svtkWin32OutputWindowOutputWindow, EM_SETSEL, (WPARAM)-1, (LPARAM)-1);
  wchar_t* wmsg = new wchar_t[mbstowcs(nullptr, someText, 32000) + 1];
  mbstowcs(wmsg, someText, 32000);
  // Append the text to the control
  SendMessageW(svtkWin32OutputWindowOutputWindow, EM_REPLACESEL, 0, (LPARAM)wmsg);
  delete[] wmsg;
#else
  // move to the end of the text area
  SendMessageA(svtkWin32OutputWindowOutputWindow, EM_SETSEL, (WPARAM)-1, (LPARAM)-1);
  // Append the text to the control
  SendMessageA(svtkWin32OutputWindowOutputWindow, EM_REPLACESEL, 0, (LPARAM)someText);
#endif
}

//----------------------------------------------------------------------------
// initialize the output window with an EDIT control and
// a container window.
//
int svtkWin32OutputWindow::Initialize()
{
  // check to see if it is already initialized
  if (svtkWin32OutputWindowOutputWindow)
  {
    return 1;
  }

  // Initialize the output window

  WNDCLASS wndClass;
  // has the class been registered ?
#ifdef UNICODE
  if (!GetClassInfo(GetModuleHandle(nullptr), L"svtkOutputWindow", &wndClass))
#else
  if (!GetClassInfo(GetModuleHandle(nullptr), "svtkOutputWindow", &wndClass))
#endif
  {
    wndClass.style = CS_HREDRAW | CS_VREDRAW;
    wndClass.lpfnWndProc = svtkWin32OutputWindowWndProc;
    wndClass.cbClsExtra = 0;
    wndClass.hInstance = GetModuleHandle(nullptr);
#ifndef _WIN32_WCE
    wndClass.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
#endif
    wndClass.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wndClass.lpszMenuName = nullptr;
#ifdef UNICODE
    wndClass.lpszClassName = L"svtkOutputWindow";
#else
    wndClass.lpszClassName = "svtkOutputWindow";
#endif
    // svtk doesn't use these extra bytes, but app writers
    // may want them, so we provide them -- big enough for
    // one run time pointer: 4 bytes on 32-bit builds, 8 bytes
    // on 64-bit builds
    wndClass.cbWndExtra = sizeof(svtkLONG);
    RegisterClass(&wndClass);
  }

  // create parent container window
#ifdef _WIN32_WCE
  HWND win = CreateWindow(L"svtkOutputWindow", L"svtkOutputWindow", WS_OVERLAPPED | WS_CLIPCHILDREN,
    0, 0, 800, 512, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);
#elif UNICODE
  HWND win =
    CreateWindow(L"svtkOutputWindow", L"svtkOutputWindow", WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, 0,
      0, 900, 700, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);
#else
  HWND win =
    CreateWindow("svtkOutputWindow", "svtkOutputWindow", WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, 0, 0,
      900, 700, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);
#endif

  // Now create child window with text display box
  CREATESTRUCT lpParam;
  lpParam.hInstance = GetModuleHandle(nullptr);
  lpParam.hMenu = nullptr;
  lpParam.hwndParent = win;
  lpParam.cx = 900;
  lpParam.cy = 700;
  lpParam.x = 0;
  lpParam.y = 0;
#if defined(_WIN32_WCE) || defined(UNICODE)
  lpParam.lpszName = L"Output Control";
  lpParam.lpszClass = L"EDIT"; // use the RICHEDIT control widget
#else
  lpParam.lpszName = "Output Control";
  lpParam.lpszClass = "EDIT"; // use the RICHEDIT control widget
#endif

#ifdef _WIN32_WCE
  lpParam.style = ES_MULTILINE | ES_READONLY | WS_CHILD | ES_AUTOVSCROLL | ES_AUTOHSCROLL |
    WS_VISIBLE | WS_VSCROLL | WS_HSCROLL;
#else
  lpParam.style = ES_MULTILINE | ES_READONLY | WS_CHILD | ES_AUTOVSCROLL | ES_AUTOHSCROLL |
    WS_VISIBLE | WS_MAXIMIZE | WS_VSCROLL | WS_HSCROLL;
#endif

  lpParam.dwExStyle = 0;
  // Create the EDIT window as a child of win
#if defined(_WIN32_WCE) || defined(UNICODE)
  svtkWin32OutputWindowOutputWindow =
    CreateWindow(lpParam.lpszClass, // pointer to registered class name
      L"",                          // pointer to window name
      lpParam.style,                // window style
      lpParam.x,                    // horizontal position of window
      lpParam.y,                    // vertical position of window
      lpParam.cx,                   // window width
      lpParam.cy,                   // window height
      lpParam.hwndParent,           // handle to parent or owner window
      nullptr,                      // handle to menu or child-window identifier
      lpParam.hInstance,            // handle to application instance
      &lpParam                      // pointer to window-creation data
    );
#else
  svtkWin32OutputWindowOutputWindow =
    CreateWindow(lpParam.lpszClass, // pointer to registered class name
      "",                           // pointer to window name
      lpParam.style,                // window style
      lpParam.x,                    // horizontal position of window
      lpParam.y,                    // vertical position of window
      lpParam.cx,                   // window width
      lpParam.cy,                   // window height
      lpParam.hwndParent,           // handle to parent or owner window
      nullptr,                      // handle to menu or child-window identifier
      lpParam.hInstance,            // handle to application instance
      &lpParam                      // pointer to window-creation data
    );
#endif

  const int maxsize = 5242880;

#ifdef UNICODE
  SendMessageW(svtkWin32OutputWindowOutputWindow, EM_LIMITTEXT, maxsize, 0L);
#else
  SendMessageA(svtkWin32OutputWindowOutputWindow, EM_LIMITTEXT, maxsize, 0L);
#endif

  // show the top level container window
  ShowWindow(win, SW_SHOW);
  return 1;
}

//----------------------------------------------------------------------------
void svtkWin32OutputWindow::PromptText(const char* someText)
{
  size_t svtkmsgsize = strlen(someText) + 100;
  char* svtkmsg = new char[svtkmsgsize];
  snprintf(svtkmsg, svtkmsgsize, "%s\nPress Cancel to suppress any further messages.", someText);
#ifdef UNICODE
  wchar_t* wmsg = new wchar_t[mbstowcs(nullptr, svtkmsg, 32000) + 1];
  mbstowcs(wmsg, svtkmsg, 32000);
  if (MessageBox(nullptr, wmsg, L"Error", MB_ICONERROR | MB_OKCANCEL) == IDCANCEL)
  {
    svtkObject::GlobalWarningDisplayOff();
  }
  delete[] wmsg;
#else
  if (MessageBox(nullptr, svtkmsg, "Error", MB_ICONERROR | MB_OKCANCEL) == IDCANCEL)
  {
    svtkObject::GlobalWarningDisplayOff();
  }
#endif
  delete[] svtkmsg;
}

//----------------------------------------------------------------------------
void svtkWin32OutputWindow::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  if (svtkWin32OutputWindowOutputWindow)
  {
    os << indent << "OutputWindow: " << svtkWin32OutputWindowOutputWindow << "\n";
  }
  else
  {
    os << indent << "OutputWindow: (null)\n";
  }
}

//----------------------------------------------------------------------------
#if !defined(SVTK_LEGACY_REMOVE)
void svtkWin32OutputWindow::SetSendToStdErr(bool val)
{
  SVTK_LEGACY_REPLACED_BODY(
    svtkWin32OutputWindow::SetSendToStdErr, "SVTK 9.0", svtkWin32OutputWindow::SetDisplayMode);
  this->SetDisplayMode(val ? ALWAYS_STDERR : DEFAULT);
}

bool svtkWin32OutputWindow::GetSendToStdErr()
{
  SVTK_LEGACY_REPLACED_BODY(
    svtkWin32OutputWindow::GetSendToStdErr, "SVTK 9.0", svtkWin32OutputWindow::GetDisplayMode);
  return this->GetDisplayMode() == ALWAYS_STDERR;
}

void svtkWin32OutputWindow::SendToStdErrOn()
{
  SVTK_LEGACY_REPLACED_BODY(
    svtkWin32OutputWindow::SendToStdErrOn, "SVTK 9.0", svtkWin32OutputWindow::SetDisplayMode);
  this->SetDisplayMode(ALWAYS_STDERR);
}
void svtkWin32OutputWindow::SendToStdErrOff()
{

  SVTK_LEGACY_REPLACED_BODY(
    svtkWin32OutputWindow::SendToStdErrOff, "SVTK 9.0", svtkWin32OutputWindow::SetDisplayMode);
  this->SetDisplayMode(DEFAULT);
}
#endif
