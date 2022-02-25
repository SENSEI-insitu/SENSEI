/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkOutputWindow.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkOutputWindow.h"
#include "svtkToolkits.h"
#if defined(_WIN32) && !defined(SVTK_USE_X)
#include "svtkWin32OutputWindow.h"
#endif
#if defined(ANDROID)
#include "svtkAndroidOutputWindow.h"
#endif

#include "svtkCommand.h"
#include "svtkLogger.h"
#include "svtkObjectFactory.h"

#include <sstream>

namespace
{
// helps in set and restore value when an instance goes in
// and out of scope respectively.
template <class T>
class svtkScopedSet
{
  T* Ptr;
  T OldVal;

public:
  svtkScopedSet(T* ptr, const T& newval)
    : Ptr(ptr)
    , OldVal(*ptr)
  {
    *this->Ptr = newval;
  }
  ~svtkScopedSet() { *this->Ptr = this->OldVal; }
};
}

//----------------------------------------------------------------------------
svtkOutputWindow* svtkOutputWindow::Instance = nullptr;
static unsigned int svtkOutputWindowCleanupCounter = 0;

// helps accessing private members in svtkOutputWindow.
class svtkOutputWindowPrivateAccessor
{
  svtkOutputWindow* Instance;

public:
  svtkOutputWindowPrivateAccessor(svtkOutputWindow* self)
    : Instance(self)
  {
    ++self->InStandardMacros;
  }
  ~svtkOutputWindowPrivateAccessor() { --(this->Instance->InStandardMacros); }
};

void svtkOutputWindowDisplayText(const char* message)
{
  svtkOutputWindow::GetInstance()->DisplayText(message);
}

void svtkOutputWindowDisplayErrorText(const char* message)
{
  svtkLogF(ERROR, "%s", message);
  if (auto win = svtkOutputWindow::GetInstance())
  {
    svtkOutputWindowPrivateAccessor helper_raii(win);
    win->DisplayErrorText(message);
  }
}

void svtkOutputWindowDisplayWarningText(const char* message)
{
  svtkLogF(WARNING, "%s", message);
  if (auto win = svtkOutputWindow::GetInstance())
  {
    svtkOutputWindowPrivateAccessor helper_raii(win);
    win->DisplayWarningText(message);
  }
}

void svtkOutputWindowDisplayGenericWarningText(const char* message)
{
  svtkLogF(WARNING, "%s", message);
  if (auto win = svtkOutputWindow::GetInstance())
  {
    svtkOutputWindowPrivateAccessor helper_raii(win);
    win->DisplayGenericWarningText(message);
  }
}

void svtkOutputWindowDisplayDebugText(const char* message)
{
  svtkLogF(INFO, "%s", message);
  if (auto win = svtkOutputWindow::GetInstance())
  {
    svtkOutputWindowPrivateAccessor helper_raii(win);
    win->DisplayDebugText(message);
  }
}

void svtkOutputWindowDisplayErrorText(
  const char* fname, int lineno, const char* message, svtkObject* sourceObj)
{
  svtkLogger::Log(svtkLogger::VERBOSITY_ERROR, fname, lineno, message);

  std::ostringstream svtkmsg;
  svtkmsg << "ERROR: In " << fname << ", line " << lineno << "\n" << message << "\n\n";
  if (sourceObj && sourceObj->HasObserver(svtkCommand::ErrorEvent))
  {
    sourceObj->InvokeEvent(svtkCommand::ErrorEvent, const_cast<char*>(svtkmsg.str().c_str()));
  }
  else if (auto win = svtkOutputWindow::GetInstance())
  {
    svtkOutputWindowPrivateAccessor helper_raii(win);
    win->DisplayErrorText(svtkmsg.str().c_str());
  }
}

void svtkOutputWindowDisplayWarningText(
  const char* fname, int lineno, const char* message, svtkObject* sourceObj)
{
  svtkLogger::Log(svtkLogger::VERBOSITY_WARNING, fname, lineno, message);

  std::ostringstream svtkmsg;
  svtkmsg << "Warning: In " << fname << ", line " << lineno << "\n" << message << "\n\n";
  if (sourceObj && sourceObj->HasObserver(svtkCommand::WarningEvent))
  {
    sourceObj->InvokeEvent(svtkCommand::WarningEvent, const_cast<char*>(svtkmsg.str().c_str()));
  }
  else if (auto win = svtkOutputWindow::GetInstance())
  {
    svtkOutputWindowPrivateAccessor helper_raii(win);
    win->DisplayWarningText(svtkmsg.str().c_str());
  }
}

void svtkOutputWindowDisplayGenericWarningText(const char* fname, int lineno, const char* message)
{
  svtkLogger::Log(svtkLogger::VERBOSITY_WARNING, fname, lineno, message);

  if (auto win = svtkOutputWindow::GetInstance())
  {
    svtkOutputWindowPrivateAccessor helper_raii(win);
    std::ostringstream svtkmsg;
    svtkmsg << "Generic Warning: In " << fname << ", line " << lineno << "\n" << message << "\n\n";
    win->DisplayGenericWarningText(svtkmsg.str().c_str());
  }
}

void svtkOutputWindowDisplayDebugText(
  const char* fname, int lineno, const char* message, svtkObject* svtkNotUsed(sourceObj))
{
  svtkLogger::Log(svtkLogger::VERBOSITY_INFO, fname, lineno, message);

  if (auto win = svtkOutputWindow::GetInstance())
  {
    svtkOutputWindowPrivateAccessor helper_raii(win);
    std::ostringstream svtkmsg;
    svtkmsg << "Debug: In " << fname << ", line " << lineno << "\n" << message << "\n\n";
    win->DisplayDebugText(svtkmsg.str().c_str());
  }
}

svtkOutputWindowCleanup::svtkOutputWindowCleanup()
{
  ++svtkOutputWindowCleanupCounter;
}

svtkOutputWindowCleanup::~svtkOutputWindowCleanup()
{
  if (--svtkOutputWindowCleanupCounter == 0)
  {
    // Destroy any remaining output window.
    svtkOutputWindow::SetInstance(nullptr);
  }
}

svtkObjectFactoryNewMacro(svtkOutputWindow);
svtkOutputWindow::svtkOutputWindow()
{
  this->PromptUser = false;
  this->CurrentMessageType = MESSAGE_TYPE_TEXT;
  this->DisplayMode = svtkOutputWindow::DEFAULT;
  this->InStandardMacros = false;
}

svtkOutputWindow::~svtkOutputWindow() = default;

void svtkOutputWindow::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "svtkOutputWindow Single instance = " << (void*)svtkOutputWindow::Instance << endl;
  os << indent << "Prompt User: " << (this->PromptUser ? "On\n" : "Off\n");
  os << indent << "DisplayMode: ";
  switch (this->DisplayMode)
  {
    case DEFAULT:
      os << "Default\n";
      break;
    case NEVER:
      os << "Never\n";
      break;
    case ALWAYS:
      os << "Always\n";
      break;
    case ALWAYS_STDERR:
      os << "AlwaysStderr\n";
      break;
  }
}

svtkOutputWindow::StreamType svtkOutputWindow::GetDisplayStream(MessageTypes msgType) const
{
  switch (this->DisplayMode)
  {
    case DEFAULT:
      if (this->InStandardMacros && svtkLogger::IsEnabled())
      {
        return StreamType::Null;
      }
      SVTK_FALLTHROUGH;

    case ALWAYS:
      switch (msgType)
      {
        case MESSAGE_TYPE_TEXT:
          return StreamType::StdOutput;

        default:
          return StreamType::StdError;
      }

    case ALWAYS_STDERR:
      return StreamType::StdError;

    case NEVER:
    default:
      return StreamType::Null;
  }
}

// default implementation outputs to cerr only
void svtkOutputWindow::DisplayText(const char* txt)
{
  // pick correct output channel to dump text on.
  const auto stream_type = this->GetDisplayStream(this->CurrentMessageType);
  switch (stream_type)
  {
    case StreamType::StdOutput:
      cout << txt;
      break;
    case StreamType::StdError:
      cerr << txt;
      break;
    case StreamType::Null:
      break;
  }

  if (this->PromptUser && this->CurrentMessageType != MESSAGE_TYPE_TEXT &&
    stream_type != StreamType::Null)
  {
    char c = 'n';
    cerr << "\nDo you want to suppress any further messages (y,n,q)?." << endl;
    cin >> c;
    if (c == 'y')
    {
      svtkObject::GlobalWarningDisplayOff();
    }
    if (c == 'q')
    {
      this->PromptUser = 0;
    }
  }

  this->InvokeEvent(svtkCommand::MessageEvent, const_cast<char*>(txt));
  if (this->CurrentMessageType == MESSAGE_TYPE_TEXT)
  {
    this->InvokeEvent(svtkCommand::TextEvent, const_cast<char*>(txt));
  }
}

void svtkOutputWindow::DisplayErrorText(const char* txt)
{
  svtkScopedSet<MessageTypes> setter(&this->CurrentMessageType, MESSAGE_TYPE_ERROR);

  this->DisplayText(txt);
  this->InvokeEvent(svtkCommand::ErrorEvent, const_cast<char*>(txt));
}

void svtkOutputWindow::DisplayWarningText(const char* txt)
{
  svtkScopedSet<MessageTypes> setter(&this->CurrentMessageType, MESSAGE_TYPE_WARNING);

  this->DisplayText(txt);
  this->InvokeEvent(svtkCommand::WarningEvent, const_cast<char*>(txt));
}

void svtkOutputWindow::DisplayGenericWarningText(const char* txt)
{
  svtkScopedSet<MessageTypes> setter(&this->CurrentMessageType, MESSAGE_TYPE_GENERIC_WARNING);

  this->DisplayText(txt);
  this->InvokeEvent(svtkCommand::WarningEvent, const_cast<char*>(txt));
}

void svtkOutputWindow::DisplayDebugText(const char* txt)
{
  svtkScopedSet<MessageTypes> setter(&this->CurrentMessageType, MESSAGE_TYPE_DEBUG);

  this->DisplayText(txt);
}

// Return the single instance of the svtkOutputWindow
svtkOutputWindow* svtkOutputWindow::GetInstance()
{
  if (!svtkOutputWindow::Instance)
  {
    // Try the factory first
    svtkOutputWindow::Instance =
      (svtkOutputWindow*)svtkObjectFactory::CreateInstance("svtkOutputWindow");
    // if the factory did not provide one, then create it here
    if (!svtkOutputWindow::Instance)
    {
#if defined(_WIN32) && !defined(SVTK_USE_X)
      svtkOutputWindow::Instance = svtkWin32OutputWindow::New();
#elif defined(ANDROID)
      svtkOutputWindow::Instance = svtkAndroidOutputWindow::New();
#else
      svtkOutputWindow::Instance = svtkOutputWindow::New();
#endif
    }
  }
  // return the instance
  return svtkOutputWindow::Instance;
}

void svtkOutputWindow::SetInstance(svtkOutputWindow* instance)
{
  if (svtkOutputWindow::Instance == instance)
  {
    return;
  }
  // preferably this will be nullptr
  if (svtkOutputWindow::Instance)
  {
    svtkOutputWindow::Instance->Delete();
  }
  svtkOutputWindow::Instance = instance;
  if (!instance)
  {
    return;
  }
  // user will call ->Delete() after setting instance
  instance->Register(nullptr);
}

#if !defined(SVTK_LEGACY_REMOVE)
void svtkOutputWindow::SetUseStdErrorForAllMessages(bool val)
{
  SVTK_LEGACY_REPLACED_BODY(
    svtkOutputWindow::SetUseStdErrorForAllMessages, "SVTK 9.0", svtkOutputWindow::SetDisplayMode);
  this->SetDisplayMode(val ? ALWAYS_STDERR : DEFAULT);
}

bool svtkOutputWindow::GetUseStdErrorForAllMessages()
{
  SVTK_LEGACY_REPLACED_BODY(
    svtkOutputWindow::GetUseStdErrorForAllMessages, "SVTK 9.0", svtkOutputWindow::GetDisplayMode);
  return this->DisplayMode == ALWAYS_STDERR;
}

void svtkOutputWindow::UseStdErrorForAllMessagesOn()
{
  SVTK_LEGACY_REPLACED_BODY(
    svtkOutputWindow::UseStdErrorForAllMessagesOn, "SVTK 9.0", svtkOutputWindow::SetDisplayMode);
  this->SetDisplayMode(ALWAYS_STDERR);
}

void svtkOutputWindow::UseStdErrorForAllMessagesOff()
{
  SVTK_LEGACY_REPLACED_BODY(
    svtkOutputWindow::UseStdErrorForAllMessagesOff, "SVTK 9.0", svtkOutputWindow::SetDisplayMode);
  this->SetDisplayMode(DEFAULT);
}
#endif
