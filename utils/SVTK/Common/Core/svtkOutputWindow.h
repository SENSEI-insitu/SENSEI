/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkOutputWindow.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkOutputWindow
 * @brief   base class for writing debug output to a console
 *
 * This class is used to encapsulate all text output, so that it will work
 * with operating systems that have a stdout and stderr, and ones that
 * do not.  (i.e windows does not).  Sub-classes can be provided which can
 * redirect the output to a window.
 */

#ifndef svtkOutputWindow_h
#define svtkOutputWindow_h

#include "svtkCommonCoreModule.h"  // For export macro
#include "svtkDebugLeaksManager.h" // Must be included before singletons
#include "svtkObject.h"

class SVTKCOMMONCORE_EXPORT svtkOutputWindowCleanup
{
public:
  svtkOutputWindowCleanup();
  ~svtkOutputWindowCleanup();

private:
  svtkOutputWindowCleanup(const svtkOutputWindowCleanup& other) = delete;
  svtkOutputWindowCleanup& operator=(const svtkOutputWindowCleanup& rhs) = delete;
};

class svtkOutputWindowPrivateAccessor;
class SVTKCOMMONCORE_EXPORT svtkOutputWindow : public svtkObject
{
public:
  // Methods from svtkObject
  svtkTypeMacro(svtkOutputWindow, svtkObject);
  /**
   * Print ObjectFactor to stream.
   */
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Creates a new instance of svtkOutputWindow. Note this *will* create a new
   * instance using the svtkObjectFactor. If you want to access the global
   * instance, use `GetInstance` instead.
   */
  static svtkOutputWindow* New();

  /**
   * Return the singleton instance with no reference counting.
   */
  static svtkOutputWindow* GetInstance();
  /**
   * Supply a user defined output window. Call ->Delete() on the supplied
   * instance after setting it.
   */
  static void SetInstance(svtkOutputWindow* instance);

  //@{
  /**
   * Display the text. Four virtual methods exist, depending on the type of
   * message to display. This allows redirection or reformatting of the
   * messages. The default implementation uses DisplayText for all.
   * Consequently, subclasses can simply override DisplayText and use
   * `GetCurrentMessageType` to determine the type of message that's being reported.
   */
  virtual void DisplayText(const char*);
  virtual void DisplayErrorText(const char*);
  virtual void DisplayWarningText(const char*);
  virtual void DisplayGenericWarningText(const char*);
  virtual void DisplayDebugText(const char*);
  //@}

  //@{
  /**
   * If PromptUser is set to true then each time a line of text
   * is displayed, the user is asked if they want to keep getting
   * messages.
   *
   * Note that PromptUser has not effect of messages displayed by directly
   * calling `DisplayText`. The prompt is never shown for such messages.
   *
   */
  svtkBooleanMacro(PromptUser, bool);
  svtkSetMacro(PromptUser, bool);
  //@}

  //@{
  /**
   * Historically (SVTK 8.1 and earlier), when printing messages to terminals,
   * svtkOutputWindow would always post messages to `cerr`. Setting this to true
   * restores that incorrect behavior. When false (default),
   * svtkOutputWindow uses `cerr` for debug, error and warning messages, and
   * `cout` for text messages.
   *
   * @deprecated use `SetDisplayModeToAlwaysStdErr` instead.
   */
  SVTK_LEGACY(void SetUseStdErrorForAllMessages(bool));
  SVTK_LEGACY(bool GetUseStdErrorForAllMessages());
  SVTK_LEGACY(void UseStdErrorForAllMessagesOn());
  SVTK_LEGACY(void UseStdErrorForAllMessagesOff());
  //@}

  //@{
  /**
   * Flag indicates how the svtkOutputWindow handles displaying of text to
   * `stderr` / `stdout`. Default is `DEFAULT` except in
   * `svtkWin32OutputWindow` where on non dashboard runs, the default is
   * `NEVER`.
   *
   * `NEVER` indicates that the messages should never be forwarded to the
   * standard output/error streams.
   *
   * `ALWAYS` will result in error/warning/debug messages being posted to the
   * standard error stream, while text messages to standard output stream.
   *
   * `ALWAYS_STDERR` will result in all messages being posted to the standard
   * error stream (this was default behavior in SVTK 8.1 and earlier).
   *
   * `DEFAULT` is similar to `ALWAYS` except when logging is enabled. If
   * logging is enabled, messages posted to the output window using SVTK error/warning macros such as
   * `svtkErrorMacro`, `svtkWarningMacro` etc. will not posted on any of the output streams. This is
   * done to avoid duplicate messages on these streams since these macros also result in add items
   * to the log.
   *
   * @note svtkStringOutputWindow does not result this flag as is never forwards
   * any text to the output streams.
   */
  enum DisplayModes
  {
    DEFAULT = -1,
    NEVER = 0,
    ALWAYS = 1,
    ALWAYS_STDERR = 2
  };
  svtkSetClampMacro(DisplayMode, int, DEFAULT, ALWAYS_STDERR);
  svtkGetMacro(DisplayMode, int);
  void SetDisplayModeToDefault() { this->SetDisplayMode(svtkOutputWindow::DEFAULT); }
  void SetDisplayModeToNever() { this->SetDisplayMode(svtkOutputWindow::NEVER); }
  void SetDisplayModeToAlways() { this->SetDisplayMode(svtkOutputWindow::ALWAYS); }
  void SetDisplayModeToAlwaysStdErr() { this->SetDisplayMode(svtkOutputWindow::ALWAYS_STDERR); }
  //@}
protected:
  svtkOutputWindow();
  ~svtkOutputWindow() override;

  enum MessageTypes
  {
    MESSAGE_TYPE_TEXT,
    MESSAGE_TYPE_ERROR,
    MESSAGE_TYPE_WARNING,
    MESSAGE_TYPE_GENERIC_WARNING,
    MESSAGE_TYPE_DEBUG
  };

  /**
   * Returns the current message type. Useful in subclasses that simply want to
   * override `DisplayText` and also know what type of message is being
   * processed.
   */
  svtkGetMacro(CurrentMessageType, MessageTypes);

  enum class StreamType
  {
    Null,
    StdOutput,
    StdError,
  };

  /**
   * Returns the standard output stream to post the message of the given type
   * on.
   */
  virtual StreamType GetDisplayStream(MessageTypes msgType) const;

  bool PromptUser;

private:
  static svtkOutputWindow* Instance;
  MessageTypes CurrentMessageType;
  int DisplayMode;
  int InStandardMacros; // used to suppress display to output streams from standard macros when
                        // logging is enabled.

  friend class svtkOutputWindowPrivateAccessor;

private:
  svtkOutputWindow(const svtkOutputWindow&) = delete;
  void operator=(const svtkOutputWindow&) = delete;
};

// Uses schwartz counter idiom for singleton management
static svtkOutputWindowCleanup svtkOutputWindowCleanupInstance;

#endif
