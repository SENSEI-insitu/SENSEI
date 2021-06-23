/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkLogger.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkLogger.h"

#include "svtkObjectFactory.h"

#if SVTK_MODULE_ENABLE_SVTK_loguru
#include <svtk_loguru.h>
#endif

#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

//=============================================================================
class svtkLogger::LogScopeRAII::LSInternals
{
public:
#if SVTK_MODULE_ENABLE_SVTK_loguru
  std::unique_ptr<loguru::LogScopeRAII> Data;
#endif
};

svtkLogger::LogScopeRAII::LogScopeRAII()
  : Internals(nullptr)
{
}

svtkLogger::LogScopeRAII::LogScopeRAII(
  svtkLogger::Verbosity verbosity, const char* fname, unsigned int lineno, const char* format, ...)
#if SVTK_MODULE_ENABLE_SVTK_loguru
  : Internals(new LSInternals())
#else
  : Internals(nullptr)
#endif
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
  va_list vlist;
  va_start(vlist, format);
  auto result = loguru::vstrprintf(format, vlist);
  va_end(vlist);
  this->Internals->Data.reset(new loguru::LogScopeRAII(
    static_cast<loguru::Verbosity>(verbosity), fname, lineno, "%s", result.c_str()));
#else
  (void)verbosity;
  (void)fname;
  (void)lineno;
  (void)format;
#endif
}

svtkLogger::LogScopeRAII::~LogScopeRAII()
{
  delete this->Internals;
}
//=============================================================================

namespace detail
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
using scope_pair = std::pair<std::string, std::shared_ptr<loguru::LogScopeRAII> >;
static std::mutex g_mutex;
static std::unordered_map<std::thread::id, std::vector<scope_pair> > g_vectors;
static std::vector<scope_pair>& get_vector()
{
  std::lock_guard<std::mutex> guard(g_mutex);
  return g_vectors[std::this_thread::get_id()];
}

static void push_scope(const char* id, std::shared_ptr<loguru::LogScopeRAII> ptr)
{
  get_vector().push_back(std::make_pair(std::string(id), ptr));
}

static void pop_scope(const char* id)
{
  auto& vector = get_vector();
  if (vector.size() > 0 && vector.back().first == id)
  {
    vector.pop_back();

    if (vector.empty())
    {
      std::lock_guard<std::mutex> guard(g_mutex);
      g_vectors.erase(std::this_thread::get_id());
    }
  }
  else
  {
    LOG_F(ERROR, "Mismatched scope! expected (%s), got (%s)", vector.back().first.c_str(), id);
  }
}
#endif
}

//=============================================================================
//----------------------------------------------------------------------------
svtkLogger::svtkLogger() {}

//----------------------------------------------------------------------------
svtkLogger::~svtkLogger() {}

//----------------------------------------------------------------------------
void svtkLogger::Init(int& argc, char* argv[], const char* verbosity_flag /*= "-v"*/)
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
  if (argc == 0)
  { // loguru::init can't handle this case -- call the no-arg overload.
    svtkLogger::Init();
    return;
  }

  loguru::g_preamble_date = false;
  loguru::g_preamble_time = false;
  loguru::init(argc, argv, verbosity_flag);
#else
  (void)argc;
  (void)argv;
  (void)verbosity_flag;
#endif
}

//----------------------------------------------------------------------------
void svtkLogger::Init()
{
  int argc = 1;
  char dummy[1] = { '\0' };
  char* argv[2] = { dummy, nullptr };
  svtkLogger::Init(argc, argv);
}

//----------------------------------------------------------------------------
void svtkLogger::SetStderrVerbosity(svtkLogger::Verbosity level)
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
  loguru::g_stderr_verbosity = static_cast<loguru::Verbosity>(level);
#else
  (void)level;
#endif
}

//----------------------------------------------------------------------------
void svtkLogger::LogToFile(
  const char* path, svtkLogger::FileMode filemode, svtkLogger::Verbosity verbosity)
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
  loguru::add_file(
    path, static_cast<loguru::FileMode>(filemode), static_cast<loguru::Verbosity>(verbosity));
#else
  (void)path;
  (void)filemode;
  (void)verbosity;
#endif
}

//----------------------------------------------------------------------------
void svtkLogger::EndLogToFile(const char* path)
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
  loguru::remove_callback(path);
#else
  (void)path;
#endif
}

//----------------------------------------------------------------------------
void svtkLogger::SetThreadName(const std::string& name)
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
  loguru::set_thread_name(name.c_str());
#else
  (void)name;
#endif
}

//----------------------------------------------------------------------------
std::string svtkLogger::GetThreadName()
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
  char buffer[128];
  loguru::get_thread_name(buffer, 128, false);
  return std::string(buffer);
#else
  return std::string("N/A");
#endif
}

//----------------------------------------------------------------------------
void svtkLogger::AddCallback(const char* id, svtkLogger::LogHandlerCallbackT callback,
  void* user_data, svtkLogger::Verbosity verbosity, svtkLogger::CloseHandlerCallbackT on_close,
  svtkLogger::FlushHandlerCallbackT on_flush)
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
  loguru::add_callback(id, reinterpret_cast<loguru::log_handler_t>(callback), user_data,
    static_cast<loguru::Verbosity>(verbosity), reinterpret_cast<loguru::close_handler_t>(on_close),
    reinterpret_cast<loguru::flush_handler_t>(on_flush));
#else
  (void)id;
  (void)callback;
  (void)user_data;
  (void)verbosity;
  (void)on_close;
  (void)on_flush;
#endif
}

//----------------------------------------------------------------------------
bool svtkLogger::RemoveCallback(const char* id)
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
  return loguru::remove_callback(id);
#else
  (void)id;
  return false;
#endif
}

//----------------------------------------------------------------------------
std::string svtkLogger::GetIdentifier(svtkObjectBase* obj)
{
  if (obj)
  {
    std::ostringstream str;
    str << obj->GetClassName() << " (" << obj << ")";
    return str.str();
  }
  return "(nullptr)";
}

//----------------------------------------------------------------------------
void svtkLogger::PrintSelf(ostream& os, svtkIndent indent)
{
  this->svtkObjectBase::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
bool svtkLogger::IsEnabled()
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
  return true;
#else
  return false;
#endif
}

//----------------------------------------------------------------------------
svtkLogger::Verbosity svtkLogger::GetCurrentVerbosityCutoff()
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
  return static_cast<svtkLogger::Verbosity>(loguru::current_verbosity_cutoff());
#else
  return VERBOSITY_INVALID; // return lowest value so no logging macros will be evaluated.
#endif
}

//----------------------------------------------------------------------------
void svtkLogger::Log(
  svtkLogger::Verbosity verbosity, const char* fname, unsigned int lineno, const char* txt)
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
  loguru::log(static_cast<loguru::Verbosity>(verbosity), fname, lineno, "%s", txt);
#else
  (void)verbosity;
  (void)fname;
  (void)lineno;
  (void)txt;
#endif
}

//----------------------------------------------------------------------------
void svtkLogger::LogF(
  svtkLogger::Verbosity verbosity, const char* fname, unsigned int lineno, const char* format, ...)
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
  va_list vlist;
  va_start(vlist, format);
  auto result = loguru::vstrprintf(format, vlist);
  va_end(vlist);
  svtkLogger::Log(verbosity, fname, lineno, result.c_str());
#else
  (void)verbosity;
  (void)fname;
  (void)lineno;
  (void)format;
#endif
}

//----------------------------------------------------------------------------
void svtkLogger::StartScope(
  Verbosity verbosity, const char* id, const char* fname, unsigned int lineno)
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
  detail::push_scope(id,
    verbosity > svtkLogger::GetCurrentVerbosityCutoff()
      ? std::make_shared<loguru::LogScopeRAII>()
      : std::make_shared<loguru::LogScopeRAII>(
          static_cast<loguru::Verbosity>(verbosity), fname, lineno, "%s", id));
#else
  (void)verbosity;
  (void)id;
  (void)fname;
  (void)lineno;
#endif
}

//----------------------------------------------------------------------------
void svtkLogger::EndScope(const char* id)
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
  detail::pop_scope(id);
#else
  (void)id;
#endif
}

//----------------------------------------------------------------------------
void svtkLogger::StartScopeF(Verbosity verbosity, const char* id, const char* fname,
  unsigned int lineno, const char* format, ...)
{
#if SVTK_MODULE_ENABLE_SVTK_loguru
  if (verbosity > svtkLogger::GetCurrentVerbosityCutoff())
  {
    detail::push_scope(id, std::make_shared<loguru::LogScopeRAII>());
  }
  else
  {
    va_list vlist;
    va_start(vlist, format);
    auto result = loguru::vstrprintf(format, vlist);
    va_end(vlist);

    detail::push_scope(id,
      std::make_shared<loguru::LogScopeRAII>(
        static_cast<loguru::Verbosity>(verbosity), fname, lineno, "%s", result.c_str()));
  }
#else
  (void)verbosity;
  (void)id;
  (void)fname;
  (void)lineno;
  (void)format;
#endif
}

//----------------------------------------------------------------------------
svtkLogger::Verbosity svtkLogger::ConvertToVerbosity(int value)
{
  if (value <= svtkLogger::VERBOSITY_INVALID)
  {
    return svtkLogger::VERBOSITY_INVALID;
  }
  else if (value > svtkLogger::VERBOSITY_MAX)
  {
    return svtkLogger::VERBOSITY_MAX;
  }
  return static_cast<svtkLogger::Verbosity>(value);
}

//----------------------------------------------------------------------------
svtkLogger::Verbosity svtkLogger::ConvertToVerbosity(const char* text)
{
  if (text != nullptr)
  {
    char* end = nullptr;
    const int ivalue = static_cast<int>(std::strtol(text, &end, 10));
    if (end != text && *end == '\0')
    {
      return svtkLogger::ConvertToVerbosity(ivalue);
    }
    if (std::string("OFF").compare(text) == 0)
    {
      return svtkLogger::VERBOSITY_OFF;
    }
    else if (std::string("ERROR").compare(text) == 0)
    {
      return svtkLogger::VERBOSITY_ERROR;
    }
    else if (std::string("WARNING").compare(text) == 0)
    {
      return svtkLogger::VERBOSITY_WARNING;
    }
    else if (std::string("INFO").compare(text) == 0)
    {
      return svtkLogger::VERBOSITY_INFO;
    }
    else if (std::string("TRACE").compare(text) == 0)
    {
      return svtkLogger::VERBOSITY_TRACE;
    }
    else if (std::string("MAX").compare(text) == 0)
    {
      return svtkLogger::VERBOSITY_MAX;
    }
  }
  return svtkLogger::VERBOSITY_INVALID;
}
