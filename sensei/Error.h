#ifndef sensei_Common_h
#define sensei_Common_h

#include "senseiConfig.h"
#include <iostream>

namespace sensei
{
// detect if we are writing to a tty, if not then
// we should not use ansi color codes
int haveTty();

// a helper class for debug and error
// messages
class parallelId {};

// print the callers rank and thread id
// to the given stream
std::ostream &operator<<(std::ostream &os, const parallelId &id);

int ioEnabled(int active_rank);
}

#define ANSI_RED "\033[1;31m"
#define ANSI_GREEN "\033[1;32m"
#define ANSI_YELLOW "\033[1;33m"
#define ANSI_WHITE "\033[1;37m"
#define ANSI_OFF "\033[0m"

#define BEGIN_HL(Color) (sensei::haveTty()?Color:"")
#define END_HL (sensei::haveTty()?ANSI_OFF:"")

#define SENSEI_MESSAGE(Rank, Head, HeadColor, Msg)                              \
if (sensei::ioEnabled(Rank))                                                            \
{                                                                               \
  std::cerr << BEGIN_HL(HeadColor) << Head << END_HL << " ["                    \
    << sensei::parallelId() << "][" << __FILE__ << ":" << __LINE__              \
    << "][" SENSEI_VERSION "]" << std::endl << BEGIN_HL(HeadColor) << Head      \
    << END_HL << " "  << BEGIN_HL(ANSI_WHITE) << "" Msg << END_HL << std::endl; \
}

#define SENSEI_ERROR(Msg) SENSEI_MESSAGE(-1, "ERROR:", ANSI_RED, Msg)
#define SENSEI_WARNING(Msg) SENSEI_MESSAGE(-1, "WARNING:", ANSI_YELLOW, Msg)
#define SENSEI_STATUS(Msg) SENSEI_MESSAGE(0, "STATUS:", ANSI_GREEN, Msg)

#endif
