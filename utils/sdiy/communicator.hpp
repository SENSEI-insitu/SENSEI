#ifndef DIY_COMMUNICATOR_HPP
#define DIY_COMMUNICATOR_HPP

#warning "sdiy::Communicator (in sdiy/communicator.hpp) is deprecated, use sdiy::mpi::communicator directly"

#include "mpi.hpp"

namespace sdiy
{
  typedef mpi::communicator         Communicator;
}

#endif
