#ifndef DIY_QUEUES_HPP
#define DIY_QUEUES_HPP

namespace diy
{
  class Queues
  {


      IncomingQueues&       get_incoming(int lid);          // load all the queues for this processor (if necessary)
      OutgoingQueues&       get_outgoing(int lid);          // load if necessary


                            unload(int lid);

  };
}

#endif
