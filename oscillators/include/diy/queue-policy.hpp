#ifndef DIY_QUEUE_POLICY_HPP
#define DIY_QUEUE_POLICY_HPP


namespace diy
{
  struct Master::QueuePolicy
  {
    virtual ~QueuePolicy()                              {}
    bool    unload_incoming(const Master& master, int from, int to, size_t size) const  =0;
    bool    unload_outgoing(const Master& master, int from, size_t size) const          =0;
  };

  struct Master::QueueSizePolicy: public QueuePolicy
  {
            QueueSizePolicy(size_t sz): size(sz)          {}
    bool    unload_incoming(const Master& master, int from, int to, size_t sz) const    { return sz > size; }
    bool    unload_outgoing(const Master& master, int from, size_t sz) const            { return sz > size; }       // FIXME

    size_t  size;
  };
}

#endif
