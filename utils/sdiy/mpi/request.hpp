namespace sdiy
{
namespace mpi
{
  struct request
  {
    inline
    status              wait();
    inline
    optional<status>    test();
    inline
    void                cancel();

    MPI_Request         r;
  };
}
}

sdiy::mpi::status
sdiy::mpi::request::wait()
{
#ifndef DIY_NO_MPI
  status s;
  MPI_Wait(&r, &s.s);
  return s;
#else
  DIY_UNSUPPORTED_MPI_CALL(sdiy::mpi::request::wait);
#endif
}

sdiy::mpi::optional<sdiy::mpi::status>
sdiy::mpi::request::test()
{
#ifndef DIY_NO_MPI
  status s;
  int flag;
  MPI_Test(&r, &flag, &s.s);
  if (flag)
    return s;
#endif
  return optional<status>();
}

void
sdiy::mpi::request::cancel()
{
#ifndef DIY_NO_MPI
  MPI_Cancel(&r);
#endif
}
