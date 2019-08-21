include "nyx_sensei_bridge.h"

#include "nyx_sensei_dataadaptor.h"

#include <vector>
#include <sensei/ConfigurableAnalysis.h>
#include <sensei/Timer.h>
#include <vtkDataObject.h>
#include <vtkNew.h>
#include <vtkSmartPointer.h>

#include <AMReX_MultiFab.H>

namespace nyx_sensei_bridge
{
static vtkSmartPointer<DataAdaptor> GlobalDataAdaptor;
static vtkSmartPointer<sensei::ConfigurableAnalysis> GlobalAnalysisAdaptor;

//-----------------------------------------------------------------------------
void initialize(MPI_Comm world,
                size_t nblocks,
                int domain_from_x, int domain_from_y, int domain_from_z,
                int domain_to_x, int domain_to_y, int domain_to_z,
                double phys_from_x, double phys_from_y, double phys_from_z,
                double phys_to_x, double phys_to_y, double phys_to_z,
                const std::string& config_file)
{
  sensei::Timer::MarkEvent mark("sensei_bridge::initialize");
  GlobalDataAdaptor = vtkSmartPointer<DataAdaptor>::New();
  GlobalDataAdaptor->Initialize(nblocks);
  GlobalDataAdaptor->SetDataTimeStep(-1);

  int dext[6] = { domain_from_x, domain_to_x, domain_from_y, domain_to_y, domain_from_z, domain_to_z};
  GlobalDataAdaptor->SetDataExtent(dext);

  double pext[6] = { phys_from_x, phys_to_x, phys_from_y, phys_to_y, phys_from_z, phys_to_z };
  GlobalDataAdaptor->SetPhysicalExtents(pext);

  GlobalDataAdaptor->ComputeSpacingAndOrigin();
  GlobalAnalysisAdaptor = vtkSmartPointer<sensei::ConfigurableAnalysis>::New();
  GlobalAnalysisAdaptor->Initialize(world, config_file);
}

//-----------------------------------------------------------------------------
//
void analyze(const amrex::MultiFab& simulation_data, amrex::Real time, int time_step)
{
  GlobalDataAdaptor->SetDataTime(time);
  GlobalDataAdaptor->SetDataTimeStep(time_step);

  sensei::Timer::MarkStartEvent("sensei_bridge::copy-data");
  {
    for (amrex::MFIter mfi(simulation_data); mfi.isValid(); ++mfi)
    {
#ifdef NYX_SENSEI_NO_COPY
      amrex::Box data_box = mfi.fabbox();

      GlobalDataAdaptor->SetValidBlockExtent(mfi.index(),
          mfi.validbox().smallEnd(0), mfi.validbox().bigEnd(0),
          mfi.validbox().smallEnd(1), mfi.validbox().bigEnd(1),
          mfi.validbox().smallEnd(2), mfi.validbox().bigEnd(2));

      GlobalDataAdaptor->SetBlockData(mfi.index(), simulation_data[mfi].dataPtr(0));

      //std::cout << "Box " << mfi.index() << " overall min: " << simulation_data[mfi].min(0) << " valid box min: " << simulation_data[mfi].min(mfi.validbox(),0);
      //std::cout << " overall max: " << simulation_data[mfi].max(0) << " valid box max: " << simulation_data[mfi].max(mfi.validbox(),0) << std::endl;;

#else
      amrex::Box data_box = mfi.validbox();

      float *data = new float[mfi.validbox().numPts()];
      float *currPtr = data;
      for (IntVect iv = mfi.validbox().smallEnd(); iv <= mfi.validbox().bigEnd(); mfi.validbox().next(iv))
      {
        *(currPtr++) = simulation_data[mfi](iv);
      }
      BL_ASSERT((currPtr - data) == mfi.validbox().numPts());
      GlobalDataAdaptor->SetBlockData(mfi.index(), data);
#endif

      GlobalDataAdaptor->SetBlockExtent(mfi.index(),
          data_box.smallEnd(0), data_box.bigEnd(0),
          data_box.smallEnd(1), data_box.bigEnd(1),
          data_box.smallEnd(2), data_box.bigEnd(2));
    }
  }

  Real range[2] = { simulation_data.min(0), simulation_data.max(0) };
  //if (ParallelDescriptor::IOProcessor()) std::cout << "Global value range is " << range[0] << " to " << range[1] << std::endl;

  sensei::Timer::MarkEndEvent("sensei_bridge::copy-data");
  sensei::Timer::MarkStartEvent("sensei_bridge::analyze");
  GlobalAnalysisAdaptor->Execute(GlobalDataAdaptor.GetPointer());
  sensei::Timer::MarkEndEvent("sensei_bridge::analyze");

  sensei::Timer::MarkStartEvent("sensei_bridge::release-data");
  GlobalDataAdaptor->ReleaseData();
  sensei::Timer::MarkEndEvent("sensei_bridge::release-data");
}

//-----------------------------------------------------------------------------
void finalize()
{
  sensei::Timer::MarkStartEvent("sensei_bridge::finalize");
  GlobalAnalysisAdaptor = NULL;
  GlobalDataAdaptor = NULL;
  sensei::Timer::MarkEndEvent("sensei_bridge::finalize");
}
}
