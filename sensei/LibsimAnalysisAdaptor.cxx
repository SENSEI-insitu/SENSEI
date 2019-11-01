#include "LibsimAnalysisAdaptor.h"
#include "LibsimImageProperties.h"
#include "DataAdaptor.h"
#include "MeshMetadata.h"
#include "VTKUtils.h"
#include "STLUtils.h"
#include "MPIUtils.h"
#include "Profiler.h"
#include "Error.h"
#include "BinaryStream.h"

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCharArray.h>
#include <vtkCompositeDataIterator.h>
#include <vtkCompositeDataSet.h>
#include <vtkDataArray.h>
#include <vtkDataObject.h>
#include <vtkImageData.h>
#include <vtkIntArray.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkRectilinearGrid.h>
#include <vtkStructuredGrid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkAMRBox.h>
#include <vtkAMRInformation.h>
#include <vtkOverlappingAMR.h>
#include <vtkUniformGrid.h>
#include <vtkDataObject.h>
#include <vtkXMLUniformGridAMRWriter.h>
#include <vtkUniformGridAMRDataIterator.h>

#include <VisItControlInterface_V2.h>
#include <VisItDataInterface_V2.h>

#include <sstream>
#include <algorithm>
#include <map>

#include <mpi.h>


// TODO -- this is temporary debugging code related to visit gui amr
// patch/level selection
//#define USE_REAL_DOMAIN

#define VISIT_DEBUG_LOG

#define VISIT_COMMAND_PROCESS 0
#define VISIT_COMMAND_SUCCESS 1
#define VISIT_COMMAND_FAILURE 2

using vtkDataObjectPtr = vtkSmartPointer<vtkDataObject>;

namespace sensei
{
using namespace sensei::STLUtils;

///////////////////////////////////////////////////////////////////////////////
class PlotRecord
{
public:
    PlotRecord() : frequency(5), imageProps(), session(), plots(), plotVars(), doExport(false), slice(false), project2d(false)
    {
        origin[0] = origin[1] = origin[2] = 0.;
        normal[0] = 1.; normal[1] = normal[2] = 0.;
    }

    ~PlotRecord()
    {
    }

    static std::vector<std::string> SplitAtCommas(const std::string &s)
    {
        std::stringstream ss(s);
        std::vector<std::string> result;
        while(ss.good())
        {
           std::string substr;
           getline(ss, substr, ',' );
           result.push_back(substr);
        }
        return result;
    }

    int frequency;
    LibsimImageProperties imageProps;
    std::string           session;
    std::vector<std::string> plots;
    std::vector<std::string> plotVars;
    bool doExport;
    bool slice;
    bool project2d;
    double origin[3];
    double normal[3];
};

std::ostream &operator << (std::ostream &os, const PlotRecord &obj)
{
    os << "{session=\"" << obj.session << "\", plots=[";
    for(size_t i = 0; i < obj.plots.size(); ++i)
    {
        if(i > 0)
           os << ", ";
        os << "\"" << obj.plots[i] << "\"";
    }
    os << "], ";
    os << "plotvars=[";
    for(size_t i = 0; i < obj.plotVars.size(); ++i)
    {
        if(i > 0)
           os << ", ";
        os << "\"" << obj.plotVars[i] << "\"";
    }
    os << "], ";
    if(obj.doExport)
    {
        os << "filename=\"" << obj.imageProps.GetFilename() << ", ";
    }
    else
    {
        os << "filename=\"" << obj.imageProps.GetFilename() << ", ";
        os << "width=" << obj.imageProps.GetWidth() << ", ";
        os << "height=" << obj.imageProps.GetHeight() << ", ";
        os << "format=" << obj.imageProps.GetFormat() << ", ";
    }
    os << "slice=" << (obj.slice?"true":"false") << ", ";
    os << "project2d=" << (obj.project2d?"true":"false") << ", ";
    os << "origin=[" << obj.origin[0] << ", " << obj.origin[1] << ", " << obj.origin[2] << "], ";
    os << "normal=[" << obj.normal[0] << ", " << obj.normal[1] << ", " << obj.normal[2] << "]}";
    return os;
}

///////////////////////////////////////////////////////////////////////////////
class LibsimAnalysisAdaptor::PrivateData
{
public:
    PrivateData();
    ~PrivateData();

    void SetTraceFile(const std::string &s);
    void SetOptions(const std::string &s);
    void SetVisItDirectory(const std::string &s);
    void SetComm(MPI_Comm Comm);
    void SetMode(const std::string &mode);
    void SetComputeNesting(int v);

    void PrintSelf(ostream& os, vtkIndent indent);

    bool Initialize();

    bool Execute(sensei::DataAdaptor *DataAdaptor);

    bool AddRender(int freq, const std::string &session,
                  const std::string &plots,
                  const std::string &plotVars,
                  bool slice, bool project2d,
                  const double origin[3], const double normal[3],
	              const LibsimImageProperties &imgProps);

    bool AddExport(int freq, const std::string &session,
                  const std::string &plots,
                  const std::string &plotVars,
                  bool slice, bool project2d,
                  const double origin[3], const double normal[3],
	              const std::string &filename);

private:
    // libsim callbacks.
    static int broadcast_int(int *value, int sender, void *cbdata);
    static int broadcast_string(char *str, int len, int sender, void *cbdata);
    static void ControlCommandCallback(const char *cmd, const char *args, void *cbdata);
    static void SlaveProcessCallback(void *cbdata);
    static visit_handle GetMetaData(void *cbdata);
    static visit_handle GetMesh(int dom, const char *name, void *cbdata);
    static visit_handle GetVariable(int dom, const char *name, void *cbdata);
    static visit_handle GetDomainList(const char *name, void *cbdata);
    static visit_handle GetDomainNesting(const char *name, void *cbdata);

    // helpers
    int ProcessVisItCommand(int rank);
    bool Execute_Batch(int rank);
    bool Execute_Interactive(int rank);

    int GetMesh(const std::string &meshName, vtkDataObjectPtr &cdp);
    int GetMesh(int dom, const std::string &meshName, vtkDataObject *&mesh);
    int GetVariable(int dom, const std::string &varName, vtkDataArray *&array);

    int DecodeVarName(const std::string &varName, std::string &meshName,
        std::string &arrayName, int &association);

    int TopologicalDimension(const int dims[3]) const;

    std::string MakeFileName(const std::string &f,
        int timestep, double time) const;

    void DetermineExportFilename(const std::string &f,
        std::string &fnoext, std::string &fmt) const;

    void ClearCache();

    // state
    MPI_Comm Comm;
    sensei::DataAdaptor *Adaptor;

    std::map<std::string, vtkDataObjectPtr> Meshes;
    std::map<std::string, sensei::MeshMetadataPtr> Metadata;

    int ComputeNesting;

    std::string               traceFile, options, visitdir;
    std::vector<PlotRecord>   plots;
    std::string               mode;
    bool                      paused;
    static bool               initialized;
    static int                instances;
};

bool LibsimAnalysisAdaptor::PrivateData::initialized = false;
int  LibsimAnalysisAdaptor::PrivateData::instances = 0;

// --------------------------------------------------------------------------
LibsimAnalysisAdaptor::PrivateData::PrivateData() : Comm(MPI_COMM_WORLD),
  Adaptor(nullptr), traceFile(), options(), visitdir(),
  mode("batch"), paused(false)
{
    ++instances;
}

// --------------------------------------------------------------------------
LibsimAnalysisAdaptor::PrivateData::~PrivateData()
{
    --instances;

    if(instances == 0 && initialized)
    {
        TimeEvent<128> mark("libsim::finalize");
        if(VisItIsConnected())
            VisItDisconnect();
    }
}

// --------------------------------------------------------------------------
void
LibsimAnalysisAdaptor::PrivateData::SetComputeNesting(int val)
{
    this->ComputeNesting = val;
}

// --------------------------------------------------------------------------
void
LibsimAnalysisAdaptor::PrivateData::SetTraceFile(const std::string &s)
{
    traceFile = s;
}

// --------------------------------------------------------------------------
void
LibsimAnalysisAdaptor::PrivateData::SetOptions(const std::string &s)
{
    options = s;
}

// --------------------------------------------------------------------------
void
LibsimAnalysisAdaptor::PrivateData::SetVisItDirectory(const std::string &s)
{
    visitdir = s;
}

// --------------------------------------------------------------------------
void
LibsimAnalysisAdaptor::PrivateData::SetComm(MPI_Comm c)
{
    Comm = c;
}

// --------------------------------------------------------------------------
void
LibsimAnalysisAdaptor::PrivateData::SetMode(const std::string &m)
{
    mode = m;
}

// --------------------------------------------------------------------------
void
LibsimAnalysisAdaptor::PrivateData::PrintSelf(ostream &os, vtkIndent)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(Comm, &rank);
    MPI_Comm_rank(Comm, &size);

    if (rank == 0)
    {
        os << "traceFile = " << traceFile << endl;
        os << "options = " << options << endl;
        os << "visitdir = " << visitdir << endl;
        os << "mode = " << mode << endl;
        os << "initialized = " << (initialized ? "true" : "false") << endl;
    }
}

// --------------------------------------------------------------------------
bool
LibsimAnalysisAdaptor::PrivateData::AddRender(int freq,
    const std::string &session,
    const std::string &plts,
    const std::string &plotVars,
    bool slice, bool project2d,
    const double origin[3], const double normal[3],
    const LibsimImageProperties &imgProps)
{
    PlotRecord p;
    p.frequency = freq;
    p.imageProps = imgProps;
    p.session = session;
    p.plots = PlotRecord::SplitAtCommas(plts);
    p.plotVars = PlotRecord::SplitAtCommas(plotVars);
    p.slice = slice;
    p.project2d = project2d;
    memcpy(p.origin, origin, 3 * sizeof(double));
    memcpy(p.normal, normal, 3 * sizeof(double));

    bool retval = false;
    if(!p.plots.empty() && (p.plots.size() == p.plotVars.size()))
      {
      plots.push_back(p);
      retval = true;
      }
    if(!session.empty())
      retval = true;
//    cout << "Libsim Render: " << (retval?"true":"false") << ", " << p << endl;
    return retval;
}

// --------------------------------------------------------------------------
bool
LibsimAnalysisAdaptor::PrivateData::AddExport(int freq,
    const std::string &session,
    const std::string &plts,
    const std::string &plotVars,
    bool slice, bool project2d,
    const double origin[3], const double normal[3],
    const std::string &filename)
{
    PlotRecord p;
    p.frequency = freq;
    p.doExport = true;
    p.imageProps.SetFilename(filename);
    std::vector<std::string> plotTypes = PlotRecord::SplitAtCommas(plts);
    std::vector<std::string> first;
    first.push_back(plotTypes[0]);
    p.session = session;
    p.plots = first;
    p.plotVars = PlotRecord::SplitAtCommas(plotVars);
    p.slice = slice;
    p.project2d = project2d;
    memcpy(p.origin, origin, 3 * sizeof(double));
    memcpy(p.normal, normal, 3 * sizeof(double));

    bool retval = false;
    if(!p.plots.empty() && !p.plotVars.empty())
    {
        retval = true;
        plots.push_back(p);
    }
    if(!session.empty())
        retval = true;

    return retval;
}

// --------------------------------------------------------------------------
bool
LibsimAnalysisAdaptor::PrivateData::Initialize()
{
    if (initialized)
        return true;

     TimeEvent<128> mark("libsim::initialize");
#ifdef VISIT_DEBUG_LOG
    VisItDebug5("SENSEI: LibsimAnalysisAdaptor::PrivateData::Initialize\n");
#endif

     int rank = 0, size = 1;
     MPI_Comm_rank(Comm, &rank);
     MPI_Comm_size(Comm, &size);

     if(!traceFile.empty())
     {
         char suffix[100];
         snprintf(suffix, 100, ".%04d", rank);
         VisItOpenTraceFile((traceFile + suffix).c_str());
     }

     if(!options.empty())
         VisItSetOptions(const_cast<char*>(options.c_str()));

     if(!visitdir.empty())
         VisItSetDirectory(const_cast<char *>(visitdir.c_str()));

     // Install callback functions for global communication.
     VisItSetBroadcastIntFunction2(broadcast_int, this);
     VisItSetBroadcastStringFunction2(broadcast_string, this);

     // Tell libsim whether the simulation is parallel.
     VisItSetParallel(size > 1);
     VisItSetParallelRank(rank);

     // Install Comm into VisIt.
     VisItSetMPICommunicator((void *)&Comm);

     // Set up the environment.
     char *env = nullptr;
     if(rank == 0)
         env = VisItGetEnvironment();
     VisItSetupEnvironment2(env);
     if(env != nullptr)
         free(env);

     bool i0 = mode == "interactive";
     bool i1 = mode == "interactive,paused";
     if(i0 || i1)
     {
         // We can start paused if desired.
         this->paused = i1;

         // Write out .sim file that VisIt uses to connect.
         if (rank == 0)
         {
             VisItInitializeSocketAndDumpSimFile(
                 "sensei",
                 "Connected via SENSEI",
                 "/path/to/where/sim/was/started",
                 NULL, NULL, "sensei.sim2");
         }

         initialized = true;
     }
     else
     {
        // Try and initialize the runtime.
        if(VisItInitializeRuntime() == VISIT_ERROR)
        {
            SENSEI_ERROR("Could not initialize the VisIt runtime library.")
            return false;
        }
        else
        {
            // Register Libsim callbacks.
            VisItSetSlaveProcessCallback2(SlaveProcessCallback, (void*)this); // needed in batch?
            VisItSetGetMetaData(GetMetaData, (void*)this);
            VisItSetGetMesh(GetMesh, (void*)this);
            VisItSetGetVariable(GetVariable, (void*)this);
            VisItSetGetDomainList(GetDomainList, (void*)this);
            if (this->ComputeNesting)
                VisItSetGetDomainNesting(GetDomainNesting, (void*)this);

            initialized = true;
        }
    }

    return initialized;
}

// --------------------------------------------------------------------------
std::string
LibsimAnalysisAdaptor::PrivateData::MakeFileName(const std::string &f, int timestep, double time) const
{
    std::string filename(f);

    char ts5[20];
    sprintf(ts5, "%05d", timestep);

    // replace "%ts" with timestep in filename
    std::string::size_type pos = filename.find("%ts");
    while (pos != std::string::npos)
    {
        filename.replace(pos, 3, ts5);
        pos = filename.find("%ts");
    }
    // replace "%t" with time in filename
    std::ostringstream t_stream;
    t_stream << time;
    pos = filename.find("%t");
    while (pos != std::string::npos)
    {
        filename.replace(pos, 2, t_stream.str());
        pos = filename.find("%t");
    }
    return filename;
}

// --------------------------------------------------------------------------
static bool EndsWith(const std::string &s, const std::string &ext)
{
    bool retval = false;
    if(s.size() >= ext.size() && !ext.empty())
    {
        retval = s.substr(s.size() - ext.size(), ext.size()) == ext;
    }
    return retval;
}

// --------------------------------------------------------------------------
void
LibsimAnalysisAdaptor::PrivateData::DetermineExportFilename(const std::string &f,
    std::string &fnoext, std::string &fmt) const
{
    // This is kind of a hack. We don't have a mechanism to interrogate the
    // format from the filename... Maybe VisIt should be doing this.
    if(EndsWith(f, ".silo"))
    {
        fnoext = f.substr(0, f.size() - 5);
        fmt = "Silo_1.0";
    }
    else if(EndsWith(f, ".xdb"))
    {
        fnoext = f;
        fmt = "FieldViewXDB_1.0";
    }
    else if(EndsWith(f, ".raw"))
    {
        fnoext = f.substr(0, f.size() - 4);
        fmt = "RAW_1.0";
    }
    else if(EndsWith(f, ".tec") || EndsWith(f, ".plt"))
    {
        fnoext = f.substr(0, f.size() - 4);
        fmt = "Tecplot_1.0";
    }
    else if(EndsWith(f, ".ply"))
    {
        fnoext = f.substr(0, f.size() - 4);
        fmt = "PLY_1.0";
    }
    else if(EndsWith(f, ".stl"))
    {
        fnoext = f.substr(0, f.size() - 4);
        fmt = "STL_1.0";
    }
    else if(EndsWith(f, ".obj"))
    {
        fnoext = f.substr(0, f.size() - 4);
        fmt = "WavefrontOBJ_1.0";
    }
    else if(EndsWith(f, ".bov"))
    {
        fnoext = f.substr(0, f.size() - 4);
        fmt = "BOV_1.0";
    }
    else
    {
        fnoext = f; // The VTK writer makes ok filenames.
        fmt = "VTK_1.0";
    }
}

// --------------------------------------------------------------------------
bool
LibsimAnalysisAdaptor::PrivateData::Execute(sensei::DataAdaptor *DataAdaptor)
{
#ifdef VISIT_DEBUG_LOG
    VisItDebug5("SENSEI: LibsimAnalysisAdaptor::PrivateData::Execute\n");
#endif

    if (!initialized)
    {
        SENSEI_ERROR("Execute before initialization")
        return false;
    }

    // Keep a pointer to the data adaptor so the callbacks can access it.
    Adaptor = DataAdaptor;

    int rank = 0;
    MPI_Comm_rank(Comm, &rank);

    // Let visit get new metadata.
    VisItTimeStepChanged();

    // Execute
    bool retval = true;
    if (mode.substr(0, 11) == "interactive")
        retval = Execute_Interactive(rank);
    else
        retval = Execute_Batch(rank);

    // during execution data and metadata are cached due to
    // the differnece between how sensei presents data and
    // visit consumes it. you must clear the cache after each
    // execute.
    ClearCache();

    return retval;
}

// --------------------------------------------------------------------------
bool
LibsimAnalysisAdaptor::PrivateData::Execute_Batch(int rank)
{
    bool retval = true;

    // Now that the runtime stuff is loaded, we can execute some plots.
    for(size_t i = 0; i < plots.size(); ++i)
    {
        // Skip if we're not executing now.
        if(Adaptor->GetDataTimeStep() % plots[i].frequency != 0)
            continue;

        // If we have a session file for this plot output, then add it now.
        if(!plots[i].session.empty())
        {
            VisItRestoreSession(plots[i].session.c_str());
        }
        else if(!plots[i].plots.empty())
        {
            // Add all of the plots in this group.
            // For now, disallow sessions + plots since we are unable to query the number
            // of plots that were created using the session.

            int *ap = new int[plots[i].plots.size()];
            int np = 0;
            for(size_t j = 0; j < plots[i].plots.size(); ++j)
            {
                if(VisItAddPlot(plots[i].plots[j].c_str(),plots[i].plotVars[j].c_str()) == VISIT_OKAY)
                {
                    // Use a better color table.
                    const char *ctName = "hot_desaturated";
                    if(plots[i].plots[j] == "Pseudocolor")
                        VisItSetPlotOptionsS("colorTableName", ctName);
                    else if(plots[i].plots[j] == "Vector")
                    {
                        VisItSetPlotOptionsS("colorTableName", ctName);
                        VisItSetPlotOptionsB("colorByMag", true);
                    }

                   ap[np] = np;
                   np++;
                }
                else if(rank == 0)
                {
                    SENSEI_ERROR("VisItAddPlot failed.")
                }
            }

            // Select all plots.
            VisItSetActivePlots(ap, np);
            delete [] ap;

            // Add a slice operator to all plots (not from session).
            if(plots[i].slice)
            {
                VisItAddOperator("Slice", 1);
                VisItSetOperatorOptionsI("originType", 0); // point intercept
                VisItSetOperatorOptionsDv("originPoint", plots[i].origin, 3);
                VisItSetOperatorOptionsDv("normal", plots[i].normal, 3);
                VisItSetOperatorOptionsB("project2d", plots[i].project2d ? 1 : 0);
            }
        }

        if(VisItDrawPlots() == VISIT_OKAY)
        {
            std::string filename;
            filename = MakeFileName(plots[i].imageProps.GetFilename(),
                                    Adaptor->GetDataTimeStep(),
                                    Adaptor->GetDataTime());

            if(plots[i].doExport)
            {
                std::string fmt, filename_no_ext;
                DetermineExportFilename(filename, filename_no_ext, fmt);
                visit_handle vars = VISIT_INVALID_HANDLE;
                if(VisIt_NameList_alloc(&vars))
                {
                    for(size_t v = 0; v < plots[i].plotVars.size(); ++v)
                        VisIt_NameList_addName(vars, plots[i].plotVars[v].c_str());

                    // Export the data instead of rendering it.
                    if(VisItExportDatabase(filename_no_ext.c_str(),
                                           fmt.c_str(), vars) != VISIT_OKAY)
                    {
                        if(rank == 0)
                            SENSEI_ERROR("VisItExportDatabase failed.")
                        retval = false;
                    }

                    VisIt_NameList_free(vars);
                }
                else
                {
                    if(rank == 0)
                        SENSEI_ERROR("VisIt_NameList_alloc failed.")
                    retval = false;
                }
            }
            else
            {
                // Get the image properties.
                int w = plots[i].imageProps.GetWidth();
                int h = plots[i].imageProps.GetHeight();
                int format = VISIT_IMAGEFORMAT_PNG;
                if(plots[i].imageProps.GetFormat() == "bmp")
                    format = VISIT_IMAGEFORMAT_BMP;
                else if(plots[i].imageProps.GetFormat() == "jpeg")
                    format = VISIT_IMAGEFORMAT_JPEG;
                else if(plots[i].imageProps.GetFormat() == "png")
                    format = VISIT_IMAGEFORMAT_PNG;
                else if(plots[i].imageProps.GetFormat() == "ppm")
                    format = VISIT_IMAGEFORMAT_PPM;
                else if(plots[i].imageProps.GetFormat() == "tiff")
                    format = VISIT_IMAGEFORMAT_TIFF;

                // Save an image.
                if(VisItSaveWindow(filename.c_str(), w, h, format) != VISIT_OKAY)
                {
                    if(rank == 0)
                        SENSEI_ERROR("VisItSaveWindow failed.")
                    retval = false;
                }
            } // doExport
        }
        else
        {
            if(rank == 0)
                SENSEI_ERROR("VisItDrawPlots failed.")
            retval = false;
        }

        // Delete the plots. We don't have a "DeleteAllPlots" so just delete a
        // bunch of times in the case of sessions so we are most likely going to
        // cause all plots to be deleted (after each deletion, plot 0 becomes active)
        for(int p = 0; p < 10; ++p)
            VisItDeleteActivePlots();
    }

    return retval;
}

// --------------------------------------------------------------------------
int
LibsimAnalysisAdaptor::PrivateData::ProcessVisItCommand(int rank)
{
    int command = VISIT_COMMAND_PROCESS;
    if (rank==0)
    {
        int success = VisItProcessEngineCommand();

        if (success == VISIT_OKAY)
        {
            command = VISIT_COMMAND_SUCCESS;
            MPI_Bcast(&command, 1, MPI_INT, 0, this->Comm);
            return 1;
        }
        else
        {
            command = VISIT_COMMAND_FAILURE;
            MPI_Bcast(&command, 1, MPI_INT, 0, this->Comm);
            return 0;
        }
    }
    else
    {
        /* Note: only through the SlaveProcessCallback callback
         * above can the rank 0 process send a VISIT_COMMAND_PROCESS
         * instruction to the non-rank 0 processes. */
        while (1)
        {
            MPI_Bcast(&command, 1, MPI_INT, 0, this->Comm);
            switch (command)
            {
            case VISIT_COMMAND_PROCESS:
                VisItProcessEngineCommand();
                break;
            case VISIT_COMMAND_SUCCESS:
                return 1;
            case VISIT_COMMAND_FAILURE:
                return 0;
            }
        }
    }
}

// --------------------------------------------------------------------------
bool
LibsimAnalysisAdaptor::PrivateData::Execute_Interactive(int rank)
{
    int visitstate = 0, blocking = 0, err = 0;

    // If we are paused, block. We can do this even if we're not connected
    // if we gave "interactive,paused" as the mode. This means that we want
    // to start paused so we can connect.
    if(this->paused)
        blocking = 1;

    if(VisItIsConnected())
    {
        // If we've connected, we might have plots to update.
        VisItUpdatePlots();
    }

    do
    {
        // Get input from VisIt
        if(rank == 0)
        {
            visitstate = VisItDetectInputWithTimeout(blocking, 200, -1);
        }
        // Broadcast the return value of VisItDetectInput to all procs.
        MPI_Bcast(&visitstate, 1, MPI_INT, 0, this->Comm);

        // Do different things depending on the output from VisItDetectInput.
        switch(visitstate)
        {
        case 0:
            // There was no input from VisIt, try again.
            break;
        case 1:
            // VisIt is trying to connect to sim.
            if(VisItAttemptToCompleteConnection() == VISIT_OKAY)
            {
                // Register Libsim callbacks.
                VisItSetCommandCallback(ControlCommandCallback, (void*)this);
                VisItSetSlaveProcessCallback2(SlaveProcessCallback, (void*)this);
                VisItSetGetMetaData(GetMetaData, (void*)this);
                VisItSetGetMesh(GetMesh, (void*)this);
                VisItSetGetVariable(GetVariable, (void*)this);
                VisItSetGetDomainList(GetDomainList, (void*)this);
                if (this->ComputeNesting)
                    VisItSetGetDomainNesting(GetDomainNesting, (void*)this);

                // Pause when we connect.
                this->paused = true;
            }
            else
            {
                // Print the error message
                if(rank == 0)
                {
                    char *err = VisItGetLastError();
                    fprintf(stderr, "VisIt did not connect: %s\n", err);
                    free(err);
                }
            }
            break;
        case 2:
            // VisIt wants to tell the engine something.
            if(!ProcessVisItCommand(rank))
            {
                // Disconnect on an error or closed connection.
                VisItDisconnect();
                // Start running again if VisIt closes.
                this->paused = false;
            }
            break;
        case 3:
            // No console input.
            break;
        default:
            //fprintf(stderr, "Can't recover from error %d!\n", visitstate);
            //err = 1;
            break;
        }
    } while(this->paused && err == 0);

    return true;
}

// --------------------------------------------------------------------------
int
LibsimAnalysisAdaptor::PrivateData::TopologicalDimension(const int dims[3]) const
{
    int d = 0;
    if(dims[0] > 1) ++d;
    if(dims[1] > 1) ++d;
    if(dims[2] > 1) ++d;
    return d;
}

// --------------------------------------------------------------------------
int
LibsimAnalysisAdaptor::PrivateData::DecodeVarName(const std::string &varName,
    std::string &meshName, std::string &arrayName, int &association)
{
    size_t slash1 = varName.find('/', 0);
    if (slash1 == std::string::npos)
    {
        SENSEI_ERROR("Invalid variable name \"" << varName << "\""
            << " Expected the format <mesh>/<centering>/<array>")
        return -1;
    }


    size_t slash2 = varName.find('/', slash1+1);
    if (slash2 == std::string::npos)
    {
        SENSEI_ERROR("Invalid variable name \"" << varName << "\""
            << " Expected the format <mesh>/<centering>/<array>")
        return -1;
    }

    meshName = varName.substr(0, slash1);

    std::string centering = varName.substr(slash1+1, slash2 - slash1 - 1);
    if (centering == "point")
    {
        association = vtkDataObject::POINT;
    }
    else if (centering == "cell")
    {
        association = vtkDataObject::CELL;
    }
    else
    {
        SENSEI_ERROR("Invalid centering " << centering
          << " for variable " << varName)
        return -1;
    }

    arrayName = varName.substr(slash2+1, std::string::npos);

    return 0;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// VTK to Libsim helper functions
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

static visit_handle
vtkDataArray_To_VisIt_VariableData(vtkDataArray *arr)
{
    visit_handle h = VISIT_INVALID_HANDLE;
    if(arr != nullptr)
    {
        // If we have a standard memory layout in a supported type,
        // zero-copy expose the data to libsim.
        if(VisIt_VariableData_alloc(&h) != VISIT_ERROR)
        {
            bool copy = false;
            int nc = arr->GetNumberOfComponents();
            int nt = arr->GetNumberOfTuples();
            if(arr->HasStandardMemoryLayout())
            {
                if(arr->GetDataType() == VTK_CHAR || arr->GetDataType() == VTK_UNSIGNED_CHAR)
                    VisIt_VariableData_setDataC(h, VISIT_OWNER_SIM, nc, nt, (char *)arr->GetVoidPointer(0));
                else if(arr->GetDataType() == VTK_INT)
                    VisIt_VariableData_setDataI(h, VISIT_OWNER_SIM, nc, nt, (int *)arr->GetVoidPointer(0));
                else if(arr->GetDataType() == VTK_LONG)
                    VisIt_VariableData_setDataL(h, VISIT_OWNER_SIM, nc, nt, (long *)arr->GetVoidPointer(0));
                else if(arr->GetDataType() == VTK_FLOAT)
                    VisIt_VariableData_setDataF(h, VISIT_OWNER_SIM, nc, nt, (float *)arr->GetVoidPointer(0));
                else if(arr->GetDataType() == VTK_DOUBLE)
                    VisIt_VariableData_setDataD(h, VISIT_OWNER_SIM, nc, nt, (double *)arr->GetVoidPointer(0));
                else
                    copy = true;

#ifdef VISIT_DEBUG_LOG
                if(!copy)
                {
                    VisItDebug5("SENSEI: Standard memory layout: nc=%d, nt=%d\n", nc, nt);
                }
#endif
            }
            else
            {
                // NOTE: we could detect some non-contiguous memory layouts here and
                //       expose to Libsim that way. Just copy for now...
                copy = true;
            }

            // Expose the data as a copy, converting to double.
            if(copy)
            {
#ifdef VISIT_DEBUG_LOG
                VisItDebug5("SENSEI: Copying required: nc=%d, nt=%d\n", nc, nt);
#endif

                double *v = (double *)malloc(sizeof(double) * nc * nt);
                double *tuple = v;
                for(int i = 0; i < nt; ++i)
                {
                    arr->GetTuple(i, tuple);
                    tuple += nc;
                }
                VisIt_VariableData_setDataD(h, VISIT_OWNER_VISIT, nc, nt, v);
            }
        }
    }

    return h;
}

// -----------------------------------------------------------------------------
static int vtk_to_libsim[VTK_NUMBER_OF_CELL_TYPES];
static bool vtk_to_libsim_init = false;
static int
celltype_vtk_to_libsim(unsigned char vtkcelltype)
{
    if(!vtk_to_libsim_init)
    {
        for(int i =0; i < VTK_NUMBER_OF_CELL_TYPES; ++i)
            vtk_to_libsim[i] = -1;

        vtk_to_libsim[VTK_LINE] = VISIT_CELL_BEAM;
        vtk_to_libsim[VTK_TRIANGLE] =  VISIT_CELL_TRI;
        vtk_to_libsim[VTK_QUAD] =  VISIT_CELL_QUAD;
        vtk_to_libsim[VTK_TETRA] =  VISIT_CELL_TET;
        vtk_to_libsim[VTK_PYRAMID] =  VISIT_CELL_PYR;
        vtk_to_libsim[VTK_WEDGE] =  VISIT_CELL_WEDGE;
        vtk_to_libsim[VTK_HEXAHEDRON] =  VISIT_CELL_HEX;
        vtk_to_libsim[VTK_VERTEX] =  VISIT_CELL_POINT;

        vtk_to_libsim[VTK_QUADRATIC_EDGE] =  VISIT_CELL_QUADRATIC_EDGE;
        vtk_to_libsim[VTK_QUADRATIC_TRIANGLE] =  VISIT_CELL_QUADRATIC_TRI;
        vtk_to_libsim[VTK_QUADRATIC_QUAD] =  VISIT_CELL_QUADRATIC_QUAD;
        vtk_to_libsim[VTK_QUADRATIC_TETRA] =  VISIT_CELL_QUADRATIC_TET;
        vtk_to_libsim[VTK_QUADRATIC_PYRAMID] =  VISIT_CELL_QUADRATIC_PYR;
        vtk_to_libsim[VTK_QUADRATIC_WEDGE] =  VISIT_CELL_QUADRATIC_WEDGE;
        vtk_to_libsim[VTK_QUADRATIC_HEXAHEDRON] =  VISIT_CELL_QUADRATIC_HEX;

        vtk_to_libsim[VTK_BIQUADRATIC_TRIANGLE] =  VISIT_CELL_BIQUADRATIC_TRI;
        vtk_to_libsim[VTK_BIQUADRATIC_QUAD] =  VISIT_CELL_BIQUADRATIC_QUAD;
        vtk_to_libsim[VTK_TRIQUADRATIC_HEXAHEDRON] =  VISIT_CELL_TRIQUADRATIC_HEX;
        vtk_to_libsim[VTK_QUADRATIC_LINEAR_QUAD] =  VISIT_CELL_QUADRATIC_LINEAR_QUAD;
        vtk_to_libsim[VTK_QUADRATIC_LINEAR_WEDGE] =  VISIT_CELL_QUADRATIC_LINEAR_WEDGE;
        vtk_to_libsim[VTK_BIQUADRATIC_QUADRATIC_WEDGE] =  VISIT_CELL_BIQUADRATIC_QUADRATIC_WEDGE;
        vtk_to_libsim[VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON] =  VISIT_CELL_BIQUADRATIC_QUADRATIC_HEX;

        vtk_to_libsim_init = true;
    }

    return vtk_to_libsim[vtkcelltype];
}

// -----------------------------------------------------------------------------
static visit_handle
vtkDataSet_GhostData(vtkDataSetAttributes *dsa, const std::string &name)
{
    visit_handle h = VISIT_INVALID_HANDLE;
    // Check that we have the array and it is of allowed types.
    vtkDataArray *arr = dsa->GetArray(name.c_str());
    if(arr &&
       arr->GetNumberOfComponents() == 1 &&
       arr->GetNumberOfTuples() > 0 &&
       (vtkUnsignedCharArray::SafeDownCast(arr) ||
        vtkCharArray::SafeDownCast(arr) ||
        vtkIntArray::SafeDownCast(arr))
      )
    {
        h = vtkDataArray_To_VisIt_VariableData(arr);
    }
    return h;
}

// -----------------------------------------------------------------------------
static visit_handle
vtkDataSet_to_VisIt_Mesh(vtkDataObject *dobj)
{
    vtkDataSet *ds = dynamic_cast<vtkDataSet*>(dobj);
    if (dobj && !ds)
    {
        SENSEI_ERROR("Can't convert a "
           << (dobj ? dobj->GetClassName() : "nullptr"))
        return VISIT_INVALID_HANDLE;
    }

    visit_handle mesh = VISIT_INVALID_HANDLE;
    vtkImageData *igrid = vtkImageData::SafeDownCast(ds);
    vtkRectilinearGrid *rgrid = vtkRectilinearGrid::SafeDownCast(ds);
    vtkStructuredGrid  *sgrid = vtkStructuredGrid::SafeDownCast(ds);
    vtkPolyData *pgrid = vtkPolyData::SafeDownCast(ds);
    vtkUnstructuredGrid *ugrid = vtkUnstructuredGrid::SafeDownCast(ds);
    if(igrid != nullptr)
    {
#ifdef VISIT_DEBUG_LOG
        VisItDebug5("SENSEI: \tExposing vtkImageData as a rectilinear grid.\n");
#endif
        // We already have a VTK dataset. Libsim doesn't have a path to just
        // pass it through to SimV2+VisIt so we have to pull some details
        // out to make the right Libsim calls so the SimV2 reader will be
        // able to make the right VTK dataset on the other end. Silly/Stupid
        // but giving VTK datasets to Libsim has never come up before.

        double x0[3] = {0.0};
        double dx[3] = {0.0};
        int dims[3] = {0};
        int ext[6] = {0};
        igrid->GetDimensions(dims);
        igrid->GetExtent(ext);
        igrid->GetOrigin(x0);
        igrid->GetSpacing(dx);

        if(VisIt_RectilinearMesh_alloc(&mesh) == VISIT_OKAY)
        {
            int nx = std::max(dims[0], 1);
            int ny = std::max(dims[1], 1);
            int nz = std::max(dims[2], 1);
            float *x = (float *)malloc(sizeof(float) * nx);
            float *y = (float *)malloc(sizeof(float) * ny);
            float *z = (float *)malloc(sizeof(float) * nz);
            if(x != nullptr && y != nullptr && z != nullptr)
            {
                visit_handle xc = VISIT_INVALID_HANDLE,
                             yc = VISIT_INVALID_HANDLE,
                             zc = VISIT_INVALID_HANDLE;
                if(VisIt_VariableData_alloc(&xc) == VISIT_OKAY &&
                   VisIt_VariableData_alloc(&yc) == VISIT_OKAY &&
                   VisIt_VariableData_alloc(&zc) == VISIT_OKAY)
                {
                    for(int i = 0; i < nx; ++i)
                        x[i] = x0[0] + (ext[0] + i)*dx[0];
                    for(int i = 0; i < ny; ++i)
                        y[i] = x0[1] + (ext[2] + i)*dx[1];
                    VisIt_VariableData_setDataF(xc, VISIT_OWNER_VISIT, 1, nx, x);
                    VisIt_VariableData_setDataF(yc, VISIT_OWNER_VISIT, 1, ny, y);
                    if(nz > 1)
                    {
                        for(int i = 0; i < nz; ++i)
                            z[i] = x0[2] + (ext[4] + i)*dx[2];
                        VisIt_VariableData_setDataF(zc, VISIT_OWNER_VISIT, 1, nz, z);
                        VisIt_RectilinearMesh_setCoordsXYZ(mesh, xc, yc, zc);
                    }
                    else
                    {
                        VisIt_VariableData_free(zc); // didn't use it.
                        VisIt_RectilinearMesh_setCoordsXY(mesh, xc, yc);
                    }

                    // Try and make some ghost nodes.
                    visit_handle gn = vtkDataSet_GhostData(ds->GetPointData(),
                                          "vtkGhostType");
                    if(gn != VISIT_INVALID_HANDLE)
                        VisIt_RectilinearMesh_setGhostNodes(mesh, gn);
                    // Try and make some ghost cells.
                    visit_handle gz = vtkDataSet_GhostData(ds->GetCellData(),
                                          "vtkGhostType");
                    if(gz != VISIT_INVALID_HANDLE)
                        VisIt_RectilinearMesh_setGhostCells(mesh, gz);
                }
                else
                {
                    VisIt_RectilinearMesh_free(mesh);
                    mesh = VISIT_INVALID_HANDLE;
                    if(xc != VISIT_INVALID_HANDLE)
                        VisIt_VariableData_free(xc);
                    if(yc != VISIT_INVALID_HANDLE)
                        VisIt_VariableData_free(yc);
                    if(zc != VISIT_INVALID_HANDLE)
                        VisIt_VariableData_free(zc);
                    if(x != nullptr) free(x);
                    if(y != nullptr) free(y);
                    if(z != nullptr) free(z);
                }
            }
            else
            {
                VisIt_RectilinearMesh_free(mesh);
                mesh = VISIT_INVALID_HANDLE;
                if(x != nullptr) free(x);
                if(y != nullptr) free(y);
                if(z != nullptr) free(z);
            }
        }
    }
    else if(rgrid != nullptr)
    {
        if(VisIt_RectilinearMesh_alloc(&mesh) != VISIT_ERROR)
        {
            visit_handle hx, hy, hz;
            hx = vtkDataArray_To_VisIt_VariableData(rgrid->GetXCoordinates());
            hy = vtkDataArray_To_VisIt_VariableData(rgrid->GetYCoordinates());
            if(hx != VISIT_INVALID_HANDLE && hy != VISIT_INVALID_HANDLE)
            {
                hz = vtkDataArray_To_VisIt_VariableData(rgrid->GetZCoordinates());
                if(hz != VISIT_INVALID_HANDLE)
                    VisIt_RectilinearMesh_setCoordsXYZ(mesh, hx, hy, hz);
                else
                    VisIt_RectilinearMesh_setCoordsXY(mesh, hx, hy);

                // Try and make some ghost nodes.
                visit_handle gn = vtkDataSet_GhostData(ds->GetPointData(),
                                      "vtkGhostType");
                if(gn != VISIT_INVALID_HANDLE)
                    VisIt_RectilinearMesh_setGhostNodes(mesh, gn);
                // Try and make some ghost cells.
                visit_handle gz = vtkDataSet_GhostData(ds->GetCellData(),
                                      "vtkGhostType");
                if(gz != VISIT_INVALID_HANDLE)
                    VisIt_RectilinearMesh_setGhostCells(mesh, gz);
            }
            else
            {
                if(hx != VISIT_INVALID_HANDLE)
                    VisIt_VariableData_free(hx);
                if(hy != VISIT_INVALID_HANDLE)
                    VisIt_VariableData_free(hy);
                VisIt_RectilinearMesh_free(mesh);
                mesh = VISIT_INVALID_HANDLE;
            }
        }
    }
    else if(sgrid != nullptr)
    {
        if(VisIt_CurvilinearMesh_alloc(&mesh) != VISIT_ERROR)
        {
            int dims[3];
            sgrid->GetDimensions(dims);
            visit_handle pts = vtkDataArray_To_VisIt_VariableData(sgrid->GetPoints()->GetData());
            if(pts != VISIT_INVALID_HANDLE)
            {
                VisIt_CurvilinearMesh_setCoords3(mesh, dims, pts);

                // Try and make some ghost nodes.
                visit_handle gn = vtkDataSet_GhostData(ds->GetPointData(),
                                      "vtkGhostType");
                if(gn != VISIT_INVALID_HANDLE)
                    VisIt_CurvilinearMesh_setGhostNodes(mesh, gn);
                // Try and make some ghost cells.
                visit_handle gz = vtkDataSet_GhostData(ds->GetCellData(),
                                      "vtkGhostType");
                if(gz != VISIT_INVALID_HANDLE)
                    VisIt_CurvilinearMesh_setGhostCells(mesh, gz);
            }
            else
            {
                VisIt_CurvilinearMesh_free(mesh);
                mesh = VISIT_INVALID_HANDLE;
            }
        }
    }
    else if(pgrid && pgrid->GetVerts())
    {
        if(VisIt_PointMesh_alloc(&mesh) != VISIT_ERROR)
        {
            bool perr = true;
            vtkPoints *p = pgrid->GetPoints();
            if(p != nullptr)
            {
                visit_handle pts = vtkDataArray_To_VisIt_VariableData(p->GetData());
                if(pts != VISIT_INVALID_HANDLE)
                {
                    VisIt_PointMesh_setCoords(mesh, pts);
                    perr = false;
                }
            }

            if(perr)
            {
                SENSEI_ERROR("The vtkPolyData's coordinates are not set.")
                VisIt_PointMesh_free(mesh);
                mesh = VISIT_INVALID_HANDLE;
            }
        }
    }
    else if(ugrid != nullptr)
    {
#ifdef VISIT_DEBUG_LOG
        VisItDebug5("SENSEI: vtkUnstructuredGrid: npts = %d, ncells = %d\n",
            (int)ugrid->GetNumberOfPoints(), (int)ugrid->GetNumberOfCells());
#endif
        if(VisIt_UnstructuredMesh_alloc(&mesh) != VISIT_ERROR)
        {
            bool err = false;
            visit_handle pts = vtkDataArray_To_VisIt_VariableData(ugrid->GetPoints()->GetData());
            if(pts != VISIT_INVALID_HANDLE)
                VisIt_UnstructuredMesh_setCoords(mesh, pts);
            else
                err = true;

            // Libsim and VTK connectivity is a little different. Why'd we do that?
            vtkIdType ncells = ugrid->GetNumberOfCells();
            if(ncells > 0 && !err)
            {
                const unsigned char *cellTypes = (const unsigned char *)ugrid->GetCellTypesArray()->GetVoidPointer(0);
                const vtkIdType *vtkconn = (const vtkIdType *)ugrid->GetCells()->GetData()->GetVoidPointer(0);
                const vtkIdType *offsets = (const vtkIdType *)ugrid->GetCellLocationsArray()->GetVoidPointer(0);
                int connlen = ugrid->GetCells()->GetNumberOfConnectivityEntries();
                int *newconn = new int[connlen];
                int *lsconn = newconn;
                for(int cellid = 0; cellid < ncells; ++cellid)
                {
                    // Map VTK cell type to Libsim cell type.
                    int lsct = celltype_vtk_to_libsim(cellTypes[cellid]);
                    if(lsct != -1)
                    {
                        *lsconn++ = lsct;

                        // The number of points is the first number for the cell.
                        const vtkIdType *cellConn = vtkconn + offsets[cellid];
                        vtkIdType npts = cellConn[0];
                        cellConn++;
                        for(vtkIdType idx = 0; idx < npts; ++idx)
                            *lsconn++ = static_cast<int>(cellConn[idx]);
                    }
                    else
                    {
                        // We got a cell type we don't support. Make a vertex cell
                        // so we at least don't mess up the cell data later.
                        *lsconn++ = VISIT_CELL_POINT;
                        const vtkIdType *cellConn = vtkconn + offsets[cellid];
                        *lsconn++ = cellConn[1];
                    }
                }

                visit_handle hc = VISIT_INVALID_HANDLE;
                if(VisIt_VariableData_alloc(&hc) != VISIT_ERROR)
                {
                    // Wrap newconn, let VisIt own it.
                    VisIt_VariableData_setDataI(hc, VISIT_OWNER_VISIT, 1, connlen, newconn);
                    VisIt_UnstructuredMesh_setConnectivity(mesh, ncells, hc);

                    // Try and make some ghost nodes.
                    visit_handle gn = vtkDataSet_GhostData(ds->GetPointData(),
                                          "vtkGhostType");
                    if(gn != VISIT_INVALID_HANDLE)
                        VisIt_RectilinearMesh_setGhostNodes(mesh, gn);
                    // Try and make some ghost cells.
                    visit_handle gz = vtkDataSet_GhostData(ds->GetCellData(),
                                          "vtkGhostType");
                    if(gz != VISIT_INVALID_HANDLE)
                        VisIt_UnstructuredMesh_setGhostCells(mesh, gz);
                }
                else
                {
                    delete [] newconn;
                    err = true;
                }
            }

            if(err)
            {
                VisIt_UnstructuredMesh_free(mesh);
                mesh = VISIT_INVALID_HANDLE;
            }
        }
    }
    // TODO: expand to other mesh types.
    else
    {
        SENSEI_ERROR("Unsupported VTK mesh type "
          << (ds ? ds->GetClassName() : dobj ? dobj->GetClassName() : "nullptr"))
#ifdef VISIT_DEBUG_LOG
        VisItDebug5("SENSEI: Unsupported VTK mesh type.\n");
#endif
    }

    return mesh;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// CALLBACK FUNCTIONS FOR LIBSIM
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// --------------------------------------------------------------------------

int
LibsimAnalysisAdaptor::PrivateData::broadcast_int(int *value, int sender, void *cbdata)
{
    PrivateData *This = (PrivateData *)cbdata;
    return MPI_Bcast(value, 1, MPI_INT, sender, This->Comm);
}

// --------------------------------------------------------------------------
int
LibsimAnalysisAdaptor::PrivateData::broadcast_string(char *str, int len, int sender, void *cbdata)
{
    PrivateData *This = (PrivateData *)cbdata;
    return MPI_Bcast(str, len, MPI_CHAR, sender, This->Comm);
}

// --------------------------------------------------------------------------
void
LibsimAnalysisAdaptor::PrivateData::SlaveProcessCallback(void *cbdata)
{
    int value = 0;
    broadcast_int(&value, 0, cbdata);
}

void
LibsimAnalysisAdaptor::PrivateData::ControlCommandCallback(
    const char *cmd, const char *args, void *cbdata)
{
    (void)args;
    PrivateData *This = (PrivateData *)cbdata;

    if(strcmp(cmd, "pause") == 0)
        This->paused = true;
    else if(strcmp(cmd, "run") == 0)
        This->paused = false;
}

// --------------------------------------------------------------------------
visit_handle
LibsimAnalysisAdaptor::PrivateData::GetMetaData(void *cbdata)
{
#ifdef VISIT_DEBUG_LOG
    VisItDebug5("SENSEI: LibsimAnalysisAdaptor::PrivateData::GetMetaData\n");
#endif
    PrivateData *This = (PrivateData *)cbdata;
    sensei::DataAdaptor *Adaptor = This->Adaptor;

    // allocate sturcture to tell VisIt about the simulation data
    visit_handle md = VISIT_INVALID_HANDLE;
    if (VisIt_SimulationMetaData_alloc(&md) != VISIT_OKAY)
    {
        SENSEI_ERROR("Failed to allocate simulation metadata")
        return VISIT_INVALID_HANDLE;
    }

    // Set the simulation state.
    VisIt_SimulationMetaData_setMode(md,
      This->paused ? VISIT_SIMMODE_STOPPED : VISIT_SIMMODE_RUNNING);

    VisIt_SimulationMetaData_setCycleTime(md,
      Adaptor->GetDataTimeStep(), Adaptor->GetDataTime());

    // for each mesh we'll pass metadata onto VisIt
    unsigned int nMeshes = 0;
    if (Adaptor->GetNumberOfMeshes(nMeshes))
    {
        SENSEI_ERROR("Failed to get the number of meshes")
        return VISIT_INVALID_HANDLE;
    }

    // set up the metadata cache
    This->Metadata.clear();

    for (unsigned int i = 0; i < nMeshes; ++i)
    {
        sensei::MeshMetadataPtr mmd = sensei::MeshMetadata::New();

        // enable optional metadata
        mmd->Flags.SetBlockDecomp();
        mmd->Flags.SetBlockExtents();
        mmd->Flags.SetBlockBounds();

        if (Adaptor->GetMeshMetadata(i, mmd))
        {
            SENSEI_ERROR("Failed to get metadata for mesh " << i)
            return VISIT_INVALID_HANDLE;
        }

        // check if the sim gave us what we asked for
        MeshMetadataFlags reqFlags = mmd->Flags;
        reqFlags.ClearBlockBounds();

        if (mmd->Validate(This->Comm, reqFlags))
        {
            SENSEI_ERROR("Invalid metadata for mesh " << i)
            return VISIT_INVALID_HANDLE;
        }

        // this simplifies things substantially to be able to have a global view
        // the driver behind this is AMR data, for which we require a global view.
        if (!mmd->GlobalView)
            mmd->GlobalizeView(This->Comm);

        // cache the metadata
        This->Metadata[mmd->MeshName] = mmd;

        // pass to VisIt
        // Add mesh metadata.
        visit_handle vmmd = VISIT_INVALID_HANDLE;
        if (VisIt_MeshMetaData_alloc(&vmmd) != VISIT_OKAY)
        {
            SENSEI_ERROR("Failed to allocate mesh metadata")
            return VISIT_INVALID_HANDLE;
        }

        if (mmd->MeshType == VTK_OVERLAPPING_AMR)
        {
            int dims[3] = {mmd->Extent[1] - mmd->Extent[0] + 1,
                mmd->Extent[3] - mmd->Extent[2] + 1,
                mmd->Extent[5] - mmd->Extent[4] + 1};

            int topoDims = This->TopologicalDimension(dims);

            VisIt_MeshMetaData_setMeshType(vmmd, VISIT_MESHTYPE_AMR);
            VisIt_MeshMetaData_setTopologicalDimension(vmmd, topoDims);
            VisIt_MeshMetaData_setSpatialDimension(vmmd, topoDims);

            VisIt_MeshMetaData_setDomainTitle(vmmd, "Patches");
            VisIt_MeshMetaData_setDomainPieceName(vmmd, "patch");

            VisIt_MeshMetaData_setNumGroups(vmmd, mmd->NumLevels);
            VisIt_MeshMetaData_setGroupTitle(vmmd, "Levels");
            VisIt_MeshMetaData_setGroupPieceName(vmmd, "level");

            for (int j = 0; j < mmd->NumBlocks; ++j)
            {
                int b = 0;
#ifdef USE_REAL_DOMAIN
                for (; b < mmd->NumBlocks; ++b)
                    if (mmd->BlockIds[b] == j)
                        break;
#else
                b = j;
#endif
                VisIt_MeshMetaData_addGroupId(vmmd, mmd->BlockLevel[b]);
            }
        }
        else
        {
            switch (mmd->BlockType)
            {
                case VTK_STRUCTURED_GRID:
                case VTK_RECTILINEAR_GRID:
                case VTK_IMAGE_DATA:
                {
                    int meshType = mmd->BlockType == VTK_STRUCTURED_GRID ?
                        VISIT_MESHTYPE_CURVILINEAR : VISIT_MESHTYPE_RECTILINEAR;

                    int dims[3] = {mmd->Extent[1] - mmd->Extent[0] + 1,
                        mmd->Extent[3] - mmd->Extent[2] + 1,
                        mmd->Extent[5] - mmd->Extent[4] + 1};

                    int topoDims = This->TopologicalDimension(dims);

                    VisIt_MeshMetaData_setMeshType(vmmd, meshType);
                    VisIt_MeshMetaData_setTopologicalDimension(vmmd, topoDims);
                    VisIt_MeshMetaData_setSpatialDimension(vmmd, topoDims);
                }
                break;

                case VTK_UNSTRUCTURED_GRID:
                case VTK_POLY_DATA:
                {
                    VisIt_MeshMetaData_setMeshType(vmmd, VISIT_MESHTYPE_UNSTRUCTURED);
                    VisIt_MeshMetaData_setTopologicalDimension(vmmd, 3);
                    VisIt_MeshMetaData_setSpatialDimension(vmmd, 3);
                }
                break;

                default:
                {
                    SENSEI_ERROR("Unsupported block type " << mmd->BlockType)
                    return VISIT_INVALID_HANDLE;
                }
            }
        }

        VisIt_MeshMetaData_setName(vmmd, mmd->MeshName.c_str());
        VisIt_MeshMetaData_setNumDomains(vmmd, mmd->NumBlocks);
        VisIt_SimulationMetaData_addMesh(md, vmmd);

        // Add variables
        for (int j = 0; j < mmd->NumArrays; ++j)
        {
            visit_handle vmd = VISIT_INVALID_HANDLE;
            if (VisIt_VariableMetaData_alloc(&vmd) != VISIT_OKAY)
            {
                SENSEI_ERROR("Failed to allocate variable metadata")
                return VISIT_INVALID_HANDLE;
            }

            // naming convention: <mesh>/<centering>/<var>
            std::string arrayName = mmd->MeshName + "/" +
                (mmd->ArrayCentering[j] == vtkDataObject::POINT ? "point" : "cell") +
                "/" + mmd->ArrayName[j];

            VisIt_VariableMetaData_setName(vmd, arrayName.c_str());

            VisIt_VariableMetaData_setMeshName(vmd, mmd->MeshName.c_str());

            int varType = -1;
            switch (mmd->ArrayComponents[j])
            {
                case 1:
                    varType = VISIT_VARTYPE_SCALAR;
                    break;
                case 3:
                    varType = VISIT_VARTYPE_VECTOR;
                    break;
                case 6:
                    varType = VISIT_VARTYPE_SYMMETRIC_TENSOR;
                    break;
                case 9:
                    varType = VISIT_VARTYPE_TENSOR;
                    break;
                default:
                    SENSEI_ERROR("Failed to proccess an array with "
                      << mmd->ArrayComponents[j] << " compnents")
                    return VISIT_INVALID_HANDLE;
            }
            VisIt_VariableMetaData_setType(vmd, varType);

            int varCen = mmd->ArrayCentering[j] == vtkDataObject::POINT ?
                VISIT_VARCENTERING_NODE : VISIT_VARCENTERING_ZONE;

            VisIt_VariableMetaData_setCentering(vmd, varCen);

            VisIt_SimulationMetaData_addVariable(md, vmd);
        }
    }

    // Add some commands.
    static const char *cmd_names[] = {"pause", "run"};
    for(int i = 0; i < 2; ++i)
    {
        visit_handle cmd = VISIT_INVALID_HANDLE;
        if(VisIt_CommandMetaData_alloc(&cmd) == VISIT_OKAY)
        {
            VisIt_CommandMetaData_setName(cmd, cmd_names[i]);
            VisIt_SimulationMetaData_addGenericCommand(md, cmd);
        }
    }

    return md;
}

// --------------------------------------------------------------------------
void LibsimAnalysisAdaptor::PrivateData::ClearCache()
{
#ifdef VISIT_DEBUG_LOG
    VisItDebug5("SENSEI: LibsimAnalysisAdaptor::PrivateData::ClearCache\n");
#endif
    this->Meshes.clear();
    this->Metadata.clear();
}

// --------------------------------------------------------------------------
int LibsimAnalysisAdaptor::PrivateData::GetMesh(const std::string &meshName,
    vtkDataObjectPtr &dobjp)
{
    dobjp = nullptr;

    // get the mesh. it's cached because visit wants things block
    // by block but sensei only works with the whole object
    auto it = this->Meshes.find(meshName);
    if (it  == this->Meshes.end())
    {
        // mesh was not in the cache add it now
        vtkDataObject *dobj = nullptr;
        if (this->Adaptor->GetMesh(meshName, false, dobj))
        {
            SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
            return -1;
        }

        // get the metadata, it should already be available
        auto mdit = this->Metadata.find(meshName);
        if (mdit == this->Metadata.end())
        {
            SENSEI_ERROR("No metadata for mesh \"" << meshName << "\"")
            return -1;
        }
        MeshMetadataPtr mmd = mdit->second;

        // add ghost zones. if the simulation has them we always want/need
        // them
        if (mmd->NumGhostCells && this->Adaptor->AddGhostCellsArray(dobj, meshName))
        {
            SENSEI_ERROR("Failed to add ghost cells to mesh \""
              << meshName << "\"")
            return -1;
        }

        if (mmd->NumGhostNodes && this->Adaptor->AddGhostNodesArray(dobj, meshName))
        {
            SENSEI_ERROR("Failed to add ghost nodes to mesh \""
              << meshName << "\"")
            return -1;
        }

        dobjp.TakeReference(dobj);
        this->Meshes[meshName] = dobjp;
    }
    else
    {
        dobjp = it->second;
    }

    return 0;
}

// --------------------------------------------------------------------------
int LibsimAnalysisAdaptor::PrivateData::GetVariable(int dom,
    const std::string &varName, vtkDataArray *&array)
{
#ifdef VISIT_DEBUG_LOG
    VisItDebug5("SENSEI: LibsimAnalysisAdaptor::PrivateData::GetVariable\n");
#endif

    array = nullptr;

    // convert from visit variable name into sensei's mesh, centering, and
    // array name
    std::string meshName;
    std::string arrayName;
    int association = -1;
    if (this->DecodeVarName(varName, meshName, arrayName, association))
        return -1;

    // get the mesh
    vtkDataObjectPtr dobj;
    if (this->GetMesh(meshName, dobj))
        return -1;

    vtkCompositeDataSetPtr cd =
      VTKUtils::AsCompositeData(this->Comm, dobj.GetPointer(), false);

    // see if we already have this array
    vtkCompositeDataIterator *cdit = cd->NewIterator();

    // this rank has no local data
    if (cdit->IsDoneWithTraversal())
    {
        cdit->Delete();
        return 0;
    }

    // read the array if we have not yet
    if (!cdit->GetCurrentDataObject()->GetAttributes(association)->GetArray(arrayName.c_str()))
    {
        if (this->Adaptor->AddArray(dobj.GetPointer(), meshName, association, arrayName))
        {
            SENSEI_ERROR("Failed to add " << VTKUtils::GetAttributesName(association)
              << " data array \"" << arrayName << "\"")
            cdit->Delete();
            return -1;
        }
    }

    // extract array from the requested block

    // VTK's iterators for AMR datasets behave differently than for multiblock
    // datasets.  we are going to have to handle AMR data as a special case for
    // now.

    vtkUniformGridAMRDataIterator *amrIt = dynamic_cast<vtkUniformGridAMRDataIterator*>(cdit);
    vtkOverlappingAMR *amrMesh = dynamic_cast<vtkOverlappingAMR*>(cd.Get());

    for (cdit->InitTraversal(); !cdit->IsDoneWithTraversal(); cdit->GoToNextItem())
    {
        long blockId = 0;
        if (amrIt)
        {
            // special case for AMR
            int level = amrIt->GetCurrentLevel();
            int index = amrIt->GetCurrentIndex();
            blockId = amrMesh->GetAMRBlockSourceIndex(level, index);
        }
        else
        {
            // other composite data
            blockId = cdit->GetCurrentFlatIndex() - 1;
        }

        if (blockId == dom)
        {
            array = cdit->GetCurrentDataObject()->GetAttributes(
                association)->GetArray(arrayName.c_str());
            break;
        }
    }

    cdit->Delete();

    if (!array)
    {
        SENSEI_ERROR("Failed to get array \"" << arrayName << "\" for domain "
            << dom << " of mesh \"" << meshName << "\"")
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int LibsimAnalysisAdaptor::PrivateData::GetMesh(int dom,
    const std::string &meshName, vtkDataObject *&mesh)
{
    mesh = nullptr;

    // get the metadata, it should already be available
    auto mdit = this->Metadata.find(meshName);
    if (mdit == this->Metadata.end())
    {
        SENSEI_ERROR("No metadata for mesh \"" << meshName << "\"")
        return -1;
    }
    MeshMetadataPtr mmd = mdit->second;

    // get the mesh
    vtkDataObjectPtr dobj;
    if (this->GetMesh(meshName, dobj))
        return -1;

    vtkCompositeDataSetPtr cd =
      VTKUtils::AsCompositeData(this->Comm, dobj.GetPointer(), false);

    // get the block that visit is after
    vtkCompositeDataIterator *cdit = cd->NewIterator();

    // extract array from the requested block

    // VTK's iterators for AMR datasets behave differently than for multiblock
    // datasets.  we are going to have to handle AMR data as a special case for
    // now.

    vtkUniformGridAMRDataIterator *amrIt = dynamic_cast<vtkUniformGridAMRDataIterator*>(cdit);
    vtkOverlappingAMR *amrMesh = dynamic_cast<vtkOverlappingAMR*>(cd.Get());

    for (cdit->InitTraversal(); !cdit->IsDoneWithTraversal(); cdit->GoToNextItem())
    {
        long blockId = 0;
        if (amrIt)
        {
            // special case for AMR
            int level = amrIt->GetCurrentLevel();
            int index = amrIt->GetCurrentIndex();
            blockId = amrMesh->GetAMRBlockSourceIndex(level, index);
        }
        else
        {
            // other composite data
            blockId = cdit->GetCurrentFlatIndex() - 1;
        }

        if (blockId == dom)
        {
            mesh = cdit->GetCurrentDataObject();
            break;
        }
    }

    cdit->Delete();

    if (!mesh)
    {
        SENSEI_ERROR("Failed to get domain " << dom << " from mesh \""
            << meshName << "\"")
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
visit_handle
LibsimAnalysisAdaptor::PrivateData::GetMesh(int dom, const char *meshName,
  void *cbdata)
{
    VisItDebug5("==== LibsimAnalysisAdaptor::PrivateData::GetMesh ====\n");

    PrivateData *This = (PrivateData *)cbdata;

    vtkDataObject *block = nullptr;

    if (This->GetMesh(dom, std::string(meshName), block))
        return VISIT_INVALID_HANDLE;

    return vtkDataSet_to_VisIt_Mesh(block);
}

// --------------------------------------------------------------------------
visit_handle
LibsimAnalysisAdaptor::PrivateData::GetVariable(int dom, const char *varName,
  void *cbdata)
{
    VisItDebug5("==== LibsimAnalysisAdaptor::PrivateData::GetVariable ====\n");

    PrivateData *This = (PrivateData *)cbdata;

    // get the array for this block
    vtkDataArray *array = nullptr;

    if (This->GetVariable(dom, std::string(varName), array))
        return VISIT_INVALID_HANDLE;

    // this rank has no local blocks
    if (!array)
        return VISIT_INVALID_HANDLE;

    return vtkDataArray_To_VisIt_VariableData(array);
}

// --------------------------------------------------------------------------
visit_handle
LibsimAnalysisAdaptor::PrivateData::GetDomainList(const char *meshName, void *cbdata)
{
    VisItDebug5("==== LibsimAnalysisAdaptor::PrivateData::GetDomainList ====\n");
    TimeEvent<128> mark("LibsimAnalysisAdaptor::GetDomainList");

    PrivateData *This = (PrivateData *)cbdata;

    int rank = 0;
    MPI_Comm_rank(This->Comm, &rank);


    // Create a list of domains owned by this rank.
    MeshMetadataPtr mmd = This->Metadata[meshName];
    std::vector<int> localDomains;
    for (int i = 0; i < mmd->NumBlocks; ++i)
    {
        if (mmd->BlockOwner[i] == rank)
            localDomains.push_back(mmd->BlockIds[i]);
    }

    visit_handle h = VISIT_INVALID_HANDLE;
    if (localDomains.size() > 0)
    {
        VisIt_DomainList_alloc(&h);

        visit_handle hdl;
        VisIt_VariableData_alloc(&hdl);

        VisIt_VariableData_setDataI(hdl, VISIT_OWNER_COPY,
            1, localDomains.size(), localDomains.data());

        VisIt_DomainList_setDomains(h, mmd->NumBlocks, hdl);
    }

    return h;
}

std::array<int,6> refine(const std::array<int,6> &cext, const std::array<int,3> rr)
{
    // refine the coarse patch to put it in the index space of the fine patch
    return std::array<int,6>{{
        cext[0]*rr[0], (cext[1] + 1) * rr[0] - 1,
        cext[2]*rr[1], (cext[3] + 1) * rr[1] - 1,
        cext[4]*rr[2], (cext[5] + 1) * rr[2] - 1}};
}

// check if a course patch,cext, intersects a rr refined fine patch, fext.
// extents are to provided in VTK order: [ilo,ihi, jlo,jhi, klo,khi]
// for 2D patches the 3d dim should have hi=lo and rr=1
bool
intersects(const std::array<int,6> &rcext, const std::array<int,6> &fext)
{
    // check that at least one corner of the fine patch is inside
    // the coarse patch. calculation is made in the fine level index
    // space
    if ((((fext[0] >= rcext[0]) && (fext[0] <= rcext[1])) ||
        ((fext[1] >= rcext[0]) && (fext[1] <= rcext[1]))) &&
        (((fext[2] >= rcext[2]) && (fext[2] <= rcext[3])) ||
        ((fext[3] >= rcext[2]) && (fext[3] <= rcext[3]))) &&
        (((fext[4] >= rcext[4]) && (fext[4] <= rcext[5])) ||
        ((fext[5] >= rcext[4]) && (fext[5] <= rcext[5]))))
        return true;

    return false;
}

// TODO -- this isn't working
// --------------------------------------------------------------------------
visit_handle
LibsimAnalysisAdaptor::PrivateData::GetDomainNesting(const char *name,
    void *cbdata)
{
    VisItDebug5("==== LibsimAnalysisAdaptor::PrivateData::GetDomainNesting ====\n");
    TimeEvent<128> mark("LibsimAnalysisAdaptor::GetDomainNesting");

    visit_handle h = VISIT_INVALID_HANDLE;
    std::string meshName(name);

    PrivateData *This = (PrivateData *)cbdata;
    MeshMetadataPtr mmd = This->Metadata[meshName];

    // skip non-amr datasets
    if (mmd->MeshType != VTK_OVERLAPPING_AMR)
        return VISIT_INVALID_HANDLE;

    // Try and allocate the domain nesting object.
    if(VisIt_DomainNesting_alloc(&h) == VISIT_ERROR)
    {
        VisItDebug1("failed to allocate DomainNesting object.\n");
        return VISIT_INVALID_HANDLE;
    }

    int rank, size;
    MPI_Comm_rank(This->Comm, &rank);
    MPI_Comm_size(This->Comm, &size);

    int dims[3] = {mmd->Extent[1] - mmd->Extent[0] + 1,
        mmd->Extent[3] - mmd->Extent[2] + 1,
        mmd->Extent[5] - mmd->Extent[4] + 1};

    int topoDims = This->TopologicalDimension(dims);

    // Populate the domain nesting structure.
    VisIt_DomainNesting_set_dimensions(h,
        mmd->NumBlocks, mmd->NumLevels, topoDims);

    // Set the refinement ratios.
    for(int i = 0; i < mmd->NumLevels; ++i)
    {
        std::array<int,3> &rr =  mmd->RefRatio[i];
        VisIt_DomainNesting_set_levelRefinement(h, i, rr.data());
    }

    // for each block figure out the list of children
    std::vector<int> nesting;
    nesting.reserve(mmd->NumBlocks);
    for (int i = 0; i < mmd->NumBlocks; ++i)
    {
        int activeLevel = mmd->BlockLevel[i];
        int nextLevel = activeLevel + 1;
        std::array<int,6> &activeExt = mmd->BlockExtents[i];
        std::array<int,3> &rr = mmd->RefRatio[activeLevel];

        // for blocks not on the finest level, find children
        if (nextLevel < mmd->NumLevels)
        {
            // refine the active extent, calculations are made in
            // the finer level index space
            std::array<int,6> rActiveExt = refine(activeExt, rr);

            for (int j = 0; j < mmd->NumBlocks; ++j)
            {
                if (mmd->BlockLevel[j] != nextLevel)
                    continue;

                const std::array<int,6> &otherExt = mmd->BlockExtents[j];

                if (intersects(rActiveExt, otherExt))
                {
#ifdef USE_REAL_DOMAIN
                    nesting.push_back(mmd->BlockIds[j]);
#else
                    nesting.push_back(j);
#endif
                }
            }
        }

        // re-roder the block extent to be compatible w/ VisIt
        int vExt[6] = {activeExt[0], activeExt[2], activeExt[4],
            activeExt[1], activeExt[3], activeExt[5]};

#ifdef USE_REAL_DOMAIN
        VisIt_DomainNesting_set_nestingForPatch(h, mmd->BlockIds[i],
            activeLevel, nesting.data(), nesting.size(), vExt);
#else
        VisIt_DomainNesting_set_nestingForPatch(h, i,
            activeLevel, nesting.data(), nesting.size(), vExt);
#endif

        nesting.clear();
    }

    return h;
}



//-----------------------------------------------------------------------------
// LibsimAnalysisAdaptor PUBLIC INTERFACE
//-----------------------------------------------------------------------------
senseiNewMacro(LibsimAnalysisAdaptor);

//-----------------------------------------------------------------------------
LibsimAnalysisAdaptor::LibsimAnalysisAdaptor()
{
    internals = new PrivateData;
}

//-----------------------------------------------------------------------------
LibsimAnalysisAdaptor::~LibsimAnalysisAdaptor()
{
    delete internals;
}

//-----------------------------------------------------------------------------
void LibsimAnalysisAdaptor::SetTraceFile(const std::string &s)
{
    internals->SetTraceFile(s);
}

//-----------------------------------------------------------------------------
void LibsimAnalysisAdaptor::SetOptions(const std::string &s)
{
    internals->SetOptions(s);
}

//-----------------------------------------------------------------------------
void LibsimAnalysisAdaptor::SetVisItDirectory(const std::string &s)
{
    internals->SetVisItDirectory(s);
}

//-----------------------------------------------------------------------------
void LibsimAnalysisAdaptor::SetComputeNesting(int val)
{
    internals->SetComputeNesting(val);
}

//-----------------------------------------------------------------------------
void LibsimAnalysisAdaptor::SetMode(const std::string &mode)
{
    internals->SetMode(mode);
}

//-----------------------------------------------------------------------------
bool LibsimAnalysisAdaptor::AddRender(int frequency, const std::string &session,
    const std::string &plots,
    const std::string &plotVars,
    bool slice, bool project2d,
    const double origin[3], const double normal[3],
    const LibsimImageProperties &imgProps)
{
    TimeEvent<128> mark("LibsimAnalysisAdaptor::AddRender");
    return internals->AddRender(frequency, session, plots, plotVars, slice,
      project2d, origin, normal, imgProps);
}

//-----------------------------------------------------------------------------
bool LibsimAnalysisAdaptor::AddExport(int frequency, const std::string &session,
    const std::string &plots,
    const std::string &plotVars,
    bool slice, bool project2d,
    const double origin[3], const double normal[3],
    const std::string &filename)
{
    TimeEvent<128> mark("LibsimAnalysisAdaptor::AddExport");
    return internals->AddExport(frequency, session, plots, plotVars, slice,
      project2d, origin, normal, filename);
}

//-----------------------------------------------------------------------------
void LibsimAnalysisAdaptor::Initialize()
{
    TimeEvent<128> mark("LibsimAnalysisAdaptor::Initialize");
    internals->SetComm(this->GetCommunicator());
    internals->Initialize();
}

//-----------------------------------------------------------------------------
bool LibsimAnalysisAdaptor::Execute(DataAdaptor* DataAdaptor)
{
    TimeEvent<128> mark("LibsimAnalysisAdaptor::Execute");
    return internals->Execute(DataAdaptor);
}

//-----------------------------------------------------------------------------
int LibsimAnalysisAdaptor::Finalize()
{
    TimeEvent<128> mark("LibsimAnalysisAdaptor::Finalize");
    delete this->internals;
    this->internals = nullptr;
    return 0;
}

}
