#include "LibsimAnalysisAdaptor.h"
#include "LibsimImageProperties.h"
#include "DataAdaptor.h"
#include "VTKUtils.h"
#include "Timer.h"
#include "Error.h"

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

#include <VisItControlInterface_V2.h>
#include <VisItDataInterface_V2.h>

#include <sstream>
#include <algorithm>
#include <map>

#include <mpi.h>

#define DEBUG_PRINT

#define VISIT_COMMAND_PROCESS 0
#define VISIT_COMMAND_SUCCESS 1
#define VISIT_COMMAND_FAILURE 2

namespace sensei
{

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
struct MeshInfo
{
    MeshInfo() : dataObj(nullptr), datasets(), doms_per_rank(nullptr), size(1)
    {
    }

    ~MeshInfo()
    {
        if(doms_per_rank != nullptr)
            delete [] doms_per_rank;

        if(dataObj != nullptr)
        {
            dataObj->Delete();
            dataObj = nullptr;
        }

        for(size_t i = 0; i < datasets.size(); ++i)
        {
            if(datasets[i] != nullptr)
            {
                datasets[i]->Delete();
                datasets[i] = nullptr;
            }
        }
    }

    void SetDataObject(vtkDataObject *obj)
    {
        if(dataObj != nullptr)
            dataObj->Delete();
        dataObj = obj;
        if(dataObj != nullptr)
            dataObj->Register(nullptr);
    }

    vtkDataObject *GetDataObject()
    {
        return dataObj;
    }

    void SetDataSet(int idx, vtkDataSet *ds)
    {
        if(idx >= 0 && idx < (int)datasets.size())
        {
            if(datasets[idx] != nullptr)
                datasets[idx]->Delete();

            datasets[idx] = ds;
            if(ds != nullptr)
                ds->Register(nullptr);
        }
    }

    vtkDataObject            *dataObj;       // The data object returned from SENSEI.
    std::vector<vtkDataSet *> datasets;      // The leaf datasets of dataObj or dataObj itself if it was a simple vtkDataSet.
    int                      *doms_per_rank;
    int                       size;
};

std::ostream &operator << (std::ostream &os, const MeshInfo &obj)
{
    os <<"{dataObj=" << (void*)obj.dataObj << ", datasets=[";
    for(size_t i = 0; i < obj.datasets.size(); ++i)
        os << (void*)obj.datasets[i] << ", ";
    os << "], doms_per_rank=[";
    for(int i = 0; i < obj.size; ++i)
        os << obj.doms_per_rank[i] << ", ";
    os << "]}";
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
    void SetComm(MPI_Comm comm);
    void SetMode(const std::string &mode);

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
    static int broadcast_int(int *value, int sender, void *cbdata);
    static int broadcast_string(char *str, int len, int sender, void *cbdata);
    static void ControlCommandCallback(const char *cmd, const char *args, void *cbdata);
    static void SlaveProcessCallback(void *cbdata);
    static visit_handle GetMetaData(void *cbdata);
    static visit_handle GetMesh(int dom, const char *name, void *cbdata);
    static visit_handle GetVariable(int dom, const char *name, void *cbdata);
    static visit_handle GetDomainList(const char *name, void *cbdata);

    int ProcessVisItCommand(int rank);
    bool Execute_Batch(int rank);
    bool Execute_Interactive(int rank);

    void ClearMeshDataCache();
    MeshInfo *AddMeshDataCacheEntry(const std::string &meshName, int ndatasets);

    void FetchMesh(const std::string &meshName);
    int  AddArray(const std::string &meshName, int association, const std::string &arrayName);
    void GetArrayInfoFromVariableName(const std::string &varName,
                                      std::string &meshName, std::string &var, int &association);
    int *GetDomains(const std::string &meshName, int &size);

    int GetTotalDomains(const std::string &meshName) const;
    int GetLocalDomain(const std::string &meshName, int globaldomain) const;
    int GetNumDataSets(const std::string &meshName) const;
    vtkDataSet *GetDataSet(const std::string &meshName, int localdomain) const;

    int TopologicalDimension(const int dims[3]) const;
    std::string MakeFileName(const std::string &f, int timestep, double time) const;
    void DetermineExportFilename(const std::string &f, 
                                 std::string &fnoext, std::string &fmt) const;

    sensei::DataAdaptor              *da;
    std::map<std::string, MeshInfo *> meshData;

    std::string               traceFile, options, visitdir;
    std::vector<PlotRecord>   plots;
    MPI_Comm                  comm;
    std::string               mode;
    bool                      paused;
    static bool               initialized;
    static int                instances;
};

bool LibsimAnalysisAdaptor::PrivateData::initialized = false;
int  LibsimAnalysisAdaptor::PrivateData::instances = 0;

// --------------------------------------------------------------------------
LibsimAnalysisAdaptor::PrivateData::PrivateData() : da(nullptr),
  meshData(), traceFile(), options(), visitdir(), mode("batch"), 
  paused(false)
{
    comm = MPI_COMM_WORLD;
    ++instances;
}

// --------------------------------------------------------------------------
LibsimAnalysisAdaptor::PrivateData::~PrivateData()
{
    ClearMeshDataCache();

    --instances;
    if(instances == 0 && initialized)
    {
        timer::MarkEvent mark("libsim::finalize");
        if(VisItIsConnected())
            VisItDisconnect();
    }
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
    comm = c;
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
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_rank(comm, &size);
    if(rank == 0)
    {
        os << "traceFile = " << traceFile << endl;
        os << "options = " << options << endl;
        os << "visitdir = " << visitdir << endl;
        os << "mode = " << mode << endl;
        os << "initialized = " << (initialized ? "true" : "false") << endl;
        os << "meshData = {" << endl;
        std::map<std::string, MeshInfo*>::const_iterator it = meshData.begin();
        for( ; it != meshData.end(); ++it)
        {
             os << "\"" << it->first << "\" : " << *(it->second) << endl;
        }  
        os << "}" << endl;
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
//    cout << "Libsim Export: " << (retval?"true":"false") << ", " << p << endl;

    return retval;
}

// --------------------------------------------------------------------------
bool
LibsimAnalysisAdaptor::PrivateData::Initialize()
{
    // Load the runtime if we have not done it before.
    if(!initialized)
    {
        timer::MarkEvent mark("libsim::initialize");

        int rank = 0, size = 1;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        if(!traceFile.empty())
        {
            char suffix[100];
            snprintf(suffix, 100, ".%04d", rank);
            VisItOpenTraceFile((traceFile + suffix).c_str());
        }

        VisItDebug5("==== LibsimAnalysisAdaptor::PrivateData::Initialize ====\n");

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

        // Install comm into VisIt.
        VisItSetMPICommunicator((void *)&comm);

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
            if(rank == 0)
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
            }
            else
            {
                // Register Libsim callbacks.
                VisItSetSlaveProcessCallback2(SlaveProcessCallback, (void*)this); // needed in batch?
                VisItSetGetMetaData(GetMetaData, (void*)this);
                VisItSetGetMesh(GetMesh, (void*)this);
                VisItSetGetVariable(GetVariable, (void*)this);
                VisItSetGetDomainList(GetDomainList, (void*)this);

                initialized = true;
            }
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
    VisItDebug5("==== LibsimAnalysisAdaptor::PrivateData::Execute ====\n");

    // Keep a pointer to the data adaptor so the callbacks can access it.
    da = DataAdaptor;

    // If we for some reason have not initialized by now, do it.
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    bool retval = Initialize();

    // Let's get new metadata.
    VisItTimeStepChanged();

    if(mode.substr(0, 11) == "interactive")
        retval = Execute_Interactive(rank);
    else
        retval = Execute_Batch(rank);

    // Clear out any data that we've cached over the lifetime of this
    // Execute function.
    ClearMeshDataCache();

    return retval;
}

// --------------------------------------------------------------------------
bool
LibsimAnalysisAdaptor::PrivateData::Execute_Batch(int rank)
{
    bool retval = true;

    // NOTE: this executes a set of really simple pipelines prescribed by the
    //       options from the SENSEI config file.

    // Now that the runtime stuff is loaded, we can execute some plots.
    for(size_t i = 0; i < plots.size(); ++i)
    {
        // Skip if we're not executing now.
        if(da->GetDataTimeStep() % plots[i].frequency != 0)
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
                                    da->GetDataTimeStep(),
                                    da->GetDataTime());

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
            MPI_Bcast(&command, 1, MPI_INT, 0, this->comm);
            return 1;
        }
        else
        {
            command = VISIT_COMMAND_FAILURE;
            MPI_Bcast(&command, 1, MPI_INT, 0, this->comm);
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
            MPI_Bcast(&command, 1, MPI_INT, 0, this->comm);
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
        MPI_Bcast(&visitstate, 1, MPI_INT, 0, this->comm);

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
void
LibsimAnalysisAdaptor::PrivateData::ClearMeshDataCache()
{
    std::map<std::string, MeshInfo *>::iterator it = meshData.begin();
    for( ; it != meshData.end(); ++it)
        delete it->second;
    meshData.clear();
}

// --------------------------------------------------------------------------
MeshInfo *
LibsimAnalysisAdaptor::PrivateData::AddMeshDataCacheEntry(const std::string &meshName,
    int ndatasets)
{
    MeshInfo *mInfo = new MeshInfo;

    std::map<std::string, MeshInfo *>::iterator it = meshData.find(meshName);
    if(it != meshData.end())
    {
        // Delete the old information.
        delete it->second;
        // Add the new struct.
        it->second = mInfo;
    }
    else
    {
        // We found no entry so add it.
        meshData[meshName] = mInfo;
    }

    // We'll insert nullptr pointers for the datasets since we don't have their
    // complete definitions yet.
    for(int i = 0; i < ndatasets; ++i)
        mInfo->datasets.push_back(nullptr);

    // Determine the number of domains on each rank so we can
    // make the right metadata and later do the domain list right.
    int rank = 0, size = 1;
    MPI_Comm_rank(this->comm, &rank);
    MPI_Comm_size(this->comm, &size);
    mInfo->size = size;
    mInfo->doms_per_rank = new int[size];
    memset(mInfo->doms_per_rank, 0, sizeof(int) * size);
    int ndoms = ndatasets;
#ifdef DEDBUG_PRINT
    char tmp[100];
    sprintf(tmp, "Rank %d has %d domains.\n", rank, ndoms);
    VisItDebug5(tmp);
#endif
    MPI_Allgather(&ndoms, 1, MPI_INT,
                  mInfo->doms_per_rank, 1, MPI_INT, this->comm);

#ifdef DEDBUG_PRINT
    VisItDebug5("doms_per_rank = {");
    for(int i = 0; i < size; ++i)
    {
        sprintf(tmp, "%d, ", mInfo->doms_per_rank[i]);
        VisItDebug5(tmp);
    }
    VisItDebug5("}\n");
#endif
    return mInfo;
}

// --------------------------------------------------------------------------
int
LibsimAnalysisAdaptor::PrivateData::GetTotalDomains(const std::string &meshName) const
{
    int total = 0;
    std::map<std::string, MeshInfo *>::const_iterator it;
    if((it = meshData.find(meshName)) != meshData.end())
    {
        int size = 1;
        MPI_Comm_size(comm, &size);
        for(int i = 0; i < size; ++i)
            total += it->second->doms_per_rank[i];
    }
    return total;
}

// --------------------------------------------------------------------------
int *
LibsimAnalysisAdaptor::PrivateData::GetDomains(const std::string &meshName, int &size)
{
    int *iptr = nullptr;
    size = 0;
    std::map<std::string, MeshInfo *>::const_iterator it;
    if((it = meshData.find(meshName)) != meshData.end())
    {
        int rank = 0;
        MPI_Comm_rank(this->comm, &rank);
        MPI_Comm_size(this->comm, &size);

        // Compute the offset to this rank's domains.
        int offset = 0;
        for(int i = 0; i < rank; ++i)
            offset += it->second->doms_per_rank[i];

        // Make a list of this rank's domains using global domain ids.
        iptr = (int *)malloc(sizeof(int) * it->second->doms_per_rank[rank]);
        for(int i = 0; i < it->second->doms_per_rank[rank]; ++i)
            iptr[i] = offset + i;

        size = it->second->doms_per_rank[rank];
    }
    return iptr;
}

// --------------------------------------------------------------------------
int
LibsimAnalysisAdaptor::PrivateData::GetLocalDomain(const std::string &meshName, int globaldomain) const
{
    int dom = -1;
    std::map<std::string, MeshInfo *>::const_iterator it;
    if((it = meshData.find(meshName)) != meshData.end())
    {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        int offset = 0;
        for(int i = 0; i < rank; ++i)
            offset += it->second->doms_per_rank[i];

        if(globaldomain >= offset && globaldomain < offset+it->second->doms_per_rank[rank])
            dom = globaldomain - offset;
    }

    return dom;
}

// --------------------------------------------------------------------------
// @brief Returns the number of datasets for the mesh on this MPI rank.
int
LibsimAnalysisAdaptor::PrivateData::GetNumDataSets(const std::string &meshName) const
{
    int ndom = 0;
    std::map<std::string, MeshInfo *>::const_iterator it;
    if((it = meshData.find(meshName)) != meshData.end())
    {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        ndom = it->second->doms_per_rank[rank];
    }
    return ndom;
}

// --------------------------------------------------------------------------
// @brief Returns the VTK dataset for the mesh given the local domain numbering.
//        This may return nullptr if our real mesh has not been cached yet. that
//        happens in the GetMesh Libsim callback. 
vtkDataSet *
LibsimAnalysisAdaptor::PrivateData::GetDataSet(const std::string &meshName, int localdomain) const
{
    vtkDataSet *ds = nullptr;
    std::map<std::string, MeshInfo *>::const_iterator it;
    if((it = meshData.find(meshName)) != meshData.end())
    {
        if(localdomain >= 0 && localdomain < (int)it->second->datasets.size())
            ds = it->second->datasets[localdomain];
    }

    return ds;
}

// --------------------------------------------------------------------------
void
LibsimAnalysisAdaptor::PrivateData::FetchMesh(const std::string &meshName)
{
    std::map<std::string, MeshInfo *>::const_iterator mit;
    if((mit = meshData.find(meshName)) != meshData.end())
    {
        // Get the mesh, the whole thing. No vars though.
        vtkDataObject *obj = nullptr;
        bool structureOnly = false;
        if (da->GetMesh(meshName, structureOnly, obj))
        {
            SENSEI_ERROR("GetMesh request failed.")
        }
        else
        {
            // SENSEI gave us the data object. Save it off. We'll use it in
            // other callbacks to get vars, etc.
            mit->second->SetDataObject(obj);

            // Unpack the data object into a vector of vtkDataSets if it is a
            // compound dataset.
            int dsId = 0;
            vtkCompositeDataSet *cds = vtkCompositeDataSet::SafeDownCast(obj);
            if(cds != nullptr)
            {
                vtkCompositeDataIterator *it = cds->NewIterator();
                it->SkipEmptyNodesOn();
                it->InitTraversal();
                while(!it->IsDoneWithTraversal())
                {
                    vtkDataObject *obj2 = cds->GetDataSet(it);
                    if(obj2 != nullptr && vtkDataSet::SafeDownCast(obj2) != nullptr)
                        mit->second->SetDataSet(dsId++, vtkDataSet::SafeDownCast(obj2));
                    it->GoToNextItem();
                }
            }
            else if(vtkDataSet::SafeDownCast(obj) != nullptr)
            {
                mit->second->SetDataSet(0, vtkDataSet::SafeDownCast(obj));
            }

            // If the data adaptor can provide a ghost nodes array, add it to the
            // data object now. The datasets we've registered will then have that
            // as a point data array.
            int nLayers = 0;
            if(da->GetMeshHasGhostNodes(meshName, nLayers) == 0)
            {
                if(nLayers > 0)
                    da->AddGhostNodesArray(obj, meshName);
            }

            // If the data adaptor can provide a ghost cells array, add it to the
            // data object now. The datasets we've registered will then have that
            // as a point data array.
            if(da->GetMeshHasGhostCells(meshName, nLayers) == 0)
            {
                if(nLayers > 0)
                    da->AddGhostCellsArray(obj, meshName);
            }
        }
    }
}

// --------------------------------------------------------------------------
int
LibsimAnalysisAdaptor::PrivateData::AddArray(const std::string &meshName, 
    int association, const std::string &arrayName)
{
    int retval = 1;
    std::map<std::string, MeshInfo *>::iterator mit;
    if((mit = meshData.find(meshName)) != meshData.end())
    {
        retval = da->AddArray(mit->second->dataObj, meshName, association, arrayName);
#ifdef DEBUG_PRINT
        VisItDebug5("da->AddArray returned %d\n", retval);
#endif
    }
    return retval;
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
void
LibsimAnalysisAdaptor::PrivateData::GetArrayInfoFromVariableName(
    const std::string &varName,
    std::string &meshName, std::string &var, int &association)
{
    // Get the mesh names from the data adaptor and figure out the mesh name
    // that we're using for this variable.
    std::vector<std::string> meshNames;
    da->GetMeshNames(meshNames);
    bool findAssociation = false;
    if(meshNames.size() > 1)
    {
        std::string::size_type pos = varName.find("/");
        if(pos != std::string::npos)
        {
            meshName = varName.substr(0, pos);
            std::string tmpVar = varName.substr(pos+1, std::string::npos);
            if(tmpVar.substr(0, 5) == "cell_")
            {
                var = tmpVar.substr(5, std::string::npos);
                association = vtkDataObject::FIELD_ASSOCIATION_CELLS;
            }
            else
            {
                var = tmpVar;
                findAssociation = true;
            }
        }
    }
    else
    {
        meshName = meshNames[0];
        if(varName.substr(0, 5) == "cell_")
        {
            var = varName.substr(5, std::string::npos);
            association = vtkDataObject::FIELD_ASSOCIATION_CELLS;
        }
        else
        {
            var = varName;
            findAssociation = true;
        }
    }

    if(findAssociation)
    {
        std::vector<std::string> pointvars;
        da->GetArrayNames(meshName, vtkDataObject::FIELD_ASSOCIATION_POINTS, pointvars);
        if(std::find(pointvars.begin(), pointvars.end(), var) != pointvars.end())
            association = vtkDataObject::FIELD_ASSOCIATION_POINTS;
        else
        {
            std::vector<std::string> cellvars;
            da->GetArrayNames(meshName, vtkDataObject::FIELD_ASSOCIATION_CELLS, cellvars);
            if(std::find(cellvars.begin(), cellvars.end(), var) != cellvars.end())
                association = vtkDataObject::FIELD_ASSOCIATION_CELLS;
        }
    }
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

                if(!copy)
                {
                    VisItDebug5("==== Standard memory layout: nc=%d, nt=%d ====\n", nc, nt);
                }
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
                VisItDebug5("==== Copying required: nc=%d, nt=%d ====\n", nc, nt);

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
vtkDataSet_to_VisIt_Mesh(vtkDataSet *ds, DataAdaptor */*da*/)
{
    visit_handle mesh = VISIT_INVALID_HANDLE;
    vtkImageData *igrid = vtkImageData::SafeDownCast(ds);
    vtkRectilinearGrid *rgrid = vtkRectilinearGrid::SafeDownCast(ds);
    vtkStructuredGrid  *sgrid = vtkStructuredGrid::SafeDownCast(ds);
    vtkPolyData *pgrid = vtkPolyData::SafeDownCast(ds);
    vtkUnstructuredGrid *ugrid = vtkUnstructuredGrid::SafeDownCast(ds);
    if(igrid != nullptr)
    {
        VisItDebug5("\tExposing vtkImageData as a rectilinear grid.\n");

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
#if 0
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
if(rank == 0)
{
cout << "dims=" << dims[0] << ", " << dims[1] << ", " << dims[2] << endl;
cout << "ext=" << ext[0] << ", " << ext[1] << ", " << ext[2] << ", "
               << ext[3] << ", " << ext[4] << ", " << ext[5] << endl;
cout << "x0=" << x0[0] << ", " << x0[1] << ", " << x0[2] << endl;
cout << "dx=" << dx[0] << ", " << dx[1] << ", " << dx[2] << endl;
}
#endif
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
                    for(int i = 0; i < nz; ++i)
                        z[i] = x0[2] + (ext[4] + i)*dx[2];
                    VisIt_VariableData_setDataF(xc, VISIT_OWNER_VISIT, 1, nx, x);
                    VisIt_VariableData_setDataF(yc, VISIT_OWNER_VISIT, 1, ny, y);
                    VisIt_VariableData_setDataF(zc, VISIT_OWNER_VISIT, 1, nz, z);
                    VisIt_RectilinearMesh_setCoordsXYZ(mesh, xc, yc, zc);

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
        VisItDebug5("vtkUnstructuredGrid: npts = %d, ncells = %d\n", 
            (int)ugrid->GetNumberOfPoints(), (int)ugrid->GetNumberOfCells());
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
        SENSEI_ERROR("Unsupported VTK mesh type \"" << ds->GetClassName() << "\"")
        VisItDebug5("Unsupported VTK mesh type.\n");
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
    return MPI_Bcast(value, 1, MPI_INT, sender, This->comm);
}

// --------------------------------------------------------------------------
int
LibsimAnalysisAdaptor::PrivateData::broadcast_string(char *str, int len, int sender, void *cbdata)
{
    PrivateData *This = (PrivateData *)cbdata;
    return MPI_Bcast(str, len, MPI_CHAR, sender, This->comm);
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
    PrivateData *This = (PrivateData *)cbdata;
    sensei::DataAdaptor *da = This->da;

#ifdef DEBUG_PRINT
    char msg[1000];
    VisItDebug5("==== LibsimAnalysisAdaptor::PrivateData::GetMetaData ====\n");
#endif

    // Get the mesh names.
    std::vector<std::string> meshNames;
    if (da->GetMeshNames(meshNames))
    {
        SENSEI_ERROR("Failed to get mesh names")
        return VISIT_INVALID_HANDLE;
    }
#ifdef DEBUG_PRINT
    VisItDebug5("meshNames = {");
    for(size_t i = 0; i < meshNames.size(); ++i)
    {
        sprintf(msg, "%s,", meshNames[i].c_str());
        VisItDebug5(msg);
    }
    VisItDebug5("}\n");
#endif

    // Create metadata.
    visit_handle md = VISIT_INVALID_HANDLE;
    if (VisIt_SimulationMetaData_alloc(&md) != VISIT_OKAY)
    {
        SENSEI_ERROR("Failed to allocate simulation metadata")
        return VISIT_INVALID_HANDLE;
    }

    // Set the simulation state.
    VisIt_SimulationMetaData_setMode(md, This->paused ? VISIT_SIMMODE_STOPPED : VISIT_SIMMODE_RUNNING);
    VisIt_SimulationMetaData_setCycleTime(md, da->GetDataTimeStep(), da->GetDataTime());

    unsigned int nMeshes = meshNames.size();
    for (unsigned int j = 0; j < nMeshes; ++j)
    {
        const std::string &meshName = meshNames[j];

#if 1
// NOTE: This stupid big block of code basically does 3 things.
// 1. Make an attempt to get the actual mesh types by querying SENSEI for the mesh structure.
//    How important is this? Could I just tell VisIt I don't know the mesh type? Downsides?
// 2. Take the VTK data object and see if it is really a collection of domains.
//    if it is then we keep track of the leaves as separate domains that we'll
//    tell VisIt about since VisIt probably won't be too happy about getting a
//    multiblock dataset. It might work for GetMesh but GetVar would probably flop.
// 3. (future) see if there are AMR settings we can glean from the mesh structure.

#ifdef DEBUG_PRINT
        sprintf(msg, "GetMesh(%s) structure only\n", meshName.c_str());
        VisItDebug5(msg);
#endif

        // Get the mesh, structure only.
        vtkDataObject *obj = nullptr;

        // Do not bother with structure-only. Most data adaptors are stupid and
        // it causes problems if we later call GetMesh with different values
        // of structureOnly.
        bool structureOnly = false;

        if (da->GetMesh(meshName, structureOnly, obj))
        {
            SENSEI_ERROR("GetMesh request failed. Skipping that mesh.")
            continue;
        }

        // Unpack the data object into a vector of vtkDataSets if it is a compound 
        // dataset. These datasets will be incomplete and just for the structure 
        // and number of domains only.
        std::vector<vtkDataSet *> datasets;
        vtkCompositeDataSet *cds = vtkCompositeDataSet::SafeDownCast(obj);
        if(cds != nullptr)
        {
            vtkCompositeDataIterator *it = cds->NewIterator();
            it->SkipEmptyNodesOn();
            it->InitTraversal();
            while(!it->IsDoneWithTraversal())
            {
                vtkDataObject *obj2 = cds->GetDataSet(it);
                if(obj2 != nullptr && vtkDataSet::SafeDownCast(obj2) != nullptr)
                    datasets.push_back(vtkDataSet::SafeDownCast(obj2));
                it->GoToNextItem();
            }
        }
        else if(vtkDataSet::SafeDownCast(obj) != nullptr)
        {
            datasets.push_back(vtkDataSet::SafeDownCast(obj));
        }
        else
        {
            SENSEI_ERROR("The data object is not supported data type. Skipping it.")
            continue;
        }
#ifdef DEBUG_PRINT
        VisItDebug5("datasets.size() = %d\n", (int)datasets.size());
#endif

        // Let's create a new mesh information object to contain the data object and
        // its sub-datasets.
        MeshInfo *mInfo = This->AddMeshDataCacheEntry(meshName, (int)datasets.size());

        // Now, let's create some metadata for the object.

        // Not all ranks might have data when we have multiple meshes. Figure out
        // which rank has data and we'll let that one broadcast the information 
        // about this mesh to all ranks.
        int rank = 0, size = 1;
        MPI_Comm_rank(This->comm, &rank);
        MPI_Comm_size(This->comm, &size);
        int bcast_rank = -1;
        for(int i = 0; i < size; ++i)
        {
            if(mInfo->doms_per_rank[i] > 0)
            {
                bcast_rank = i;
                break;
            }
        }
        if(bcast_rank == -1)
            continue;
#ifdef DEBUG_PRINT
        std::stringstream ss;
        ss << *mInfo << endl;
        VisItDebug5(ss.str().c_str());

        sprintf(msg, "bcast_rank=%d\n", bcast_rank);
        VisItDebug5(msg);
#endif

        // Populate mesh information on bcast_rank.
        int dims[3] = {0,0,0};
        int iMesh[5] = {-1,0,0,0,0};
        if(bcast_rank == rank)
        {
            // ASSUMPTION FOR NOW: datasets will be the same type data on all ranks.
            vtkDataSet *ds = datasets[0];

            if (vtkImageData *igrid = vtkImageData::SafeDownCast(ds))
            {
// TODO: if cds != nullptr, can we get some AMR properties?
                igrid->GetDimensions(dims);
                iMesh[0] = VISIT_MESHTYPE_RECTILINEAR;
                iMesh[1] = dims[0];
                iMesh[2] = dims[1];
                iMesh[3] = dims[2];
            }
            else if (vtkRectilinearGrid *rgrid = vtkRectilinearGrid::SafeDownCast(ds))
            {
// TODO: if cds != nullptr, can we get some AMR properties?
                rgrid->GetDimensions(dims);
                iMesh[0] = VISIT_MESHTYPE_RECTILINEAR;
                iMesh[1] = dims[0];
                iMesh[2] = dims[1];
                iMesh[3] = dims[2];
            }
            else if (vtkStructuredGrid *sgrid = vtkStructuredGrid::SafeDownCast(ds))
            {
                sgrid->GetDimensions(dims);
                iMesh[0] = VISIT_MESHTYPE_CURVILINEAR;
                iMesh[1] = dims[0];
                iMesh[2] = dims[1];
                iMesh[3] = dims[2];
            }
            else if (vtkUnstructuredGrid *ugrid = vtkUnstructuredGrid::SafeDownCast(ds))
            {
                (void)ugrid;
                iMesh[0] = VISIT_MESHTYPE_UNSTRUCTURED;
                iMesh[1] = 3;// just do 3.
                iMesh[2] = 3;
            }
            else if (vtkPolyData *pgrid = vtkPolyData::SafeDownCast(ds))
            {
                if (pgrid->GetVerts())
                {
                    iMesh[0] = VISIT_MESHTYPE_POINT;
                    iMesh[1] = 0;
                    iMesh[2] = 3;
                }
            }
            else
            {
                cout << "Libsim adaptor does not currently support: "
                     << ds->GetClassName() << " datasets." << endl;
            }
        }
        // Broadcast the iMesh data to all.
        MPI_Bcast(iMesh, 5, MPI_INT, bcast_rank, This->comm);
        VisItDebug5("iMesh = {%d, %d, %d, %d, %d}\n", 
                    iMesh[0],iMesh[1],iMesh[2],iMesh[3],iMesh[4]);
        // Add mesh metadata.
        visit_handle mmd = VISIT_INVALID_HANDLE;
        if (VisIt_MeshMetaData_alloc(&mmd) != VISIT_OKAY)
        {
            SENSEI_ERROR("Failed to allocate mesh metadata")
            return VISIT_INVALID_HANDLE;
        }

        // Use the iMesh data to make metadata.
        bool supported = true;
        switch(iMesh[0])
        {
        case VISIT_MESHTYPE_RECTILINEAR:
            dims[0] = iMesh[1];
            dims[1] = iMesh[2];
            dims[2] = iMesh[3];
            VisIt_MeshMetaData_setTopologicalDimension(mmd, This->TopologicalDimension(dims));
            VisIt_MeshMetaData_setMeshType(mmd, VISIT_MESHTYPE_RECTILINEAR);
            VisIt_MeshMetaData_setSpatialDimension(mmd, This->TopologicalDimension(dims));
            break;
        case VISIT_MESHTYPE_CURVILINEAR:
            dims[0] = iMesh[1];
            dims[1] = iMesh[2];
            dims[2] = iMesh[3];
            VisIt_MeshMetaData_setTopologicalDimension(mmd, This->TopologicalDimension(dims));
            VisIt_MeshMetaData_setMeshType(mmd, VISIT_MESHTYPE_CURVILINEAR);
            VisIt_MeshMetaData_setSpatialDimension(mmd, This->TopologicalDimension(dims));
            break;
        case VISIT_MESHTYPE_UNSTRUCTURED:
            VisIt_MeshMetaData_setMeshType(mmd, VISIT_MESHTYPE_UNSTRUCTURED);
            VisIt_MeshMetaData_setTopologicalDimension(mmd, iMesh[1]);
            VisIt_MeshMetaData_setSpatialDimension(mmd, iMesh[2]);
            break;
        case VISIT_MESHTYPE_POINT:
            VisIt_MeshMetaData_setMeshType(mmd, VISIT_MESHTYPE_POINT);
            VisIt_MeshMetaData_setTopologicalDimension(mmd, iMesh[1]);
            VisIt_MeshMetaData_setSpatialDimension(mmd, iMesh[2]);
            break;
        default:
            supported = false;
            break;
        }

        // If we had a supported mesh type then add the mesh to the metadata.
        if(supported)
        {
#ifdef DEBUG_PRINT
            VisItDebug5("mesh: %s\n", meshName.c_str());
#endif
            VisIt_MeshMetaData_setName(mmd, meshName.c_str());
            VisIt_MeshMetaData_setNumDomains(mmd, This->GetTotalDomains(meshName));
            VisIt_SimulationMetaData_addMesh(md, mmd);
        }
        else
        {
            SENSEI_ERROR("Unsupported mesh type for \"" << meshName << "\"")
            continue;
        }
#endif 
        //
        // Add variables.
        //

        // ISSUE: The SENSEI API doesn't tell us the number of components for
        //        the variables so we don't know whether it's a scalar, vector, etc.

        // get point data arrays. It seems that data adaptors are allowed to fail
        // on this so don't bother checking the return value. Just check the
        // vector of names.
        int assoc = vtkDataObject::FIELD_ASSOCIATION_POINTS;
        std::vector<std::string> node_vars;
        da->GetArrayNames(meshName, assoc, node_vars);
        unsigned int nArrays = node_vars.size();
#if 1
        VisItDebug5("#node vars: %d\n", nArrays);
#endif
        for(unsigned int i = 0; i < nArrays; ++i)
        {
            visit_handle vmd = VISIT_INVALID_HANDLE;
            if (VisIt_VariableMetaData_alloc(&vmd) != VISIT_OKAY)
            {
                SENSEI_ERROR("Failed to allocate variable metadata")
                return VISIT_INVALID_HANDLE;
            }
            std::string arrayName(node_vars[i]);
            if(nMeshes > 1)
                arrayName = (meshName + "/") + arrayName;
#if 1
            VisItDebug5("point var: %s\n", arrayName.c_str());
#endif
            VisIt_VariableMetaData_setName(vmd, arrayName.c_str());
            VisIt_VariableMetaData_setMeshName(vmd, meshName.c_str());
            VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
            VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_NODE);
            VisIt_SimulationMetaData_addVariable(md, vmd);
        }

        // get cell data arrays
        assoc = vtkDataObject::FIELD_ASSOCIATION_CELLS;
        std::vector<std::string> cell_vars;
        da->GetArrayNames(meshName, assoc, cell_vars);
        nArrays = cell_vars.size();
#if 1
        VisItDebug5("#cell vars: %d\n", nArrays);
#endif
        for(unsigned int i = 0; i < nArrays; ++i)
        {
            visit_handle vmd = VISIT_INVALID_HANDLE;
            if (VisIt_VariableMetaData_alloc(&vmd) != VISIT_OKAY)
            {
                SENSEI_ERROR("Failed to allocate variable metadata")
                return VISIT_INVALID_HANDLE;
            }

            std::string arrayName = cell_vars[i];

            // See if the variable is already in the nodal vars. If so,
            // we prepend "cell_" to the name.
            bool alreadyDefined = std::find(node_vars.begin(),
                 node_vars.end(), arrayName) != node_vars.end();

            if (alreadyDefined)
            {
                if(nMeshes > 1)
                    arrayName = (meshName + "/cell_") + arrayName;
                else
                    arrayName = std::string("cell_") + arrayName;
            }
            else if(nMeshes > 1)
            {
                arrayName = (meshName + "/") + arrayName;
            }

#if 1
            VisItDebug5("cell var: %s\n", arrayName.c_str());
#endif
            VisIt_VariableMetaData_setName(vmd, arrayName.c_str());
            VisIt_VariableMetaData_setMeshName(vmd, meshName.c_str());
            VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
            VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);
            VisIt_SimulationMetaData_addVariable(md, vmd);
        }
    }

    // Add some commands.
    static const char *cmd_names[] = {"pause", "run"};
    for(int i = 0; i < static_cast<int>(sizeof(cmd_names)/sizeof(const char *)); ++i)
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
visit_handle
LibsimAnalysisAdaptor::PrivateData::GetMesh(int dom, const char *name, void *cbdata)
{
    VisItDebug5("==== LibsimAnalysisAdaptor::PrivateData::GetMesh ====\n");
    PrivateData *This = (PrivateData *)cbdata;

    std::string meshName(name);
    int localdomain = This->GetLocalDomain(meshName, dom);
    visit_handle mesh = VISIT_INVALID_HANDLE;

#ifdef DEBUG_PRINT
    char tmp[200];
    sprintf(tmp, "\tdom=%d, localdomain = %d, nLocalDomains=%d\n",
            dom, localdomain, (int)This->GetNumDataSets(meshName));
    VisItDebug5(tmp);
#endif

    if(localdomain >= 0)
    {
        // Get the dataset for localdomain.
        vtkDataSet *ds = This->GetDataSet(meshName, localdomain);

        // If we have not retrieved the dataset for localdomain, do that now.
        if(ds == nullptr)
        {
            This->FetchMesh(meshName);
            ds = This->GetDataSet(meshName, localdomain);
        }

        mesh = vtkDataSet_to_VisIt_Mesh(ds, This->da);
    }

    return mesh;
}

// --------------------------------------------------------------------------
visit_handle
LibsimAnalysisAdaptor::PrivateData::GetVariable(int dom, const char *name, void *cbdata)
{
    PrivateData *This = (PrivateData *)cbdata;
    VisItDebug5("==== LibsimAnalysisAdaptor::PrivateData::GetVariable ====\n");

    // Given the VisIt variable name, turn it back into the SENSEI variable name.
    std::string meshName, varName;
    int association;
    This->GetArrayInfoFromVariableName(name, meshName, varName, association);
#ifdef DEBUG_PRINT
    VisItDebug5("dom=%d, name=%s\n", dom, name);
    VisItDebug5("meshName=%s, varName=%s, association=%d\n",
                meshName.c_str(), varName.c_str(), association);
#endif

    // Get the local domain.
    int localdomain = This->GetLocalDomain(meshName, dom);
    visit_handle h = VISIT_INVALID_HANDLE;
#ifdef DEBUG_PRINT
    VisItDebug5("localdomain=%d\n", localdomain);
#endif

    if(localdomain >= 0)
    {
        // See if the right data array exists in the VTK dataset.
        vtkDataSet *ds = This->GetDataSet(meshName, localdomain);

        // See if the array is present.
        vtkDataArray *arr = nullptr;
        if(association == vtkDataObject::FIELD_ASSOCIATION_POINTS)
            arr = ds->GetPointData()->GetArray(varName.c_str());
        else
            arr = ds->GetCellData()->GetArray(varName.c_str());

#ifdef DEBUG_PRINT
        VisItDebug5("arr=%p\n", arr);
#endif

        // If we did not find the array then get it from SENSEI's
        // data adaptor.
        if(arr == nullptr)
        {
            This->AddArray(meshName, association, varName);

            // Look for the data array again.
            if(association == vtkDataObject::FIELD_ASSOCIATION_POINTS)
                arr = ds->GetPointData()->GetArray(varName.c_str());
            else
                arr = ds->GetCellData()->GetArray(varName.c_str());
#ifdef DEBUG_PRINT
            VisItDebug5("After AddArray: arr=%p\n", arr);
#endif
        }

        // Wrap the VTK data array's data as a VisIt_VariableData.
        if(arr != nullptr)
        {
#ifdef DEBUG_PRINT
            VisItDebug5("Converting to VisIt_VariableData\n");
#endif
            h = vtkDataArray_To_VisIt_VariableData(arr);
        }
    }

    return h;
}

// --------------------------------------------------------------------------
visit_handle
LibsimAnalysisAdaptor::PrivateData::GetDomainList(const char *name, void *cbdata)
{
    PrivateData *This = (PrivateData *)cbdata;
    visit_handle h = VISIT_INVALID_HANDLE;
    std::string meshName(name);
    VisItDebug5("==== LibsimAnalysisAdaptor::PrivateData::GetDomainList ====\n");

    if(VisIt_DomainList_alloc(&h) != VISIT_ERROR)
    {
        visit_handle hdl;
        int *iptr = nullptr, size = 0;

        // Create a list of domains owned by this rank.
        iptr = This->GetDomains(name, size);

        VisIt_VariableData_alloc(&hdl);
        VisIt_VariableData_setDataI(hdl, VISIT_OWNER_VISIT, 1, size, iptr);
        VisIt_DomainList_setDomains(h, This->GetTotalDomains(meshName), hdl);
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
    return internals->AddExport(frequency, session, plots, plotVars, slice,
      project2d, origin, normal, filename);
}

//-----------------------------------------------------------------------------
void LibsimAnalysisAdaptor::Initialize()
{
    internals->SetComm(this->GetCommunicator());
    internals->Initialize();
}

//-----------------------------------------------------------------------------
bool LibsimAnalysisAdaptor::Execute(DataAdaptor* DataAdaptor)
{
    timer::MarkEvent mark("libsim::execute");
    return internals->Execute(DataAdaptor);
}

//-----------------------------------------------------------------------------
int LibsimAnalysisAdaptor::Finalize()
{
  delete this->internals;
  this->internals = nullptr;
  return 0;
}

//-----------------------------------------------------------------------------
void LibsimAnalysisAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
    internals->PrintSelf(os, indent);
}

}
