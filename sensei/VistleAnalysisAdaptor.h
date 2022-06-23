#ifndef VistleAnalysisAdaptor_h
#define VistleAnalysisAdaptor_h

#include "AnalysisAdaptor.h"

#include <string>
#include <memory>

namespace vistle{
class Object;
class DataBase;
namespace insitu{
namespace sensei{
 class SenseiAdapter; 
}
}
}

namespace sensei
{

/// @brief Analysis adaptor for Vistle https://github.com/vistle/vistle

class VistleAnalysisAdaptor : public AnalysisAdaptor
{
public:
  static VistleAnalysisAdaptor* New();
  senseiTypeMacro(VistleAnalysisAdaptor, AnalysisAdaptor);

    /// @brief Initialize vistle with the available meshes and their data arrays. 
    /// This assumes this data does noth change during simulation
    bool Initialize(DataAdaptor* data);

    bool Execute(DataAdaptor* dataIn, DataAdaptor** dataOut) override;

    int Finalize() override;

    /// Set configured Vistle startup options.
    void SetTraceFile(const std::string &traceFile);
    void SetOptions(const std::string &options);
    void SetMode(const std::string &mode);
    void SetFrequency(int f);
    int SetCommunicator(MPI_Comm comm) override;

  protected:
    VistleAnalysisAdaptor();
    ~VistleAnalysisAdaptor();

  private:

    VistleAnalysisAdaptor(const VistleAnalysisAdaptor&); // Not implemented.
    void operator=(const VistleAnalysisAdaptor&); // Not implemented.

    class PrivateData;
    PrivateData *m_internals;


    
};


}




#endif
