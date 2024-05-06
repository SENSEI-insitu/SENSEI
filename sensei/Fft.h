#ifndef Fft_h
#define Fft_h

#include <AnalysisAdaptor.h>

class svtkDataArray;
class svtkDataObject;

namespace sensei
{
/// Performs Fast Fourier Transform in parallel.
class SENSEI_EXPORT Fft : public AnalysisAdaptor
{
public:
  /// allocates a new instance
  static Fft* New();

  senseiTypeMacro(Fft, AnalysisAdaptor);

  /// @brief  Sets up new instance of FFT endpoint
  /// @param direction FFTW_FORWARD / FFTW_INVERSE (based on fftw lib.)
  /// @param python_xml XML for further python based image display
  /// @param mesh_name name of the mesh in simulation
  void Initialize(std::string const& direction, std::string const& python_xml, std::string const& mesh_name);

  /// @brief Execute the FFTW analysis endpoint
  /// @param data Data Adaptor input with simulation data
  /// @param * Adaptor output for returning data
  /// @return 0 or 1
  bool Execute(DataAdaptor* data, DataAdaptor**) override;

  /// finalize the run
  int Finalize() override;

  // MPI_Comm GetCommunicator() { return this->Comm; }

protected:
  Fft();
  ~Fft();

  Fft(const Fft&) = delete;
  void operator=(const Fft&) = delete;

  MPI_Comm Comm;
  
  struct InternalsType;
  InternalsType *Internals;

  svtkDataArray* GetArray(svtkDataObject* dobj, const std::string& arrayname);
};

}
#endif
