#ifndef sensei_Calculator_h
#define sensei_Calculator_h

#include "AnalysisAdaptor.h"

namespace sensei
{

class Calculator : public AnalysisAdaptor
{
public:
  static Calculator* New();
  senseiTypeMacro(Calculator, AnalysisAdaptor);

  void Initialize(const std::string& meshName, int association, const std::string& expression, const std::string& result);
  bool Execute(DataAdaptor* data, DataAdaptor*&) override;
  int Finalize() override;

protected:
  Calculator();
  ~Calculator();

private:
  Calculator(const Calculator&) = delete;
  void operator=(const Calculator&) = delete;
  std::string Result;
  std::string MeshName;
  std::string Expression;
  int Association;
};

}

#endif
