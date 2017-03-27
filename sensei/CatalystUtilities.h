#ifndef sensei_CatalystUtilities_h
#define sensei_CatalystUtilities_h

class vtkSMProxy;
class vtkSMViewProxy;
class vtkSMSourceProxy;
class vtkSMRepresentationProxy;

namespace sensei
{
namespace catalyst
{

vtkSMSourceProxy* CreatePipelineProxy( const char* group,
  const char* name, vtkSMProxy* input=nullptr);

void DeletePipelineProxy(vtkSMProxy* proxy);

vtkSMViewProxy* CreateViewProxy(const char* group,
  const char* name);

vtkSMRepresentationProxy* Show(vtkSMSourceProxy* producer,
  vtkSMViewProxy* view);

}
}

#endif
