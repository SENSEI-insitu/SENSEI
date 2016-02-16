#ifndef vtkCatalystUtilities_h
#define vtkCatalystUtilities_h

#include <vtkNew.h>
#include <vtkSMParaViewPipelineController.h>
#include <vtkSMPropertyHelper.h>
#include <vtkSMProxy.h>
#include <vtkSMProxyManager.h>
#include <vtkSMSessionProxyManager.h>
#include <vtkSMSourceProxy.h>

#include <vtkSMParaViewPipelineControllerWithRendering.h>
#include <vtkSMRenderViewProxy.h>
#include <vtkSMPVRepresentationProxy.h>
#include <vtkDataObject.h>

namespace catalyst
{
  vtkSMSourceProxy* CreatePipelineProxy(const char* group, const char* name, vtkSMProxy* input=NULL)
    {
    vtkSMSessionProxyManager* pxm =
      vtkSMProxyManager::GetProxyManager()->GetActiveSessionProxyManager();
    vtkSmartPointer<vtkSMProxy> proxy;
    proxy.TakeReference(pxm->NewProxy(group, name));
    if (!proxy || !vtkSMSourceProxy::SafeDownCast(proxy))
      {
      return NULL;
      }
    vtkNew<vtkSMParaViewPipelineController> controller;
    controller->PreInitializeProxy(proxy);
    if (input)
      {
      vtkSMPropertyHelper(proxy, "Input").Set(input);
      }
    controller->PostInitializeProxy(proxy);
    controller->RegisterPipelineProxy(proxy);
    return vtkSMSourceProxy::SafeDownCast(proxy);
    }

  void DeletePipelineProxy(vtkSMProxy* proxy)
    {
    if (proxy)
      {
      vtkNew<vtkSMParaViewPipelineController> controller;
      controller->UnRegisterProxy(proxy);
      }
    }

  vtkSMProxy* CreateViewProxy(const char* group, const char* name)
    {
    vtkSMSessionProxyManager* pxm =
      vtkSMProxyManager::GetProxyManager()->GetActiveSessionProxyManager();
    vtkSmartPointer<vtkSMProxy> proxy;
    proxy.TakeReference(pxm->NewProxy(group, name));
    if (!proxy || !vtkSMViewProxy::SafeDownCast(proxy))
      {
      return NULL;
      }
    vtkNew<vtkSMParaViewPipelineController> controller;
    controller->InitializeProxy(proxy);
    vtkSMPropertyHelper(proxy, "ShowAnnotation", true).Set(1);
    proxy->UpdateVTKObjects();
    controller->RegisterViewProxy(proxy);
    return proxy;
    }

  void ShowAndRender(vtkSMSourceProxy* producer, vtkSMProxy* view, double time)
    {
    if (vtkSMViewProxy* vp = vtkSMViewProxy::SafeDownCast(view))
      {
      vtkSMPropertyHelper(vp, "ViewTime").Set(time);
      vp->UpdateVTKObjects();

      vtkNew<vtkSMParaViewPipelineControllerWithRendering> controller;
      vtkSMProxy* repr = controller->Show(producer, 0, vp);
      vtkSMPVRepresentationProxy::SetScalarColoring(repr, "data", vtkDataObject::POINT);
      vtkSMPVRepresentationProxy::RescaleTransferFunctionToDataRange(repr);
      vtkSMPVRepresentationProxy::SetScalarBarVisibility(repr, view, true);
      if (vtkSMRenderViewProxy* rv = vtkSMRenderViewProxy::SafeDownCast(vp))
        {
        rv->ResetCamera();
        }
      vp->StillRender();
      }
    }
}
#endif
