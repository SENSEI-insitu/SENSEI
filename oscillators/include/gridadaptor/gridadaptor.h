#ifndef GRIDADAPTOR_H
#define GRIDADAPTOR_H


#include <grid/grid.h>
#include <diy/master.hpp>

#include <vtkAOSDataArrayTemplate.h>
#include <vtkDataObject.h>
#include <vtkImageData.h>
#include <vtkInsituDataAdaptor.h>
#include <vtkPointData.h>
namespace gridadaptor
{

/// vtkInsituDataAdaptor specialization for grid::GridRef.
template <class C, unsigned D>
class GridAdaptor : public vtkInsituDataAdaptor
{
public:
  typedef grid::GridRef<C, D> GridRef;
  typedef typename GridRef::Vertex Vertex;
  typedef diy::DiscreteBounds Bounds;

public:
  static GridAdaptor* New() { return new GridAdaptor(); }
  vtkTypeMacro(GridAdaptor, vtkInsituDataAdaptor);

  void Initialize(int gid, const Bounds& bounds, const Vertex& domain_shape)
    {
    this->GridBounds = bounds;
    this->DomainShape = domain_shape;
    this->GID = gid;
    }

  void SetGrid(GridRef grid)
    {
    this->Grid = grid;
    }

  /// Return the topology/geometry for the simulation grid.
  virtual vtkDataObject* GetMesh()
    {
    if (!this->Mesh)
      {
      this->Mesh = vtkSmartPointer<vtkImageData>::New();
      assert(this->Grid.c_order());
      this->Mesh->SetExtent(
        this->GridBounds.min[0], this->GridBounds.max[0],
        this->GridBounds.min[1], this->GridBounds.max[1],
        this->GridBounds.min[2], this->GridBounds.max[2]);
      /// XXX: This is for the time-being. We need to add API to
      /// vtkInsituDataAdaptor to add arrays to a mesh passed in as the argument.
      /// Until we do that, we assume GetMesh() will ask for all provided
      /// arrays.
      this->Mesh->GetPointData()->SetScalars(
        static_cast<vtkDataArray*>(this->GetArray(vtkDataObject::FIELD_ASSOCIATION_POINTS, "data")));
      }
    return this->Mesh;
    }

  /// Return an array.
  virtual vtkAbstractArray* GetArray(int association, const char* name)
    {
    if (association != vtkDataObject::FIELD_ASSOCIATION_POINTS ||
        name == NULL ||
        strcmp(name, "data") != 0)
      {
      return NULL;
      }
    if (!this->DataArray)
      {
      this->DataArray = vtkSmartPointer<vtkAOSDataArrayTemplate<C> > ::New();
      this->DataArray->SetName("data");
      this->DataArray->SetArray(this->Grid.data(), this->Grid.size(), 1);
      }
    return this->DataArray;
    }

  /// Return number of arrays.
  virtual unsigned int GetNumberOfArrays(int association)
    { return 1; }

  /// Returns an arrays name given the index.
  virtual const char* GetArrayName(int association, unsigned int index)
    { return index==0? "data" : NULL; }

  /// Release all data.
  virtual void ReleaseData()
    {
    this->DataArray = NULL;
    this->Mesh = NULL;
    this->SetGrid(GridRef(NULL, Vertex()));
    }

protected:
  GridAdaptor() : Grid(NULL, Vertex()) {}
  virtual ~GridAdaptor() {}

  int GID;
  GridRef Grid;
  Bounds GridBounds;
  Vertex DomainShape;
  vtkSmartPointer<vtkAOSDataArrayTemplate<C> > DataArray;
  vtkSmartPointer<vtkImageData> Mesh;
private:
  GridAdaptor(const GridAdaptor&); // not implemented.
  void operator=(const GridAdaptor&); // not implemented.
};


}
#endif
