/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkInsituDataAdaptor.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkInsituDataAdaptor
// .SECTION Description
// vtkInsituDataAdaptor is an adaptor that is used to get vtkDataObject and
// vtkAbstractArray. Each simulation code provides an implementation of
// vtkInsituDataAdaptor which maps simulation data-structures to VTK data model
// on request.

#ifndef vtkInsituDataAdaptor_h
#define vtkInsituDataAdaptor_h

#include "vtkObjectBase.h"
#include "vtkSetGet.h"

class vtkInformation;
class vtkDataObject;
class vtkAbstractArray;
class vtkInsituDataAdaptor : public vtkObjectBase
{
public:
  vtkTypeMacro(vtkInsituDataAdaptor, vtkObjectBase);
  void PrintSelf(ostream& os, vtkIndent indent);

  /// @brief Return the data object with appropriate structure.
  ///
  /// This method will return a data object of the appropriate type. The data
  /// object can be a vtkDataSet subclass or a vtkCompositeDataSet subclass.
  /// If \c structure_only is set to true, then the geometry and topology
  /// information will not be populated. For data adaptors that produce a
  /// vtkCompositeDataSet subclass, passing \c structure_only will still produce
  /// appropriate composite data hierarchy.
  ///
  /// @param structure_only When set to true (default; false) the returned mesh
  /// may not have any geometry or topology information.
  virtual vtkDataObject* GetMesh(bool structure_only=false) = 0;

  /// @brief Adds the specified field array to the mesh.
  ///
  /// This method will add the requested array to the mesh, if available. If the
  /// array was already added to the mesh, this will not add it again. The mesh
  /// should not be expected to have geometry or topology information.
  ///
  /// @param association field association; one of
  /// vtkDataObject::FieldAssociations or vtkDataObject::AttributeTypes.
  /// @return true if array was added (or already added), false is request array
  /// is not available.
  virtual bool AddArray(vtkDataObject* mesh, int association, const char* arrayname) = 0;

  /// @brief Return the number of field arrays available.
  ///
  /// This method will return the number of field arrays available. For data
  /// adaptors producing composite datasets, this is a union of arrays available
  /// on all parts of the composite dataset.
  ///
  /// @param association field association; one of
  /// vtkDataObject::FieldAssociations or vtkDataObject::AttributeTypes.
  /// @return the number of arrays.
  virtual unsigned int GetNumberOfArrays(int association) = 0;

  /// @brief Return the name for a field array.
  ///
  /// This method will return the name for a field array given its index.
  ///
  /// @param association field association; one of
  /// vtkDataObject::FieldAssociations or vtkDataObject::AttributeTypes.
  /// @param index index for the array. Must be less than value returned
  /// GetNumberOfArrays().
  /// @return name of the array.
  virtual const char* GetArrayName(int association, unsigned int index) = 0;

  /// @brief Release data allocated for the current timestep.
  ///
  /// Releases the data allocated for the current timestep. This is expected to
  /// be called after each time iteration.
  virtual void ReleaseData() = 0;

  /// @brief Provides access to meta-data about the current data.
  ///
  /// This method can provide all meta-data necessary, including global extents,
  /// fields available, etc.
  vtkInformation* GetInformation() { return this->Information; }

  /// @brief Convenience method to get the time information.
  double GetDataTime() { return this->GetDataTime(this->Information); }

  /// @brief Convenience method to get the time information.
  static double GetDataTime(vtkInformation*);

  /// @brief Convenience methods to set the time information.
  void SetDataTime(double time) { this->SetDataTime(this->Information, time); }

  /// @brief Convenience methods to set the time information.
  static void SetDataTime(vtkInformation*, double time);

protected:
  vtkInsituDataAdaptor();
  ~vtkInsituDataAdaptor();

  vtkInformation* Information;
private:
  vtkInsituDataAdaptor(const vtkInsituDataAdaptor&); // Not implemented.
  void operator=(const vtkInsituDataAdaptor&); // Not implemented.
};

#endif
