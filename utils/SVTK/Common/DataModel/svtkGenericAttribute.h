/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGenericAttribute.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkGenericAttribute
 * @brief   abstract class defined API for attribute data
 *
 * svtkGenericAttribute is an abstract class that defines an API for attribute
 * data. Attribute data is data associated with the topology or geometry of
 * a dataset (i.e., points, cells, etc.). svtkGenericAttribute is part of the
 * adaptor framework (see GenericFiltering/README.html).
 *
 * svtkGenericAttribute provides a more general interface to attribute data
 * than its counterpart svtkDataArray (which assumes a linear, contiguous
 * array). It adopts an iterator interface, and allows attributes to be
 * associated with points, edges, faces, or edges.
 */

#ifndef svtkGenericAttribute_h
#define svtkGenericAttribute_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkGenericCellIterator;
class svtkGenericAdaptorCell;
class svtkGenericPointIterator;

enum
{
  svtkPointCentered,
  svtkCellCentered,
  svtkBoundaryCentered
};

class SVTKCOMMONDATAMODEL_EXPORT svtkGenericAttribute : public svtkObject
{
public:
  svtkTypeMacro(svtkGenericAttribute, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Name of the attribute. (e.g. "velocity")
   * \post result_may_not_exist: result!=0 || result==0
   */
  virtual const char* GetName() = 0;

  /**
   * Dimension of the attribute. (1 for scalar, 3 for velocity)
   * \post positive_result: result>=0
   * \post GetType()==SVTK_SCALARS implies result==1
   * \post (GetType()==SVTK_VECTORS||(GetType()==SVTK_NORMALS)||(GetType()==SVTK_TCOORDS) implies
   * result==3 \post GetType()==SVTK_TENSORS implies result==6
   */
  virtual int GetNumberOfComponents() = 0;

  /**
   * Is the attribute centered either on points, cells or boundaries?
   * \post valid_result: (result==svtkPointCentered)||(result==svtkCellCentered)
   */
  virtual int GetCentering() = 0;

  /**
   * Type of the attribute: scalar, vector, normal, texture coordinate, tensor
   * \post valid_result: (result==svtkDataSetAttributes::SCALARS)
   * ||(result==svtkDataSetAttributes::VECTORS)
   * ||(result==svtkDataSetAttributes::NORMALS)
   * ||(result==svtkDataSetAttributes::TCOORDS)
   * ||(result==svtkDataSetAttributes::TENSORS)
   */
  virtual int GetType() = 0;

  /**
   * Type of the components of the attribute: int, float, double
   * \post valid_result: (result==SVTK_BIT)           ||(result==SVTK_CHAR)
   * ||(result==SVTK_UNSIGNED_CHAR) ||(result==SVTK_SHORT)
   * ||(result==SVTK_UNSIGNED_SHORT)||(result==SVTK_INT)
   * ||(result==SVTK_UNSIGNED_INT)  ||(result==SVTK_LONG)
   * ||(result==SVTK_UNSIGNED_LONG) ||(result==SVTK_FLOAT)
   * ||(result==SVTK_DOUBLE)        ||(result==SVTK_ID_TYPE)
   */
  virtual int GetComponentType() = 0;

  /**
   * Number of tuples.
   * \post valid_result: result>=0
   */
  virtual svtkIdType GetSize() = 0;

  /**
   * Size in kibibytes (1024 bytes) taken by the attribute.
   */
  virtual unsigned long GetActualMemorySize() = 0;

  /**
   * Range of the attribute component `component'. If `component'==-1, it
   * returns the range of the magnitude (euclidean norm).
   * It returns double, even if GetType()==SVTK_INT.
   * NOT THREAD SAFE
   * \pre valid_component: (component>=-1)&&(component<GetNumberOfComponents())
   * \post result_exists: result!=0
   */
  virtual double* GetRange(int component = 0) = 0;

  /**
   * Range of the attribute component `component'. If `component'==-1, it
   * returns the range of the magnitude (euclidean norm).
   * THREAD SAFE
   * \pre valid_component: (component>=-1)&&(component<GetNumberOfComponents())
   */
  virtual void GetRange(int component, double range[2]) = 0;

  /**
   * Return the maximum euclidean norm for the tuples.
   * \post positive_result: result>=0
   */
  virtual double GetMaxNorm() = 0;

  /**
   * Attribute at all points of cell `c'.
   * \pre c_exists: c!=0
   * \pre c_valid: !c->IsAtEnd()
   * \post result_exists: result!=0
   * \post valid_result: sizeof(result)==GetNumberOfComponents()*c->GetCell()->GetNumberOfPoints()
   */
  virtual double* GetTuple(svtkGenericAdaptorCell* c) = 0;

  /**
   * Put attribute at all points of cell `c' in `tuple'.
   * \pre c_exists: c!=0
   * \pre c_valid: !c->IsAtEnd()
   * \pre tuple_exists: tuple!=0
   * \pre valid_tuple: sizeof(tuple)>=GetNumberOfComponents()*c->GetCell()->GetNumberOfPoints()
   */
  virtual void GetTuple(svtkGenericAdaptorCell* c, double* tuple) = 0;

  /**
   * Attribute at all points of cell `c'.
   * \pre c_exists: c!=0
   * \pre c_valid: !c->IsAtEnd()
   * \post result_exists: result!=0
   * \post valid_result: sizeof(result)==GetNumberOfComponents()*c->GetCell()->GetNumberOfPoints()
   */
  virtual double* GetTuple(svtkGenericCellIterator* c) = 0;

  /**
   * Put attribute at all points of cell `c' in `tuple'.
   * \pre c_exists: c!=0
   * \pre c_valid: !c->IsAtEnd()
   * \pre tuple_exists: tuple!=0
   * \pre valid_tuple: sizeof(tuple)>=GetNumberOfComponents()*c->GetCell()->GetNumberOfPoints()
   */
  virtual void GetTuple(svtkGenericCellIterator* c, double* tuple) = 0;

  /**
   * Value of the attribute at position `p'.
   * \pre p_exists: p!=0
   * \pre p_valid: !p->IsAtEnd()
   * \post result_exists: result!=0
   * \post valid_result_size: sizeof(result)==GetNumberOfComponents()
   */
  virtual double* GetTuple(svtkGenericPointIterator* p) = 0;

  /**
   * Put the value of the attribute at position `p' into `tuple'.
   * \pre p_exists: p!=0
   * \pre p_valid: !p->IsAtEnd()
   * \pre tuple_exists: tuple!=0
   * \pre valid_tuple_size: sizeof(tuple)>=GetNumberOfComponents()
   */
  virtual void GetTuple(svtkGenericPointIterator* p, double* tuple) = 0;

  /**
   * Put component `i' of the attribute at all points of cell `c' in `values'.
   * \pre valid_component: (i>=0) && (i<GetNumberOfComponents())
   * \pre c_exists: c!=0
   * \pre c_valid: !c->IsAtEnd()
   * \pre values_exist: values!=0
   * \pre valid_values: sizeof(values)>=c->GetCell()->GetNumberOfPoints()
   */
  virtual void GetComponent(int i, svtkGenericCellIterator* c, double* values) = 0;

  /**
   * Value of the component `i' of the attribute at position `p'.
   * \pre valid_component: (i>=0) && (i<GetNumberOfComponents())
   * \pre p_exists: p!=0
   * \pre p_valid: !p->IsAtEnd()
   */
  virtual double GetComponent(int i, svtkGenericPointIterator* p) = 0;

  /**
   * Recursive duplication of `other' in `this'.
   * \pre other_exists: other!=0
   * \pre not_self: other!=this
   */
  virtual void DeepCopy(svtkGenericAttribute* other) = 0;

  /**
   * Update `this' using fields of `other'.
   * \pre other_exists: other!=0
   * \pre not_self: other!=this
   */
  virtual void ShallowCopy(svtkGenericAttribute* other) = 0;

protected:
  svtkGenericAttribute();
  ~svtkGenericAttribute() override;

private:
  svtkGenericAttribute(const svtkGenericAttribute&) = delete;
  void operator=(const svtkGenericAttribute&) = delete;
};

#endif
