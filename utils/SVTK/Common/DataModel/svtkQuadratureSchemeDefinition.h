/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkQuadratureSchemeDefinition.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkQuadratureSchemeDefinition
 *
 * An Elemental data type that holds a definition of a
 * numerical quadrature scheme. The definition contains
 * the requisite information to interpolate to the so called
 * quadrature points of the specific scheme. namely:
 *
 * <pre>
 * 1)
 * A matrix of shape function weights(shape functions evaluated
 * at parametric coordinates of the quadrature points).
 *
 * 2)
 * The number of quadrature points and cell nodes. These parameters
 * size the matrix, and allow for convinent evaluation by users
 * of the definition.
 * </pre>
 */

#ifndef svtkQuadratureSchemeDefinition_h
#define svtkQuadratureSchemeDefinition_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkInformationQuadratureSchemeDefinitionVectorKey;
class svtkInformationStringKey;
class svtkXMLDataElement;

class SVTKCOMMONDATAMODEL_EXPORT svtkQuadratureSchemeDefinition : public svtkObject
{
public:
  // svtk stuff
  svtkTypeMacro(svtkQuadratureSchemeDefinition, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  static svtkInformationQuadratureSchemeDefinitionVectorKey* DICTIONARY();
  static svtkInformationStringKey* QUADRATURE_OFFSET_ARRAY_NAME();

  /**
   * New object in an unsuable state. You'll have to call
   * "Initilaize" to get the definition in to a usable state.
   */
  static svtkQuadratureSchemeDefinition* New();

  /**
   * Deep copy.
   */
  int DeepCopy(const svtkQuadratureSchemeDefinition* other);

  /**
   * Put the object into an XML representation. The element
   * passed in is assumed to be empty.
   */
  int SaveState(svtkXMLDataElement* e);
  /**
   * Restore the object from an XML representation.
   */
  int RestoreState(svtkXMLDataElement* e);

  /**
   * Release all allocated resources and set the
   * object to an uninitialized state.
   */
  void Clear();

  /**
   * Initialize the object allocating resources as needed.
   */
  void Initialize(
    int cellType, int numberOfNodes, int numberOfQuadraturePoints, double* shapeFunctionWeights);
  /**
   * Initialize the object allocating resources as needed.
   */
  void Initialize(int cellType, int numberOfNodes, int numberOfQuadraturePoints,
    double* shapeFunctionWeights, double* quadratureWeights);

  /**
   * Access the SVTK cell type id.
   */
  int GetCellType() const { return this->CellType; }
  /**
   * Access to an alternative key.
   */
  int GetQuadratureKey() const { return this->QuadratureKey; }
  /**
   * Get the number of nodes associated with the interpolation.
   */
  int GetNumberOfNodes() const { return this->NumberOfNodes; }
  /**
   * Get the number of quadrature points associated with the scheme.
   */
  int GetNumberOfQuadraturePoints() const { return this->NumberOfQuadraturePoints; }
  /**
   * Get the array of shape function weights. Shape function weights are
   * the shape functions evaluated at the quadrature points. There are
   * "NumberOfNodes" weights for each quadrature point.
   */
  const double* GetShapeFunctionWeights() const { return this->ShapeFunctionWeights; }
  //@{
  /**
   * Get the array of shape function weights associated with a
   * single quadrature point.
   */
  const double* GetShapeFunctionWeights(int quadraturePointId) const
  {
    int idx = quadraturePointId * this->NumberOfNodes;
    return this->ShapeFunctionWeights + idx;
  }
  //@}
  /**
   * Access to the quadrature weights.
   */
  const double* GetQuadratureWeights() const { return this->QuadratureWeights; }

protected:
  svtkQuadratureSchemeDefinition();
  ~svtkQuadratureSchemeDefinition() override;

private:
  /**
   * Allocate/De-allocate resources that will be used by the definition.
   * This must be called after Set*. Caller's responsibility.
   */
  void ReleaseResources();
  /**
   * Allocate resources according to the objects
   * current internal state.
   */
  int SecureResources();
  /**
   * Initialize the shape function weights definition.
   * Must call SecureResources prior.
   */
  void SetShapeFunctionWeights(const double* W);
  /**
   * Initialize the shape function weights definition.
   * Must call SecureResources prior.
   */
  void SetQuadratureWeights(const double* W);

  //
  svtkQuadratureSchemeDefinition(const svtkQuadratureSchemeDefinition&) = delete;
  void operator=(const svtkQuadratureSchemeDefinition&) = delete;
  friend ostream& operator<<(ostream& s, const svtkQuadratureSchemeDefinition& d);
  friend istream& operator>>(istream& s, svtkQuadratureSchemeDefinition& d);
  //
  int CellType;
  int QuadratureKey;
  int NumberOfNodes;
  int NumberOfQuadraturePoints;
  double* ShapeFunctionWeights;
  double* QuadratureWeights;
};

#endif
