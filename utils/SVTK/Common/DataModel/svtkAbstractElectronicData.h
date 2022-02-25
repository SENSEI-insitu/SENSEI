/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAbstractElectronicData.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkAbstractElectronicData
 * @brief   Provides access to and storage of
 * chemical electronic data
 *
 */

#ifndef svtkAbstractElectronicData_h
#define svtkAbstractElectronicData_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDataObject.h"

class svtkImageData;

class SVTKCOMMONDATAMODEL_EXPORT svtkAbstractElectronicData : public svtkDataObject
{
public:
  svtkTypeMacro(svtkAbstractElectronicData, svtkDataObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Returns the number of molecular orbitals available.
   */
  virtual svtkIdType GetNumberOfMOs() = 0;

  /**
   * Returns the number of electrons in the molecule.
   */
  virtual svtkIdType GetNumberOfElectrons() = 0;

  /**
   * Returns the svtkImageData for the requested molecular orbital.
   */
  virtual svtkImageData* GetMO(svtkIdType orbitalNumber) = 0;

  /**
   * Returns svtkImageData for the molecule's electron density. The data
   * will be calculated when first requested, and cached for later requests.
   */
  virtual svtkImageData* GetElectronDensity() = 0;

  /**
   * Returns svtkImageData for the Highest Occupied Molecular Orbital.
   */
  svtkImageData* GetHOMO() { return this->GetMO(this->GetHOMOOrbitalNumber()); }

  /**
   * Returns svtkImageData for the Lowest Unoccupied Molecular Orbital.
   */
  svtkImageData* GetLUMO() { return this->GetMO(this->GetLUMOOrbitalNumber()); }

  // Description:
  // Returns the orbital number of the Highest Occupied Molecular Orbital.
  svtkIdType GetHOMOOrbitalNumber()
  {
    return static_cast<svtkIdType>((this->GetNumberOfElectrons() / 2) - 1);
  }

  // Description:
  // Returns the orbital number of the Lowest Unoccupied Molecular Orbital.
  svtkIdType GetLUMOOrbitalNumber()
  {
    return static_cast<svtkIdType>(this->GetNumberOfElectrons() / 2);
  }

  /**
   * Returns true if the given orbital number is the Highest Occupied
   * Molecular Orbital, false otherwise.
   */
  bool IsHOMO(svtkIdType orbitalNumber) { return (orbitalNumber == this->GetHOMOOrbitalNumber()); }

  /**
   * Returns true if the given orbital number is the Lowest Unoccupied
   * Molecular Orbital, false otherwise.
   */
  bool IsLUMO(svtkIdType orbitalNumber) { return (orbitalNumber == this->GetLUMOOrbitalNumber()); }

  /**
   * Deep copies the data object into this.
   */
  void DeepCopy(svtkDataObject* obj) override;

  //@{
  /**
   * Get the padding between the molecule and the cube boundaries. This is
   * used to determine the dataset's bounds.
   */
  svtkGetMacro(Padding, double);
  //@}

protected:
  svtkAbstractElectronicData();
  ~svtkAbstractElectronicData() override;

  double Padding;

private:
  svtkAbstractElectronicData(const svtkAbstractElectronicData&) = delete;
  void operator=(const svtkAbstractElectronicData&) = delete;
};

#endif
