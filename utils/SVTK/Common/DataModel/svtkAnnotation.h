/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAnnotation.h

-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkAnnotation
 * @brief   Stores a collection of annotation artifacts.
 *
 *
 * svtkAnnotation is a collection of annotation properties along with
 * an associated selection indicating the portion of data the annotation
 * refers to.
 *
 * @par Thanks:
 * Timothy M. Shead (tshead@sandia.gov) at Sandia National Laboratories
 * contributed code to this class.
 */

#ifndef svtkAnnotation_h
#define svtkAnnotation_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDataObject.h"

class svtkInformationStringKey;
class svtkInformationDoubleVectorKey;
class svtkInformationIntegerVectorKey;
class svtkInformationDataObjectKey;
class svtkSelection;

class SVTKCOMMONDATAMODEL_EXPORT svtkAnnotation : public svtkDataObject
{
public:
  svtkTypeMacro(svtkAnnotation, svtkDataObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  static svtkAnnotation* New();

  //@{
  /**
   * The selection to which this set of annotations will apply.
   */
  svtkGetObjectMacro(Selection, svtkSelection);
  virtual void SetSelection(svtkSelection* selection);
  //@}

  //@{
  /**
   * Retrieve a svtkAnnotation stored inside an information object.
   */
  static svtkAnnotation* GetData(svtkInformation* info);
  static svtkAnnotation* GetData(svtkInformationVector* v, int i = 0);
  //@}

  /**
   * The label for this annotation.
   */
  static svtkInformationStringKey* LABEL();

  /**
   * The color for this annotation.
   * This is stored as an RGB triple with values between 0 and 1.
   */
  static svtkInformationDoubleVectorKey* COLOR();

  /**
   * The color for this annotation.
   * This is stored as a value between 0 and 1.
   */
  static svtkInformationDoubleKey* OPACITY();

  /**
   * An icon index for this annotation.
   */
  static svtkInformationIntegerKey* ICON_INDEX();

  /**
   * Whether or not this annotation is enabled.
   * A value of 1 means enabled, 0 disabled.
   */
  static svtkInformationIntegerKey* ENABLE();

  /**
   * Whether or not this annotation is visible.
   */
  static svtkInformationIntegerKey* HIDE();

  /**
   * Associate a svtkDataObject with this annotation
   */
  static svtkInformationDataObjectKey* DATA();

  /**
   * Initialize the annotation to an empty state.
   */
  void Initialize() override;

  /**
   * Make this annotation have the same properties and have
   * the same selection of another annotation.
   */
  void ShallowCopy(svtkDataObject* other) override;

  /**
   * Make this annotation have the same properties and have
   * a copy of the selection of another annotation.
   */
  void DeepCopy(svtkDataObject* other) override;

  /**
   * Get the modified time of this object.
   */
  svtkMTimeType GetMTime() override;

protected:
  svtkAnnotation();
  ~svtkAnnotation() override;

  svtkSelection* Selection;

private:
  svtkAnnotation(const svtkAnnotation&) = delete;
  void operator=(const svtkAnnotation&) = delete;
};

#endif
