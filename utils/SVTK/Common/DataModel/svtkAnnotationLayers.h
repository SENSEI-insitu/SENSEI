/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAnnotationLayers.h

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
 * @class   svtkAnnotationLayers
 * @brief   Stores a ordered collection of annotation sets
 *
 *
 * svtkAnnotationLayers stores a vector of annotation layers. Each layer
 * may contain any number of svtkAnnotation objects. The ordering of the
 * layers introduces a prioritization of annotations. Annotations in
 * higher layers may obscure annotations in lower layers.
 */

#ifndef svtkAnnotationLayers_h
#define svtkAnnotationLayers_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkDataObject.h"

class svtkAnnotation;
class svtkSelection;

class SVTKCOMMONDATAMODEL_EXPORT svtkAnnotationLayers : public svtkDataObject
{
public:
  svtkTypeMacro(svtkAnnotationLayers, svtkDataObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  static svtkAnnotationLayers* New();

  //@{
  /**
   * The current annotation associated with this annotation link.
   */
  virtual void SetCurrentAnnotation(svtkAnnotation* ann);
  svtkGetObjectMacro(CurrentAnnotation, svtkAnnotation);
  //@}

  //@{
  /**
   * The current selection associated with this annotation link.
   * This is simply the selection contained in the current annotation.
   */
  virtual void SetCurrentSelection(svtkSelection* sel);
  virtual svtkSelection* GetCurrentSelection();
  //@}

  /**
   * The number of annotations in a specific layer.
   */
  unsigned int GetNumberOfAnnotations();

  /**
   * Retrieve an annotation from a layer.
   */
  svtkAnnotation* GetAnnotation(unsigned int idx);

  /**
   * Add an annotation to a layer.
   */
  void AddAnnotation(svtkAnnotation* ann);

  /**
   * Remove an annotation from a layer.
   */
  void RemoveAnnotation(svtkAnnotation* ann);

  /**
   * Initialize the data structure to an empty state.
   */
  void Initialize() override;

  /**
   * Copy data from another data object into this one
   * which references the same member annotations.
   */
  void ShallowCopy(svtkDataObject* other) override;

  /**
   * Copy data from another data object into this one,
   * performing a deep copy of member annotations.
   */
  void DeepCopy(svtkDataObject* other) override;

  //@{
  /**
   * Retrieve a svtkAnnotationLayers stored inside an information object.
   */
  static svtkAnnotationLayers* GetData(svtkInformation* info);
  static svtkAnnotationLayers* GetData(svtkInformationVector* v, int i = 0);
  //@}

  /**
   * The modified time for this object.
   */
  svtkMTimeType GetMTime() override;

protected:
  svtkAnnotationLayers();
  ~svtkAnnotationLayers() override;

  class Internals;
  Internals* Implementation;
  svtkAnnotation* CurrentAnnotation;

private:
  svtkAnnotationLayers(const svtkAnnotationLayers&) = delete;
  void operator=(const svtkAnnotationLayers&) = delete;
};

#endif
