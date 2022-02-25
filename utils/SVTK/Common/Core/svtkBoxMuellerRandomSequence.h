/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBoxMuellerRandomSequence.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
=========================================================================*/
/**
 * @class   svtkBoxMuellerRandomSequence
 * @brief   Gaussian sequence of pseudo random numbers implemented with the Box-Mueller transform
 *
 * svtkGaussianRandomSequence is a sequence of pseudo random numbers
 * distributed according to the Gaussian/normal distribution (mean=0 and
 * standard deviation=1).
 *
 * It based is calculation from a uniformly distributed pseudo random sequence.
 * The initial sequence is a svtkMinimalStandardRandomSequence.
 */

#ifndef svtkBoxMuellerRandomSequence_h
#define svtkBoxMuellerRandomSequence_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkGaussianRandomSequence.h"

class SVTKCOMMONCORE_EXPORT svtkBoxMuellerRandomSequence : public svtkGaussianRandomSequence
{
public:
  //@{
  /**
   * Standard methods for instantiation, type information, and printing.
   */
  static svtkBoxMuellerRandomSequence* New();
  svtkTypeMacro(svtkBoxMuellerRandomSequence, svtkGaussianRandomSequence);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  /**
   * Satisfy general API of svtkRandomSequence superclass. Initialize the
   * sequence with a seed.
   */
  void Initialize(svtkTypeUInt32 svtkNotUsed(seed)) override {}

  /**
   * Current value.
   */
  double GetValue() override;

  /**
   * Move to the next number in the random sequence.
   */
  void Next() override;

  /**
   * Return the uniformly distributed sequence of random numbers.
   */
  svtkRandomSequence* GetUniformSequence();

  /**
   * Set the uniformly distributed sequence of random numbers.
   * Default is a .
   */
  void SetUniformSequence(svtkRandomSequence* uniformSequence);

protected:
  svtkBoxMuellerRandomSequence();
  ~svtkBoxMuellerRandomSequence() override;

  svtkRandomSequence* UniformSequence;
  double Value;

private:
  svtkBoxMuellerRandomSequence(const svtkBoxMuellerRandomSequence&) = delete;
  void operator=(const svtkBoxMuellerRandomSequence&) = delete;
};

#endif // #ifndef svtkBoxMuellerRandomSequence_h
