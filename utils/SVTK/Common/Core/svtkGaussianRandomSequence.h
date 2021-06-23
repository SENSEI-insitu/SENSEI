/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGaussianRandomSequence.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
=========================================================================*/
/**
 * @class   svtkGaussianRandomSequence
 * @brief   Gaussian sequence of pseudo random numbers
 *
 * svtkGaussianRandomSequence is a sequence of pseudo random numbers
 * distributed according to the Gaussian/normal distribution (mean=0 and
 * standard deviation=1)
 *
 * This is just an interface.
 */

#ifndef svtkGaussianRandomSequence_h
#define svtkGaussianRandomSequence_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkRandomSequence.h"

class SVTKCOMMONCORE_EXPORT svtkGaussianRandomSequence : public svtkRandomSequence
{
public:
  //@{
  /**
   * Standard methods for type information and printing.
   */
  svtkTypeMacro(svtkGaussianRandomSequence, svtkRandomSequence);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  /**
   * Convenient method to return a value given the mean and standard deviation
   * of the Gaussian distribution from the Gaussian distribution of mean=0
   * and standard deviation=1.0. There is an initial implementation that can
   * be overridden by a subclass.
   */
  virtual double GetScaledValue(double mean, double standardDeviation);

protected:
  svtkGaussianRandomSequence();
  ~svtkGaussianRandomSequence() override;

private:
  svtkGaussianRandomSequence(const svtkGaussianRandomSequence&) = delete;
  void operator=(const svtkGaussianRandomSequence&) = delete;
};

#endif // #ifndef svtkGaussianRandomSequence_h
