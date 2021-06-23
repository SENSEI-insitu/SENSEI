/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkRandomSequence.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
=========================================================================*/
/**
 * @class   svtkRandomSequence
 * @brief   Generate a sequence of random numbers.
 *
 * svtkRandomSequence defines the interface of any sequence of random numbers.
 *
 * At this level of abstraction, there is no assumption about the distribution
 * of the numbers or about the quality of the sequence of numbers to be
 * statistically independent. There is no assumption about the range of values.
 *
 * To the question about why a random "sequence" class instead of a random
 * "generator" class or to a random "number" class?, see the OOSC book:
 * "Object-Oriented Software Construction", 2nd Edition, by Bertrand Meyer.
 * chapter 23, "Principles of class design", "Pseudo-random number
 * generators: a design exercise", page 754--755.
 */

#ifndef svtkRandomSequence_h
#define svtkRandomSequence_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

class SVTKCOMMONCORE_EXPORT svtkRandomSequence : public svtkObject
{
public:
  //@{
  /**
   * Standard methods for type information and printing.
   */
  svtkTypeMacro(svtkRandomSequence, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  /**
   * Initialize the sequence with a seed.
   */
  virtual void Initialize(svtkTypeUInt32 seed) = 0;

  /**
   * Return the current value.
   */
  virtual double GetValue() = 0;

  /**
   * Move to the next number in the random sequence.
   */
  virtual void Next() = 0;

protected:
  svtkRandomSequence();
  ~svtkRandomSequence() override;

private:
  svtkRandomSequence(const svtkRandomSequence&) = delete;
  void operator=(const svtkRandomSequence&) = delete;
};

#endif // #ifndef svtkRandomSequence_h
