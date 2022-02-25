/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImplicitFunction.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkImplicitFunction.h"

#include "svtkAbstractTransform.h"
#include "svtkArrayDispatch.h"
#include "svtkDataArrayRange.h"
#include "svtkMath.h"
#include "svtkTransform.h"

#include <algorithm>

svtkCxxSetObjectMacro(svtkImplicitFunction, Transform, svtkAbstractTransform);

svtkImplicitFunction::svtkImplicitFunction()
{
  this->Transform = nullptr;
}

svtkImplicitFunction::~svtkImplicitFunction()
{
  // static_cast needed since otherwise the
  // call to SetTransform becomes ambiguous
  this->SetTransform(static_cast<svtkAbstractTransform*>(nullptr));
}

namespace
{

template <class Func>
struct FunctionWorker
{
  Func F;
  FunctionWorker(Func f)
    : F(f)
  {
  }
  template <typename SourceArray, typename DestinationArray>
  void operator()(SourceArray* input, DestinationArray* output)
  {
    svtkIdType numTuples = input->GetNumberOfTuples();
    output->SetNumberOfTuples(numTuples);

    const auto srcTuples = svtk::DataArrayTupleRange<3>(input);
    auto dstValues = svtk::DataArrayValueRange<1>(output);

    using DstValueT = typename decltype(dstValues)::ValueType;

    double in[3];
    auto destIter = dstValues.begin();
    for (auto tuple = srcTuples.cbegin(); tuple != srcTuples.cend(); ++tuple, ++destIter)
    {
      in[0] = static_cast<double>((*tuple)[0]);
      in[1] = static_cast<double>((*tuple)[1]);
      in[2] = static_cast<double>((*tuple)[2]);
      *destIter = static_cast<DstValueT>(this->F(in));
    }
  }
};

class SimpleFunction
{
public:
  SimpleFunction(svtkImplicitFunction* function)
    : Function(function)
  {
  }
  double operator()(double in[3]) { return this->Function->EvaluateFunction(in); }

private:
  svtkImplicitFunction* Function;
};

class TransformFunction
{
public:
  TransformFunction(svtkImplicitFunction* function, svtkAbstractTransform* transform)
    : Function(function)
    , Transform(transform)
  {
  }
  double operator()(double in[3])
  {
    Transform->TransformPoint(in, in);
    return this->Function->EvaluateFunction(in);
  }

private:
  svtkImplicitFunction* Function;
  svtkAbstractTransform* Transform;
};

} // end anon namespace

void svtkImplicitFunction::FunctionValue(svtkDataArray* input, svtkDataArray* output)
{
  if (!this->Transform)
  {
    this->EvaluateFunction(input, output);
  }
  else // pass point through transform
  {
    FunctionWorker<TransformFunction> worker(TransformFunction(this, this->Transform));
    typedef svtkTypeList::Create<float, double> InputTypes;
    typedef svtkTypeList::Create<float, double> OutputTypes;
    typedef svtkArrayDispatch::Dispatch2ByValueType<InputTypes, OutputTypes> MyDispatch;
    if (!MyDispatch::Execute(input, output, worker))
    {
      worker(input, output); // Use svtkDataArray API if dispatch fails.
    }
  }
}

void svtkImplicitFunction::EvaluateFunction(svtkDataArray* input, svtkDataArray* output)
{

  // defend against uninitialized output datasets.
  output->SetNumberOfComponents(1);
  output->SetNumberOfTuples(input->GetNumberOfTuples());

  FunctionWorker<SimpleFunction> worker(SimpleFunction(this));
  typedef svtkTypeList::Create<float, double> InputTypes;
  typedef svtkTypeList::Create<float, double> OutputTypes;
  typedef svtkArrayDispatch::Dispatch2ByValueType<InputTypes, OutputTypes> MyDispatch;
  if (!MyDispatch::Execute(input, output, worker))
  {
    worker(input, output); // Use svtkDataArray API if dispatch fails.
  }
}

// Evaluate function at position x-y-z and return value. Point x[3] is
// transformed through transform (if provided).
double svtkImplicitFunction::FunctionValue(const double x[3])
{
  if (!this->Transform)
  {
    return this->EvaluateFunction(const_cast<double*>(x));
  }
  else // pass point through transform
  {
    double pt[3];
    this->Transform->TransformPoint(x, pt);
    return this->EvaluateFunction(pt);
  }

  /* Return negative if determinant of Jacobian matrix is negative,
     i.e. if the transformation has a flip.  This is more 'correct'
     than the above behaviour, because it turns the implicit surface
     inside-out in the same way that polygonal surfaces are turned
     inside-out by a flip.  It takes up too many valuable CPU cycles
     to check the determinant on every function evaluation, though.
  {
    double pt[3];
    double A[3][3];
    this->Transform->Update();
    this->Transform->InternalTransformDerivative(x,pt,A);
    double val = this->EvaluateFunction((double *)pt);

    if (svtkMath::Determinant3x3(A) < 0)
      {
      return -val;
      }
    else
      {
      return +val;
      }
    }
  */
}

// Evaluate function gradient at position x-y-z and pass back vector. Point
// x[3] is transformed through transform (if provided).
void svtkImplicitFunction::FunctionGradient(const double x[3], double g[3])
{
  if (!this->Transform)
  {
    this->EvaluateGradient(const_cast<double*>(x), g);
  }
  else // pass point through transform
  {
    double pt[3];
    double A[3][3];
    this->Transform->Update();
    this->Transform->InternalTransformDerivative(x, pt, A);
    this->EvaluateGradient(static_cast<double*>(pt), g);

    // The gradient must be transformed using the same math as is
    // use for a normal to a surface: it must be multiplied by the
    // inverse of the transposed inverse of the Jacobian matrix of
    // the transform, which is just the transpose of the Jacobian.
    svtkMath::Transpose3x3(A, A);
    svtkMath::Multiply3x3(A, g, g);

    /* If the determinant of the Jacobian matrix is negative,
       then the gradient points in the opposite direction.  This
       behaviour is actually incorrect, but is necessary to
       balance the incorrect behaviour of FunctionValue.  Otherwise,
       if you feed certain SVTK filters a transform with a flip
       the gradient will point in the wrong direction and they
       will never converge to a result */

    if (svtkMath::Determinant3x3(A) < 0)
    {
      g[0] = -g[0];
      g[1] = -g[1];
      g[2] = -g[2];
    }
  }
}

// Overload standard modified time function. If Transform is modified,
// then this object is modified as well.
svtkMTimeType svtkImplicitFunction::GetMTime()
{
  svtkMTimeType mTime = this->svtkObject::GetMTime();
  svtkMTimeType TransformMTime;

  if (this->Transform != nullptr)
  {
    TransformMTime = this->Transform->GetMTime();
    mTime = (TransformMTime > mTime ? TransformMTime : mTime);
  }

  return mTime;
}

void svtkImplicitFunction::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  if (this->Transform)
  {
    os << indent << "Transform:\n";
    this->Transform->PrintSelf(os, indent.GetNextIndent());
  }
  else
  {
    os << indent << "Transform: (None)\n";
  }
}

void svtkImplicitFunction::SetTransform(const double elements[16])
{
  svtkTransform* transform = svtkTransform::New();
  transform->SetMatrix(elements);
  this->SetTransform(transform);
  transform->Delete();
}
