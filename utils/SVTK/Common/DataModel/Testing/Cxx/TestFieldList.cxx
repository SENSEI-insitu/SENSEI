#include "svtkDataSetAttributes.h"
#include "svtkDoubleArray.h"
#include "svtkIntArray.h"
#include "svtkNew.h"
#include "svtkSmartPointer.h"

#include <string>

namespace
{
template <typename T>
svtkSmartPointer<T> CreateArray(const char* aname, int num_comps, svtkIdType numTuples)
{
  auto array = svtkSmartPointer<T>::New();
  array->SetName(aname);
  array->SetNumberOfComponents(num_comps);
  array->SetNumberOfTuples(numTuples);
  array->FillValue(typename T::ValueType());
  return array;
}

#define EXPECT_THAT(v, m)                                                                          \
  if ((v) != (m))                                                                                  \
  {                                                                                                \
    cerr << "FAILED at line " << __LINE__ << ": \n     " << #v << " must match " << #m << endl;    \
    return EXIT_FAILURE;                                                                           \
  }
}

int TestFieldList(int, char*[])
{

  {
    // For arrays without names, ensure we are doing a order-dependent matching.
    // For attributes, the attribute flag is preserved if the same arrays is
    // consistently flagged as an attribute on all DSA instances.
    svtkNew<svtkDataSetAttributes> dsa0;
    dsa0->SetScalars(CreateArray<svtkDoubleArray>(nullptr, 1, 20));
    dsa0->AddArray(CreateArray<svtkDoubleArray>(nullptr, 2, 20));
    dsa0->SetVectors(CreateArray<svtkDoubleArray>(nullptr, 3, 20));
    EXPECT_THAT(dsa0->GetNumberOfArrays(), 3);

    svtkNew<svtkDataSetAttributes> dsa1;
    dsa1->SetScalars(CreateArray<svtkDoubleArray>(nullptr, 1, 20));
    dsa1->AddArray(CreateArray<svtkDoubleArray>(nullptr, 3, 20));
    dsa1->AddArray(CreateArray<svtkDoubleArray>(nullptr, 3, 20));
    EXPECT_THAT(dsa1->GetNumberOfArrays(), 3);

    svtkDataSetAttributes::FieldList fl;
    fl.InitializeFieldList(dsa0);
    fl.IntersectFieldList(dsa1);

    svtkNew<svtkDataSetAttributes> output;
    fl.CopyAllocate(output, svtkDataSetAttributes::COPYTUPLE, 0, 0);
    EXPECT_THAT(output->GetNumberOfArrays(), 2);
    EXPECT_THAT(output->GetArray(0)->GetNumberOfComponents(), 1);
    EXPECT_THAT(output->GetArray(1)->GetNumberOfComponents(), 3);
    EXPECT_THAT(output->GetVectors(), nullptr);
    EXPECT_THAT(output->GetScalars() != nullptr, true);

    fl.InitializeFieldList(dsa0);
    fl.UnionFieldList(dsa1);
    output->Initialize();
    fl.CopyAllocate(output, svtkDataSetAttributes::COPYTUPLE, 0, 0);
    EXPECT_THAT(output->GetNumberOfArrays(), 4);
    EXPECT_THAT(output->GetArray(0)->GetNumberOfComponents(), 1);
    EXPECT_THAT(output->GetArray(1)->GetNumberOfComponents(), 2);
    EXPECT_THAT(output->GetArray(2)->GetNumberOfComponents(), 3);
    EXPECT_THAT(output->GetArray(3)->GetNumberOfComponents(), 3);
    EXPECT_THAT(output->GetVectors(), nullptr);
    EXPECT_THAT(output->GetScalars() != nullptr, true);

    // just to increase coverage.
    fl.PrintSelf(cout, svtkIndent());
  }

  {
    // If inputs arrays with different names for attributes,
    // make sure output doesn't have either of the arrays flagged as attributes.
    svtkNew<svtkDataSetAttributes> dsa0;
    dsa0->SetScalars(CreateArray<svtkDoubleArray>("scalars", 1, 20));
    dsa0->AddArray(CreateArray<svtkDoubleArray>("vectors", 3, 20));
    dsa0->AddArray(CreateArray<svtkDoubleArray>("common", 1, 20));
    dsa0->AddArray(CreateArray<svtkDoubleArray>("uncommon0", 1, 20));

    svtkNew<svtkDataSetAttributes> dsa1;
    dsa1->AddArray(CreateArray<svtkDoubleArray>("scalars", 1, 20));
    dsa1->SetVectors(CreateArray<svtkDoubleArray>("vectors", 3, 20));
    dsa1->AddArray(CreateArray<svtkDoubleArray>("common", 1, 20));
    dsa0->AddArray(CreateArray<svtkDoubleArray>("uncommon1", 1, 20));

    svtkDataSetAttributes::FieldList fl;
    fl.InitializeFieldList(dsa0);
    fl.IntersectFieldList(dsa1);

    svtkNew<svtkDataSetAttributes> output;
    fl.CopyAllocate(output, svtkDataSetAttributes::COPYTUPLE, 0, 0);
    EXPECT_THAT(output->GetNumberOfArrays(), 3);
    EXPECT_THAT(output->GetArray("uncommon0"), nullptr);
    EXPECT_THAT(output->GetArray("uncommon1"), nullptr);
    EXPECT_THAT(output->GetScalars(), nullptr);
    EXPECT_THAT(output->GetVectors(), nullptr);
    EXPECT_THAT(output->GetArray("scalars") != nullptr, true);
    EXPECT_THAT(output->GetArray("vectors") != nullptr, true);

    fl.InitializeFieldList(dsa0);
    fl.UnionFieldList(dsa1);
    output->Initialize();
    fl.CopyAllocate(output, svtkDataSetAttributes::COPYTUPLE, 0, 0);
    EXPECT_THAT(output->GetNumberOfArrays(), 5);
    EXPECT_THAT(output->GetScalars(), nullptr);
    EXPECT_THAT(output->GetVectors(), nullptr);
  }

  return EXIT_SUCCESS;
}
