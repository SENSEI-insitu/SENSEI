#include "svtkInformation.h"
#include "svtkInformationDoubleKey.h"
#include "svtkInformationDoubleVectorKey.h"
#include "svtkInformationStringKey.h"
#include "svtkInformationStringVectorKey.h"
#include "svtkInformationVariantKey.h"
#include "svtkInformationVariantVectorKey.h"
#include "svtkMath.h"
#include "svtkNew.h"
#include "svtkStdString.h"
#include "svtkVariant.h"

template <typename T, typename V>
int UnitTestScalarValueKey(svtkInformation* info, T* key, const V& val)
{
  key->Set(info, val);
  int ok_setget = (val == key->Get(info));
  if (!ok_setget)
  {
    cerr << "Set + Get not reflexive.\n";
  }

  svtkNew<svtkInformation> shinyNew;
  key->ShallowCopy(info, shinyNew);
  int ok_copyget = (val == key->Get(shinyNew));
  if (!ok_copyget)
  {
    cerr << "Copy + Get not reflexive.\n";
  }

  return ok_setget & ok_copyget;
}

template <typename T, typename V>
int UnitTestVectorValueKey(svtkInformation* info, T* key, const V& val)
{
  key->Set(info, const_cast<V*>(&val), 1);
  int ok_setget = (val == key->Get(info, 0));
  if (!ok_setget)
  {
    cerr << "Set + get not reflexive.\n";
  }
  int ok_setgetcomp = (val == *key->Get(info));
  if (!ok_setgetcomp)
  {
    cerr << "Set + component-wise-get not reflexive.\n";
  }

  svtkNew<svtkInformation> shinyNew;
  key->ShallowCopy(info, shinyNew);
  int ok_copyget = (val == *key->Get(shinyNew));
  if (!ok_copyget)
  {
    cerr << "Copy + get not reflexive.\n";
  }

  int ok_length = (key->Length(info) == 1);
  if (!ok_length)
  {
    cerr << "Length was " << key->Length(info) << " not 1.\n";
  }
  key->Append(info, val);
  int ok_appendedlength = (key->Length(info) == 2);
  if (!ok_appendedlength)
  {
    cerr << "Appended length was " << key->Length(info) << " not 2.\n";
  }

  return ok_setget && ok_setgetcomp && ok_copyget && ok_length && ok_appendedlength;
}

// === String adaptations of tests above ===
// Note these are not specializations.

int UnitTestScalarValueKey(
  svtkInformation* info, svtkInformationStringKey* key, const svtkStdString& val)
{
  key->Set(info, val.c_str());
  int ok_setget = (val == key->Get(info));
  if (!ok_setget)
  {
    cerr << "Set + Get not reflexive.\n";
  }

  svtkNew<svtkInformation> shinyNew;
  key->ShallowCopy(info, shinyNew);
  int ok_copyget = (val == key->Get(shinyNew));
  if (!ok_copyget)
  {
    cerr << "Copy + Get not reflexive.\n";
  }

  return ok_setget & ok_copyget;
}

int UnitTestVectorValueKey(
  svtkInformation* info, svtkInformationStringVectorKey* key, const svtkStdString& val)
{
  key->Set(info, val.c_str(), 0);
  int ok_setgetcomp = (val == key->Get(info, 0));
  if (!ok_setgetcomp)
  {
    cerr << "Set + get not reflexive.\n";
  }

  svtkNew<svtkInformation> shinyNew;
  key->ShallowCopy(info, shinyNew);
  int ok_copyget = (val == key->Get(shinyNew, 0));
  if (!ok_copyget)
  {
    cerr << "Copy + get not reflexive.\n";
  }

  int ok_length = (key->Length(info) == 1);
  if (!ok_length)
  {
    cerr << "Length was " << key->Length(info) << " not 1.\n";
  }
  key->Append(info, val.c_str());
  int ok_appendedlength = (key->Length(info) == 2);
  if (!ok_appendedlength)
  {
    cerr << "Appended length was " << key->Length(info) << " not 2.\n";
  }

  return ok_setgetcomp && ok_copyget && ok_length && ok_appendedlength;
}

int UnitTestInformationKeys(int svtkNotUsed(argc), char* svtkNotUsed(argv)[])
{
  int ok = 1;
  svtkNew<svtkInformation> info;
  svtkVariant tvval("foo");
  double tdval = svtkMath::Pi();
  svtkStdString tsval = "bar";

  svtkInformationVariantKey* tvskey = new svtkInformationVariantKey("Test", "svtkTest");
  ok &= UnitTestScalarValueKey(info, tvskey, tvval);

  svtkInformationVariantVectorKey* tvvkey = new svtkInformationVariantVectorKey("Test", "svtkTest");
  ok &= UnitTestVectorValueKey(info, tvvkey, tvval);

  svtkInformationDoubleKey* tdskey = new svtkInformationDoubleKey("Test", "svtkTest");
  ok &= UnitTestScalarValueKey(info, tdskey, tdval);

  svtkInformationDoubleVectorKey* tdvkey = new svtkInformationDoubleVectorKey("Test", "svtkTest");
  ok &= UnitTestVectorValueKey(info, tdvkey, tdval);

  svtkInformationStringKey* tsskey = new svtkInformationStringKey("Test", "svtkTest");
  ok &= UnitTestScalarValueKey(info, tsskey, tsval);

  svtkInformationStringVectorKey* tsvkey = new svtkInformationStringVectorKey("Test", "svtkTest");
  ok &= UnitTestVectorValueKey(info, tsvkey, tsval);

  return !ok;
}
