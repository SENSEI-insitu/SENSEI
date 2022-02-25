/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestVariantComparison.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkObject.h"
#include "svtkVariant.h"

#include <cstdio>
#include <map>

int TestVariantComparison(int, char*[])
{
  signed char positiveChar = 100;
  signed char negativeChar = -100;
  short positiveShort = 10000;
  short negativeShort = -10000;
  int positiveInt = 100000;
  int negativeInt = -100000;
  long positiveLong = 1000000;
  long negativeLong = -1000000;

  int shiftAmount64 = 8 * sizeof(svtkTypeInt64) - 2;
  int shiftAmountInt = 8 * sizeof(int) - 2;
  int shiftAmountLong = 8 * sizeof(long) - 2;

  svtkTypeInt64 positive64 = static_cast<svtkTypeInt64>(1) << shiftAmount64;
  svtkTypeInt64 negative64 = -positive64;

  // There is nothing inherently magical about these values.  I just
  // happen to like them and they're outside the range of signed
  // integers.
  unsigned char unsignedChar = 192;
  unsigned short unsignedShort = 49152;
  unsigned int unsignedInt = (static_cast<unsigned int>(1) << shiftAmountInt) * 3;
  unsigned long unsignedLong = (static_cast<unsigned long>(1) << shiftAmountLong) * 3;
  svtkTypeUInt64 unsigned64 = 3 * (static_cast<svtkTypeUInt64>(1) << shiftAmount64);

  svtkStdString numberString("100000");
  svtkStdString alphaString("ABCDEFG");

  float positiveFloat = 12345.678;
  float negativeFloat = -12345.678;
  double positiveDouble = 123456789.012345;
  double negativeDouble = -123456789.012345;

  svtkObject* fooObject = svtkObject::New();

  svtkVariant invalidVariant;

  // Now we need variants for all of those
  svtkVariant positiveCharVariant(positiveChar);
  svtkVariant unsignedCharVariant(unsignedChar);
  svtkVariant negativeCharVariant(negativeChar);

  svtkVariant positiveShortVariant(positiveShort);
  svtkVariant unsignedShortVariant(unsignedShort);
  svtkVariant negativeShortVariant(negativeShort);

  svtkVariant positiveIntVariant(positiveInt);
  svtkVariant unsignedIntVariant(unsignedInt);
  svtkVariant negativeIntVariant(negativeInt);

  svtkVariant positiveLongVariant(positiveLong);
  svtkVariant unsignedLongVariant(unsignedLong);
  svtkVariant negativeLongVariant(negativeLong);

  svtkVariant positive64Variant(positive64);
  svtkVariant unsigned64Variant(unsigned64);
  svtkVariant negative64Variant(negative64);

  svtkVariant positiveFloatVariant(positiveFloat);
  svtkVariant negativeFloatVariant(negativeFloat);
  svtkVariant positiveDoubleVariant(positiveDouble);
  svtkVariant negativeDoubleVariant(negativeDouble);

  svtkVariant numberStringVariant(numberString);
  svtkVariant alphaStringVariant(alphaString);

  svtkVariant fooObjectVariant(fooObject);

  int errorCount = 0;
  int overallErrorCount = 0;

#define CHECK_EXPRESSION_FALSE(expr)                                                               \
  {                                                                                                \
    if ((expr))                                                                                    \
    {                                                                                              \
      ++errorCount;                                                                                \
      cerr << "TEST FAILED: " << #expr << " should have been false\n\n";                           \
    }                                                                                              \
  }

#define CHECK_EXPRESSION_TRUE(expr)                                                                \
  {                                                                                                \
    if (!(expr))                                                                                   \
    {                                                                                              \
      ++errorCount;                                                                                \
      cerr << "TEST FAILED: " << #expr << " should have been true\n\n";                            \
    }                                                                                              \
  }

  cerr << "Testing same-type comparisons... ";
  CHECK_EXPRESSION_FALSE(positiveCharVariant < negativeCharVariant);
  CHECK_EXPRESSION_FALSE(unsignedCharVariant < positiveCharVariant);
  CHECK_EXPRESSION_FALSE(unsignedCharVariant < negativeCharVariant);

  CHECK_EXPRESSION_FALSE(positiveShortVariant < negativeShortVariant);
  CHECK_EXPRESSION_FALSE(unsignedShortVariant < positiveShortVariant);
  CHECK_EXPRESSION_FALSE(unsignedShortVariant < negativeShortVariant);

  CHECK_EXPRESSION_FALSE(positiveIntVariant < negativeIntVariant);
  CHECK_EXPRESSION_FALSE(unsignedIntVariant < positiveIntVariant);
  CHECK_EXPRESSION_FALSE(unsignedIntVariant < negativeIntVariant);

  CHECK_EXPRESSION_FALSE(positiveLongVariant < negativeLongVariant);
  CHECK_EXPRESSION_FALSE(unsignedLongVariant < positiveLongVariant);
  CHECK_EXPRESSION_FALSE(unsignedLongVariant < negativeLongVariant);

  CHECK_EXPRESSION_FALSE(positive64Variant < negative64Variant);
  CHECK_EXPRESSION_FALSE(unsigned64Variant < positive64Variant);
  CHECK_EXPRESSION_FALSE(unsigned64Variant < negative64Variant);

  CHECK_EXPRESSION_FALSE(positiveFloatVariant < negativeFloatVariant);
  CHECK_EXPRESSION_FALSE(positiveDoubleVariant < negativeDoubleVariant);

  CHECK_EXPRESSION_FALSE(alphaString < numberString);

  if (errorCount == 0)
  {
    cerr << "Test succeeded.\n";
  }
  else
  {
    cerr << errorCount << " error(s) found!\n";
  }
  overallErrorCount += errorCount;
  errorCount = 0;

  cerr << "Testing cross-type comparisons... ";

  CHECK_EXPRESSION_FALSE(positiveShortVariant < positiveCharVariant);
  CHECK_EXPRESSION_FALSE(positiveIntVariant < positiveCharVariant);
  CHECK_EXPRESSION_FALSE(positiveLongVariant < positiveCharVariant);
  CHECK_EXPRESSION_FALSE(positive64Variant < positiveCharVariant);

  CHECK_EXPRESSION_FALSE(positiveShortVariant < negativeCharVariant);
  CHECK_EXPRESSION_FALSE(positiveIntVariant < negativeCharVariant);
  CHECK_EXPRESSION_FALSE(positiveLongVariant < negativeCharVariant);
  CHECK_EXPRESSION_FALSE(positive64Variant < negativeCharVariant);

  CHECK_EXPRESSION_FALSE(positiveShortVariant < unsignedCharVariant);
  CHECK_EXPRESSION_FALSE(positiveIntVariant < unsignedCharVariant);
  CHECK_EXPRESSION_FALSE(positiveLongVariant < unsignedCharVariant);
  CHECK_EXPRESSION_FALSE(positive64Variant < unsignedCharVariant);

  CHECK_EXPRESSION_FALSE(negativeCharVariant < negativeShortVariant);
  CHECK_EXPRESSION_FALSE(negativeCharVariant < negativeIntVariant);
  CHECK_EXPRESSION_FALSE(negativeCharVariant < negativeLongVariant);
  CHECK_EXPRESSION_FALSE(negativeCharVariant < negative64Variant);

  CHECK_EXPRESSION_FALSE(unsignedShortVariant < negativeCharVariant);
  CHECK_EXPRESSION_FALSE(unsignedIntVariant < negativeCharVariant);
  CHECK_EXPRESSION_FALSE(unsignedLongVariant < negativeCharVariant);
  CHECK_EXPRESSION_FALSE(unsigned64Variant < negativeCharVariant);

  CHECK_EXPRESSION_FALSE(positiveFloatVariant < positiveCharVariant);
  CHECK_EXPRESSION_FALSE(positiveFloatVariant < negativeCharVariant);
  CHECK_EXPRESSION_FALSE(positiveFloatVariant < unsignedCharVariant);

  CHECK_EXPRESSION_FALSE(positiveDoubleVariant < positiveCharVariant);
  CHECK_EXPRESSION_FALSE(positiveDoubleVariant < negativeCharVariant);
  CHECK_EXPRESSION_FALSE(positiveDoubleVariant < unsignedCharVariant);

  CHECK_EXPRESSION_FALSE(alphaStringVariant < positiveIntVariant);
  CHECK_EXPRESSION_FALSE(numberStringVariant != positiveIntVariant);
  CHECK_EXPRESSION_FALSE(positiveDoubleVariant < fooObjectVariant);
  CHECK_EXPRESSION_FALSE(positiveFloatVariant < invalidVariant);

  if (errorCount == 0)
  {
    cerr << "Test succeeded.\n";
  }
  else
  {
    cerr << errorCount << " error(s) found!\n";
  }
  overallErrorCount += errorCount;
  errorCount = 0;

  cerr << "Testing cross-type equality...";

  char c = 100;
  short s = 100;
  int i = 100;
  long l = 100;
  svtkTypeInt64 i64 = 100;
  float f = 100;
  double d = 100;
  svtkStdString str("100");

  CHECK_EXPRESSION_TRUE(svtkVariant(c) == svtkVariant(s));
  CHECK_EXPRESSION_TRUE(svtkVariant(c) == svtkVariant(i));
  CHECK_EXPRESSION_TRUE(svtkVariant(c) == svtkVariant(l));
  CHECK_EXPRESSION_TRUE(svtkVariant(c) == svtkVariant(i64));
  CHECK_EXPRESSION_TRUE(svtkVariant(c) == svtkVariant(f));
  CHECK_EXPRESSION_TRUE(svtkVariant(c) == svtkVariant(d));

  CHECK_EXPRESSION_TRUE(svtkVariant(s) == svtkVariant(i));
  CHECK_EXPRESSION_TRUE(svtkVariant(s) == svtkVariant(l));
  CHECK_EXPRESSION_TRUE(svtkVariant(s) == svtkVariant(i64));
  CHECK_EXPRESSION_TRUE(svtkVariant(s) == svtkVariant(f));
  CHECK_EXPRESSION_TRUE(svtkVariant(s) == svtkVariant(d));
  CHECK_EXPRESSION_TRUE(svtkVariant(s) == svtkVariant(str));

  CHECK_EXPRESSION_TRUE(svtkVariant(i) == svtkVariant(l));
  CHECK_EXPRESSION_TRUE(svtkVariant(i) == svtkVariant(i64));
  CHECK_EXPRESSION_TRUE(svtkVariant(i) == svtkVariant(f));
  CHECK_EXPRESSION_TRUE(svtkVariant(i) == svtkVariant(d));
  CHECK_EXPRESSION_TRUE(svtkVariant(i) == svtkVariant(str));

  CHECK_EXPRESSION_TRUE(svtkVariant(l) == svtkVariant(i64));
  CHECK_EXPRESSION_TRUE(svtkVariant(l) == svtkVariant(f));
  CHECK_EXPRESSION_TRUE(svtkVariant(l) == svtkVariant(d));
  CHECK_EXPRESSION_TRUE(svtkVariant(l) == svtkVariant(str));

  CHECK_EXPRESSION_TRUE(svtkVariant(i64) == svtkVariant(f));
  CHECK_EXPRESSION_TRUE(svtkVariant(i64) == svtkVariant(d));
  CHECK_EXPRESSION_TRUE(svtkVariant(i64) == svtkVariant(str));

  CHECK_EXPRESSION_TRUE(svtkVariant(f) == svtkVariant(d));
  CHECK_EXPRESSION_TRUE(svtkVariant(f) == svtkVariant(str));

  CHECK_EXPRESSION_TRUE(svtkVariant(d) == svtkVariant(str));

  if (errorCount == 0)
  {
    cerr << " Test succeeded.\n";
  }
  else
  {
    cerr << errorCount << " error(s) found!\n";
  }
  overallErrorCount += errorCount;
  errorCount = 0;

  cerr << "Testing svtkVariant as STL map key... ";

  std::map<svtkVariant, svtkStdString> TestMap;

  TestMap[svtkVariant(s)] = "short";
  TestMap[svtkVariant(i)] = "int";
  TestMap[svtkVariant(l)] = "long";
  TestMap[svtkVariant(i64)] = "int64";
  TestMap[svtkVariant(f)] = "float";
  TestMap[svtkVariant(d)] = "double";
  TestMap[svtkVariant(str)] = "string";

  CHECK_EXPRESSION_TRUE(TestMap.find(svtkVariant(100)) != TestMap.end());
  CHECK_EXPRESSION_TRUE(TestMap[svtkVariant(100)] == "string");
  CHECK_EXPRESSION_TRUE(TestMap.size() == 1);

  if (errorCount == 0)
  {
    cerr << " Test succeeded.\n";
  }
  else
  {
    cerr << errorCount << " error(s) found!\n";
  }
  overallErrorCount += errorCount;
  errorCount = 0;

  cerr << "Testing svtkVariant as STL map key with strict weak ordering (fast comparator)...";

  // This one should treat variants containing different types as
  // unequal.
  std::map<svtkVariant, svtkStdString, svtkVariantStrictWeakOrder> TestMap2;
  TestMap2[svtkVariant()] = "invalid";
  TestMap2[svtkVariant(s)] = "short";
  TestMap2[svtkVariant(i)] = "int";
  TestMap2[svtkVariant(l)] = "long";
  TestMap2[svtkVariant(i64)] = "int64";
  TestMap2[svtkVariant(f)] = "float";
  TestMap2[svtkVariant(d)] = "double";
  TestMap2[svtkVariant(str)] = "string";

  CHECK_EXPRESSION_TRUE(TestMap2.find(svtkVariant()) != TestMap2.end());
  CHECK_EXPRESSION_TRUE(TestMap2[svtkVariant()] == "invalid");

  CHECK_EXPRESSION_TRUE(TestMap2.find(svtkVariant(s)) != TestMap2.end());
  CHECK_EXPRESSION_TRUE(TestMap2[svtkVariant(s)] == "short");

  CHECK_EXPRESSION_TRUE(TestMap2.find(svtkVariant(i)) != TestMap2.end());
  CHECK_EXPRESSION_TRUE(TestMap2[svtkVariant(i)] == "int");

  CHECK_EXPRESSION_TRUE(TestMap2.find(svtkVariant(l)) != TestMap2.end());
  CHECK_EXPRESSION_TRUE(TestMap2[svtkVariant(l)] == "long");

  CHECK_EXPRESSION_TRUE(TestMap2.find(svtkVariant(i64)) != TestMap2.end());
  CHECK_EXPRESSION_TRUE(TestMap2[svtkVariant(i64)] == "int64");

  CHECK_EXPRESSION_TRUE(TestMap2.find(svtkVariant(f)) != TestMap2.end());
  CHECK_EXPRESSION_TRUE(TestMap2[svtkVariant(f)] == "float");

  CHECK_EXPRESSION_TRUE(TestMap2.find(svtkVariant(d)) != TestMap2.end());
  CHECK_EXPRESSION_TRUE(TestMap2[svtkVariant(d)] == "double");

  CHECK_EXPRESSION_TRUE(TestMap2.find(svtkVariant(str)) != TestMap2.end());
  CHECK_EXPRESSION_TRUE(TestMap2[svtkVariant("100")] == "string");

  CHECK_EXPRESSION_TRUE(TestMap2.size() == 8);

  if (errorCount == 0)
  {
    cerr << " Test succeeded.\n";
  }
  else
  {
    cerr << errorCount << " error(s) found!\n";
  }
  overallErrorCount += errorCount;

  if (overallErrorCount == 0)
  {
    cerr << "All tests succeeded.\n";
  }
  else
  {
    cerr << "Some tests failed!  Overall error count: " << overallErrorCount << "\n";
    cerr << "Debug information:\n";
    cerr << "CHAR(" << sizeof(char) << "): "
         << "positive " << positiveChar << ", "
         << "negative " << negativeChar << ", "
         << "unsigned " << unsignedChar << "\n";
    cerr << "SHORT(" << sizeof(short) << "): "
         << "positive " << positiveShort << ", "
         << "negative " << negativeShort << ", "
         << "unsigned " << unsignedShort << "\n";
    cerr << "INT(" << sizeof(int) << "): "
         << "positive " << positiveInt << ", "
         << "negative " << negativeInt << ", "
         << "unsigned " << unsignedInt << "\n";
    cerr << "LONG(" << sizeof(long) << "): "
         << "positive " << positiveLong << ", "
         << "negative " << negativeLong << ", "
         << "unsigned " << unsignedLong << "\n";
    cerr << "INT64(" << sizeof(svtkTypeInt64) << "): "
         << "positive " << positive64 << ", "
         << "negative " << negative64 << ", "
         << "unsigned " << unsigned64 << "\n";
  }

  fooObject->Delete();
  return (overallErrorCount > 0);
}
