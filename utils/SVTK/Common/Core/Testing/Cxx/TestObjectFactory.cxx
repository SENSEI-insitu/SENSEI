/*=========================================================================

  Program:   Visualization Toolkit
  Module:    ObjectFactory.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkDebugLeaks.h"
#include "svtkObjectFactory.h"
#include "svtkObjectFactoryCollection.h"
#include "svtkOutputWindow.h"
#include "svtkOverrideInformation.h"
#include "svtkOverrideInformationCollection.h"
#include "svtkPoints.h"
#include "svtkVersion.h"

static int failed = 0;

class svtkTestPoints : public svtkPoints
{
public:
  // Methods from svtkObject
  ~svtkTestPoints() override = default;

  svtkTypeMacro(svtkTestPoints, svtkPoints);
  static svtkTestPoints* New() { SVTK_STANDARD_NEW_BODY(svtkTestPoints); }
  svtkTestPoints() = default;

private:
  svtkTestPoints(const svtkTestPoints&) = delete;
  svtkTestPoints& operator=(const svtkTestPoints&) = delete;
};

class svtkTestPoints2 : public svtkPoints
{
public:
  ~svtkTestPoints2() override = default;

  // Methods from svtkObject
  svtkTypeMacro(svtkTestPoints2, svtkPoints);
  static svtkTestPoints2* New() { SVTK_STANDARD_NEW_BODY(svtkTestPoints2); }
  svtkTestPoints2() = default;

private:
  svtkTestPoints2(const svtkTestPoints2&) = delete;
  svtkTestPoints2& operator=(const svtkTestPoints2&) = delete;
};

SVTK_CREATE_CREATE_FUNCTION(svtkTestPoints);
SVTK_CREATE_CREATE_FUNCTION(svtkTestPoints2);

class SVTK_EXPORT TestFactory : public svtkObjectFactory
{
public:
  TestFactory();
  static TestFactory* New()
  {
    TestFactory* f = new TestFactory;
    f->InitializeObjectBase();
    return f;
  }
  const char* GetSVTKSourceVersion() override { return SVTK_SOURCE_VERSION; }
  const char* GetDescription() override { return "A fine Test Factory"; }

protected:
  TestFactory(const TestFactory&) = delete;
  TestFactory& operator=(const TestFactory&) = delete;
};

TestFactory::TestFactory()
{
  this->RegisterOverride("svtkPoints", "svtkTestPoints", "test vertex factory override", 1,
    svtkObjectFactoryCreatesvtkTestPoints);
  this->RegisterOverride("svtkPoints", "svtkTestPoints2", "test vertex factory override 2", 0,
    svtkObjectFactoryCreatesvtkTestPoints2);
}

void TestNewPoints(svtkPoints* v, const char* expectedClassName)
{
  if (strcmp(v->GetClassName(), expectedClassName) != 0)
  {
    failed = 1;
    cout << "Test Failed:\nExpected classname: " << expectedClassName
         << "\nCreated classname: " << v->GetClassName() << endl;
  }
}

int TestObjectFactory(int, char*[])
{
  svtkOutputWindow::GetInstance()->PromptUserOff();
  svtkGenericWarningMacro("Test Generic Warning");
  TestFactory* factory = TestFactory::New();
  svtkObjectFactory::RegisterFactory(factory);
  factory->Delete();
  svtkPoints* v = svtkPoints::New();
  TestNewPoints(v, "svtkTestPoints");
  v->Delete();

  // disable all svtkPoints creation with the
  factory->Disable("svtkPoints");
  v = svtkPoints::New();
  TestNewPoints(v, "svtkPoints");

  factory->SetEnableFlag(1, "svtkPoints", "svtkTestPoints2");
  v->Delete();
  v = svtkPoints::New();
  TestNewPoints(v, "svtkTestPoints2");

  factory->SetEnableFlag(0, "svtkPoints", "svtkTestPoints2");
  factory->SetEnableFlag(1, "svtkPoints", "svtkTestPoints");
  v->Delete();
  v = svtkPoints::New();
  TestNewPoints(v, "svtkTestPoints");
  v->Delete();
  svtkOverrideInformationCollection* oic = svtkOverrideInformationCollection::New();
  svtkObjectFactory::GetOverrideInformation("svtkPoints", oic);
  svtkOverrideInformation* oi;
  if (oic->GetNumberOfItems() != 2)
  {
    cout << "Incorrect number of overrides for svtkPoints, expected 2, got: "
         << oic->GetNumberOfItems() << "\n";
    failed = 1;
    if (oic->GetNumberOfItems() < 2)
    {
      return 1;
    }
  }
  svtkCollectionSimpleIterator oicit;
  oic->InitTraversal(oicit);
  oi = oic->GetNextOverrideInformation(oicit);
  oi->GetObjectFactory();

  if (strcmp(oi->GetClassOverrideName(), "svtkPoints"))
  {
    cout << "failed: GetClassOverrideName should be svtkPoints, is: " << oi->GetClassOverrideName()
         << "\n";
    failed = 1;
  }
  if (strcmp(oi->GetClassOverrideWithName(), "svtkTestPoints"))
  {
    cout << "failed: GetClassOverrideWithName should be svtkTestPoints, is: "
         << oi->GetClassOverrideWithName() << "\n";
    failed = 1;
  }
  if (strcmp(oi->GetDescription(), "test vertex factory override"))
  {
    cout << "failed: GetClassOverrideWithName should be test vertex factory override, is: "
         << oi->GetDescription() << "\n";
    failed = 1;
  }

  oi = oic->GetNextOverrideInformation(oicit);
  if (strcmp(oi->GetClassOverrideName(), "svtkPoints"))
  {
    cout << "failed: GetClassOverrideName should be svtkPoints, is: " << oi->GetClassOverrideName()
         << "\n";
    failed = 1;
  }
  if (strcmp(oi->GetClassOverrideWithName(), "svtkTestPoints2"))
  {
    cout << "failed: GetClassOverrideWithName should be svtkTestPoints2, is: "
         << oi->GetClassOverrideWithName() << "\n";
    failed = 1;
  }
  if (strcmp(oi->GetDescription(), "test vertex factory override 2"))
  {
    cout << "failed: GetClassOverrideWithName should be test vertex factory override 2, is: "
         << oi->GetDescription() << "\n";
    failed = 1;
  }
  oic->Delete();
  svtkObjectFactory::UnRegisterAllFactories();
  return failed;
}
