"""Test iterating through a svtkCollection with the standard python syntax"""


import svtk
from svtk.test import Testing


class TestIterateCollection(Testing.svtkTest):
    def setUp(self):
        self.svtkObjs = [svtk.svtkObject() for _ in range(30)]
        self.collection = svtk.svtkCollection()
        for obj in self.svtkObjs:
            self.collection.AddItem(obj)

        self.emptyCollection = svtk.svtkCollection()

    def testIterateCollection(self):
        newObjsList = [obj for obj in self.collection]
        self.assertEqual(self.svtkObjs, newObjsList)

        counter = 0
        for _ in self.collection:
            counter += 1
        self.assertEqual(counter, 30)

        counter = 0
        for _ in self.emptyCollection:
           counter += 1
        self.assertEqual(counter, 0)

    def testCollectionChild(self):
        #check that iteration is being inherited correctly
        dataArray = svtk.svtkIntArray()

        dataArrayCollection = svtk.svtkDataArrayCollection()
        dataArrayCollection.AddItem(dataArray)

        self.assertEqual([obj for obj in dataArrayCollection],
                         [dataArray])

    def testOperators(self):
        self.assertTrue(self.svtkObjs[0] in self.collection)
        self.assertEqual(list(self.collection), self.svtkObjs)

    def testReferenceCounting(self):
        initialReferenceCount = self.collection.GetReferenceCount()
        list(self.collection)
        self.assertEqual(self.collection.GetReferenceCount(), initialReferenceCount)


if __name__ == "__main__":
    Testing.main([(TestIterateCollection, 'test')])
