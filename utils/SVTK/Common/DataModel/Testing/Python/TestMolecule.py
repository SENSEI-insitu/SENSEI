#!/usr/bin/env python

"""
This file tests svtk.svtkMolecule, and verifies that atoms/bonds are added.
"""

import sys
import svtk
from svtk.test import Testing

class TestMolecule(Testing.svtkTest):
    def testCreation(self):
        "Testing if molecules can be created/modified."
        mol = svtk.svtkMolecule()

        self.assertEqual(mol.GetNumberOfAtoms(), 0, "Number of atoms incorrect")
        self.assertEqual(mol.GetNumberOfBonds(), 0, "Number of atoms incorrect")
        h1 = mol.AppendAtom(1, 0.0, 0.0, -0.5)
        h2 = mol.AppendAtom(1, 0.0, 0.0,  0.5)
        b = mol.AppendBond(h1, h2, 1)
        self.assertEqual(mol.GetNumberOfAtoms(), 2, "Number of atoms incorrect")
        self.assertEqual(mol.GetNumberOfBonds(), 1, "Number of atoms incorrect")

if __name__ == "__main__":
    Testing.main([(TestMolecule, 'test')])
