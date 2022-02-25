"""
Test case for the problem reported, tested and fixed on this post:

http://public.kitware.com/pipermail/svtk-developers/2008-August/005418.html

Without the fix as applied in the patch there, this code should crash on
exit.
"""

import svtk


class TestCase2 :
 def __init__(self) :
   self.Filter = svtk.svtkProgrammableFilter()
   self.Filter.SetExecuteMethod(self.ExecMethod)
 def ExecMethod(self) :
   print('execute method called')


test2 = TestCase2()

