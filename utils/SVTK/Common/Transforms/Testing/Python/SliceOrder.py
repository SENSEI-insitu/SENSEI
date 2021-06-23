#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
=========================================================================

  Program:   Visualization Toolkit
  Module:    TestNamedColorsIntegration.py

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================
'''

import svtk

class SliceOrder(object):
    '''
        These transformations permute medical image data to maintain proper
        orientation regardless of the acqusition order.
        After applying these transforms with svtkTransformFilter,
        a view up of 0,-1,0 will result in the body part
        facing the viewer.
        NOTE: some transformations have a -1 scale factor
              for one of the components.
              To ensure proper polygon orientation and normal direction,
              you must apply the svtkPolyDataNormals filter.

        Naming:
        si - superior to inferior (top to bottom)
        iss - inferior to superior (bottom to top)
        ap - anterior to posterior (front to back)
        pa - posterior to anterior (back to front)
        lr - left to right
        rl - right to left

    '''

    si = svtk.svtkTransform()
    si.SetMatrix([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1])

    # 'is' is a reserved word in Python so use 'iss'
    iss = svtk.svtkTransform()
    iss.SetMatrix([1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 1])

    ap = svtk.svtkTransform()
    ap.Scale(1, -1, 1)

    pa = svtk.svtkTransform()
    pa.Scale(1, -1, -1)

    lr = svtk.svtkTransform()
    lr.SetMatrix([0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])

    rl = svtk.svtkTransform()
    rl.SetMatrix([0, 0, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])

    #
    # the previous transforms assume radiological views of the slices
    # (viewed from the feet).
    #  Othermodalities such as physical sectioning may view from the head.
    # These transforms modify the original with a 180 rotation about y
    #
    hf = svtk.svtkTransform()
    hf.SetMatrix([-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1])

    hfsi = svtk.svtkTransform()
    hfsi.Concatenate(hf.GetMatrix())
    hfsi.Concatenate(si.GetMatrix())

    hfis = svtk.svtkTransform()
    hfis.Concatenate(hf.GetMatrix())
    hfis.Concatenate(iss.GetMatrix())

    hfap = svtk.svtkTransform()
    hfap.Concatenate(hf.GetMatrix())
    hfap.Concatenate(ap.GetMatrix())

    hfpa = svtk.svtkTransform()
    hfpa.Concatenate(hf.GetMatrix())
    hfpa.Concatenate(pa.GetMatrix())

    hflr = svtk.svtkTransform()
    hflr.Concatenate(hf.GetMatrix())
    hflr.Concatenate(lr.GetMatrix())

    hfrl = svtk.svtkTransform()
    hfrl.Concatenate(hf.GetMatrix())
    hfrl.Concatenate(rl.GetMatrix())
