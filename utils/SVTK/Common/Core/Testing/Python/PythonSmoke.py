import sys

try:
  import svtk

except:
  print("Cannot import svtk")
  sys.exit(1)
try:
  print(dir(svtk))
except:
  print("Cannot print(dir(svtk)")
  sys.exit(1)

try:
  try:
    try:
      o = svtk.svtkLineWidget()
      print("Using Hybrid")
    except:
      o = svtk.svtkActor()
      print("Using Rendering")
  except:
    o = svtk.svtkObject()
    print("Using Common")
except:
  print("Cannot create svtkObject")
  sys.exit(1)

try:
  print(o)
  print("Reference count: %d" % o.GetReferenceCount())
  print("Class name: %s" % o.GetClassName())
except:
  print("Cannot print object")
  sys.exit(1)

try:
  b = svtk.svtkObject()
  d = b.SafeDownCast(o)
  print(repr(b) + " " + repr(d))
except:
  print("Cannot downcast")
  sys.exit(1)

sys.exit(0)

