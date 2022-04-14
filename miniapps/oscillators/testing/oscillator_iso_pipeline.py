from sensei import SliceExtract, VTKPosthocIO
from svtk import svtkDataObject
import sys

# control
meshName  = None
arrayName  = None
arrayCen  = None
isoVals  = None
outputDir  = None

# state
slicer = None
writer = None

def Initialize():
    global slicer,writer
    if comm.Get_rank() == 0:
        sys.stderr.write('PipelineExample::Initialize\n')
        sys.stderr.write('meshName=%s, arrayName=%s, arrayCen=%d, '
                         'isoVals=%s, outputDir=%s\n'%(meshName,
                         arrayName, arrayCen, str(isoVals), outputDir))

    slicer = SliceExtract.New()
    slicer.EnableWriter(0)
    slicer.SetOperation(SliceExtract.OP_ISO_SURFACE)
    slicer.SetIsoValues(meshName, arrayName, arrayCen, isoVals)
    slicer.AddDataRequirement(meshName, arrayCen, [arrayName])

    writer = VTKPosthocIO.New()
    writer.SetGhostArrayName('ghosts')
    writer.SetOutputDir(outputDir)
    writer.SetWriter(VTKPosthocIO.WRITER_VTK_XML)
    writer.SetMode(VTKPosthocIO.MODE_PARAVIEW)

def Execute(daIn):
    if comm.Get_rank() == 0:
        sys.stderr.write('PipelineExample::Execute %d\n'%(daIn.GetDataTimeStep()))

    # stage 1 - iso surface
    isoOk, iso = slicer.Execute(daIn)

    if not isoOk:
        return 0

    # stage 2 - write the result
    writer.Execute(iso)

    return 1

def Finalize():
    if comm.Get_rank() == 0:
        sys.stderr.write('PipelineExample::Finalize\n')

    slicer.Finalize()
    writer.Finalize()
