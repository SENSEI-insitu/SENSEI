.. _pipeline_demo:

Building Pipelines
==================
Pipelines of in situ and in transit operations can be constructed by chaining
`sensei::AnalysisAdaptors` using C++ or Python.

The following example shows a SENSEI `PythonAnalysis <https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_python_analysis.html>`_ script that configures and executes a simple pipeline that calculates a set of iso-surfaces using SENSEI's `SliceExtract <https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_slice_extract.html>`_ and then writes them to disk using SENSEI's `VTKPosthocIO <https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_v_t_k_posthoc_i_o.html>`_ analysis adaptor.

.. code-block:: python

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

For the purposes of this demo, the above Python script is stored in a file
named "iso_pipeline.py" and refered to in the XML used to configure the system.
The XML used to configure the system to run the iso-surface and write pipeline
is shown below:

.. code-block:: xml

    <sensei>
      <analysis type="python" script_file="iso_pipeline.py" enabled="1">
        <initialize_source>
    meshName = 'mesh'
    arrayName = 'data'
    arrayCen = svtkDataObject.CELL
    isoVals = [-0.619028, -0.0739720000000001, 0.47108399999999995, 1.01614]
    outputDir = 'oscillator_iso_pipeline_%s'%(meshName)
         </initialize_source>
      </analysis>
    </sensei>

The variables `meshName`, `arrayName`, `arrayCen`, `isoVals`, and `outputDir` are used to
configure for a specific simulation or run. For the purposes of this demo this
XML is stored in a file name `iso_pipeline.xml` and passed to the SENSEI
`ConfigurableAnalaysis <https://sensei-insitu.readthedocs.io/en/latest/doxygen/classsensei_1_1_configurable_analysis.html>`_
adaptor during SENSEI initialization.

One can run the above pipeline with the oscillator miniapp that ships with
SENSEI using the following shell script.

.. code-block:: bash

   #!/bin/bash

    module load mpi/mpich-x86_64
    module use /work/SENSEI/modulefiles/
    module load sensei-vtk

    mpiexec -np 4 oscillator -t .5 -b 4 -g 1 -f iso_pipeline.xml simple.osc


