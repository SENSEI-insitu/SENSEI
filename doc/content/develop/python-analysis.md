---
markdown:
  gfm: true
  breaks: false
---
# PythonAnalysis adaptor

The python analysis adaptor enables the use of a Python scripts as an analysis
back end. It accomplishes this by embedding a Python interpreter and includes a
minimal set of the sensei python bindings. To author a new python analysis one
must provide a python script that implements three functions: `Inititalize`,
`Execute` and `Finalize`. These functions implement the `sensei::AnalysisAdaptor`
API. The `Execute` function is required while `Initialize` and `Finalize`
functions are optional. The `Execute` function is passed a `sensei::DataAdaptor`
instance from which one has access to simulation data structures. If an error
occurs during processing one should `raise` an exception. If the analysis
required MPI communication, one must make use of the adaptor's MPI communicator
which is stored in the global variable `comm`.  Additionally one can provide a
secondary script that is executed prior to the API functions. This script can
set global variables that control runtime behavior.

End users will make use of the `sensei::ConfigurableAnalysis` and point to the
python analysis script. The script can be loaded in one of two ways: via
python's import machinery or via a customized mechanism that reads the file on
MPI rank 0 and broadcasts it to the other ranks. The latter is the recommended
approach.

## ConfigurableAnalysis XML

The `<analysis>` element is used to create and configure the `PythonAnalysis` instance.

| name | type | allowable value(s) | description |
|------|------|--------------------|-------------|
| `type` | attribute | "python" | Creates a `PythonAnalysis` instance |
| `script_module` | attribute | a module name | Names a module that is in the PYTHONPATH. The module should define the 3 analysis adaptor API functions: `Initialize`, `Execute`, and `Finalize`. It is imported during initialization using python's import machinery. \* |
| `script_file` | attribute | a file path | A path to a python script to be loaded and broadcast by rank 0. The script should define the 3 analysis adaptor API functions: `Initialize`, `Execute`, and `Finalize`. \*|
| `enabled` | attribute | 0,1 | When 0 the analysis is skipped |
| `initialize_source` | child element | python source code | A snippet of source code that can be used to control run time behavior. The source code must be properly formatted and indented. The contents of the element are taken verbatim including newline tabs and spaces. |

\* -- use one of `script_file` or `script_module`. Prefer `script_file`.

See the [example](#histogramxml) below.

## Code Template
The following template provides stubs that one can fill in to write a new python
analysis.

```python
# YOUR IMPORTS HERE

def Initialize():
  """ Initialization code """
  # YOUR CODE HERE
  return

def Execute(dataAdaptor):
  """ Use sensei::DataAdaptor instance passed in
      dataAdaptor to access and process simulation data """
  # YOUR CODE HERE
  return

def Finalize():
  """ Finalization code """
  # YOUR CODE HERE
  return
```

## Example
The following example computes a histogram in parallel. It is included in the source code at `sensei/Histogram.py`.

### Histogram.py
```python
import sys
import numpy as np
import vtk.util.numpy_support as vtknp
from vtk import vtkDataObject, vtkCompositeDataSet, vtkMultiBlockDataSet

# default values of control parameters
numBins = 10
meshName = ''
arrayName = ''
arrayCen = vtkDataObject.POINT
outFile = 'hist'

def Initialize():
    # check for valid control parameters
    if not meshName:
        raise RuntimeError('meshName was not set')
    if not arrayName:
        raise RuntimeError('arrayName was not set')

def Execute(adaptor):
    r = comm.Get_rank()

    # get the mesh and array we need
    mesh = adaptor.GetMesh(meshName, True)
    adaptor.AddArray(mesh, meshName, arrayCen, arrayName)

    # force composite data to simplify computations
    if not isinstance(mesh, vtkCompositeDataSet):
        s = comm.Get_size()
        mb = vtkMultiBlockDataSet()
        mb.SetNumberOfBlocks(s)
        mb.SetBlock(r, mesh)
        mesh = mb

    # compute the min and max over local blocks
    mn = sys.float_info.max
    mx = -mn
    it = mesh.NewIterator()
    while not it.IsDoneWithTraversal():
        do = it.GetCurrentDataObject()

        atts = do.GetPointData() if arrayCen == vtkDataObject.POINT \
             else do.GetCellData()

        da = vtknp.vtk_to_numpy(atts.GetArray(arrayName))

        mn = min(mn, np.min(da))
        mx = max(mx, np.max(da))

        it.GoToNextItem()

    # compute global min and max
    mn = comm.allreduce(mn, op=MPI.MIN)
    mx = comm.allreduce(mx, op=MPI.MAX)

    # compute the histogram over local blocks
    it.InitTraversal()
    while not it.IsDoneWithTraversal():
        do = it.GetCurrentDataObject()

        atts = do.GetPointData() if arrayCen == vtkDataObject.POINT \
             else do.GetCellData()

        da = vtknp.vtk_to_numpy(atts.GetArray(arrayName))

        h,be = np.histogram(da, bins=numBins, range=(mn,mx))

        hist = hist + h if 'hist' in globals() else h

        it.GoToNextItem()

    # compute the global histogram on rank 0
    h = comm.reduce(hist, root=0, op=MPI.SUM)

    # rank 0 write to disk
    if r == 0:
        ts = adaptor.GetDataTimeStep()
        fn = '%s_%s_%d.txt'%(outFile, arrayName, ts)
        f = file(fn, 'w')
        f.write('num bins : %d\n'%(numBins))
        f.write('range : %0.6g %0.6g\n'%(mn, mx))
        f.write('bin edges: ')
        for v in be:
            f.write('%0.6g '%(v))
        f.write('\n')
        f.write('counts : ')
        for v in h:
            f.write('%d '%(v))
        f.write('\n')
        f.close()

def Finalize():
    return
```

### Histogram.xml

```xml
  <analysis type="python" script_file="./Histogram.py" enabled="1">
    <initialize_source>
numBins=10
meshName='mesh'
arrayName='values'
arrayCen=1
     </initialize_source>
  </analysis>
```
