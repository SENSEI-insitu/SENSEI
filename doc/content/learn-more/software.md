# Software Design

SENSEI provides simulations with a generic data interface that they use to provide access to their state.  SENSEI then passes this data to zero or more analysis and visualization tasks each time the simulation provides more data.

## Generic Data Interface

There are two main challenges to using in situ analysis for advanced modeling and simulation workflows. First, on the simulation side, is the complexity of instrumenting simulation codes to use any in situ infrastructure. Presently, one has to instrument their simulation codes separately for each of the infrastructures. Each infrastructure has their own idiosyncrasies that the application developer has to endure, including mapping simulation data structures to the target infrastructure. Second, on the analysis side, analysis developers face the challenge of having to decide on the infrastructure in which to implement their analysis. It is not feasible to write analysis code once and use it in various infrastructure without modifications.

The SENSEI generic data interface addresses both these key challenges. First, it provides application developers with a generic data interface that they then tailor for a particular use. Second, it provides analysis developers with a data model that they may use to write analysis routines. Both of these components are independent of the in situ infrastructure being used and hence provide both the simulation and the analysis routine isolation from which in situ infrastructure is being used. For example, if the application is instrumented with the SENSEI interface, application end-users can easily choose between ParaView/Catalyst and VisIt/Libsim for generating visualizations in situ. Furthermore, since ParaView/Catalyst and VisIt/Libsim both are treated as analysis routines under SENSEI, these visualizations can be run in situ, or in transit using ADIOS transparently.

This write once, use anywhere goal is only achievable when we have a mutually agreed platform for communicating the data between the simulation and analysis components – the data model. For the SENSEI interface, we selected the VTK data model. The VTK data model is widely used in the scientific and engineering data analysis and visualization community, leveraged by visualization tools like ParaView and VisIt and hence already familiar to a broader community.

To minimize effort and memory overhead when mapping memory layouts for data arrays from applications to VTK, we enhanced the VTK data model to support arbitrary layouts for multi-component arrays. VTK now natively supports the commonly encountered structure-of- arrays and array-of-structures layouts. This allows for mapping data arrays from application codes to the VTK data model without additional memory copying (zero-copy).

Besides the data model, the other components that comprise the SENSEI interface are simple and quite light weight. The figure shows the main components of the SENSEI interface. The data adaptor provides a mapping between simulation data structures and the VTK data model. The analysis adaptor passes the data described in form of VTK data objects to any analysis code, doing any necessary transformations. The in situ bridge is a simple mechanism to put together the analysis workflow i.e. initialize the data adaptor and execute selected analysis routines.

To instrument an application with SENSEI, one provides a concrete implementation for the data adaptor API. The data adaptor API provides the analysis code with access to mesh and attributes arrays as needed. By providing an API that encourages lazy mapping to VTK data model for the mesh and attribute arrays, the data adaptor avoids any work to map simulation data to VTK data when not needed. Thus when no analysis is enabled, the SENSEI instrumentation overhead is almost nonexistent.

To add an analysis routine to SENSEI, one provides a concrete implementation for the analysis adaptor API. The analysis adaptor is provided an instance of the data adaptor that it may use to gain access to the simulation data through VTK data model.

Finally, the in situ bridge is simply an API and the corresponding implementation that the application developer implements to pass data and control to SENSEI during the application execution. A typical bridge implementation will initialize the data adaptor and one or more analysis adaptors during the initialization phase of the simulation; then for each time step pass the current simulation data arrays and any other metadata to the data adaptor and call execute on the analysis adaptors.

The analysis adaptor is also the mechanism for the SENSEI interface to connect with the different in situ infrastructures. For example, an analysis adaptor may use ADIOS to save the data out to an ADIOS BP file, or it may serve as a ParaView/Catalyst-based adaptor that starts up ParaView/Catalyst to process the data using ParaView/Catalyst data processing pipelines, including rendering.

The SENSEI generic data interface creates several possibilities for in situ, in transit, in flight and hybrid analysis. In enables a developer to instrument a simulation code once, then have access to multiple in situ infrastructures. Allowing additional in situ infrastructures to be coupled via the SENSEI generic data interface provides a number of analysis techniques to map to future high-performance computing architectures.
 The current limitations of the SENSEI interface are an incomplete data model and an immature analysis adaptor specification. The SENSEI interface will truly be simpler when more complex simulation data structures easily map to the SENSEI data model through the data adaptor. Although this study examined several analysis and visualization use cases, this is just the tip of an iceberg of analysis techniques, and the adaptor infrastructure must grow to accommodate the requirements of the others.

Download the SENSEI generic data interface [here](https://gitlab.kitware.com/sensei/sensei).

## Analysis and Visualization

**ParaView Catalyst** (aka Catalyst) is an in situ analysis and visualization library that enables using ParaView’s visualization capabilities in in situ workflows. Applications can use Catalyst to execute complex analysis pipelines in step with the simulation, as well as connecting with the ParaView GUI for live, interactive visualization. To minimize memory footprint, Catalyst libraries are available in various flavors, called editions, that only enable components of ParaView used in the analysis pipelines.

[http://www.paraview.org/in-situ/](http://www.paraview.org/in-situ/)

**Libsim** is a library that makes available the full complement of features from VisIt so they may be used in situ. Libsim enables VisIt to connect interactively to running simulations for live exploration. Libsim can also be used directly to set up visualizations or it can use VisIt session files, which are XML files saved from the VisIt GUI, which can specify more complex visualizations. Once visualizations are set up, Libsim can save images for movie-making or it can save reduced-size data extracts for post hoc analysis.

**ADIOS** is an adaptive I/O service that is designed to allow applications to easily change between different I/O service providers. Only a tweak to the input parameters is needed to swap methods. This design allows for rapid conversion of post hoc analysis pipelines to in situ, in transit, or hybrid solutions by using one of the memory-to-memory “staging” methods, such as FlexPath or DataSpaces. The Flex-Path transport used in this effort can support same-node, multi-node, or even multi-machine deployment configurations. Unlike Catalyst and Libsim, however, ADIOS does not include any of the analytics functionality itself; it marshals the memory and metadata to make such code self-describing and adaptable to new situations. As such, it can partner effectively with Catalyst, Libsim, and other analytics infrastructures to provide whatever tools the scientist currently needs.

<!-- extra line breaks to prevent footer from obscuring text -->
<br><br><br>
