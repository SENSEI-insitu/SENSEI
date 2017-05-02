# ADIOSAnalysisEndPoint

The end point reads data being serialized by the ADIOS analysis adaptor and
passes it back into a SENSEI bridge for further analysis. It can be given
an XML configuration for selecting analysis routines to run via SENSEI the
infrastructure.

To use the ADIOS end point first select a mini-app, and make a copy of it's XML
configurations file(located in the configs directory). Edit the configuration
so that the ADIOS analysis is enabled and the other analyses are disabled. Copy
the XML configuration a second time. The second copy is for the end point. In
the second config enable one or more of the other analyses while disabling the
ADIOS analysis. Run the mini-app with the first config file and the end point
with the second config.

Usage:
```bash
./bin/ADIOSAnalysisEndPoint [OPTIONS] input-stream-name
Options:
   -r, --readmethod STRING   specify read method: bp, bp_aggregate, dataspaces, dimes, or flexpath  [default: bp]
   -f, --config STRING       SENSEI analysis configuration xml (required)
   -h, --help                show help
```
