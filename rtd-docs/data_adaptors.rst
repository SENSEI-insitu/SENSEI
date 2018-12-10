DataAdaptor API
===============
(Dave T)

Discuss the DataADaptor pupose use etc
Some points to mention:
link to data model section for info on mapping simulation domains to sensei meshes
link to data model section for info on providing metadata for mesh types and arrays
Lifecycle, we use ref counting, AnalysisAdaptor takes ownership, DataADaptors only cache when necessary, thus Release is typically a no-op
Deep dive into the API
template for writing a new adaptor
