DataAdaptor API
===============

SENSEI's data adaptor API abstracts away the differences between simulations
allowing SENSEI's transports and analysis back ends to access data from any
simulation in the same way. A simulation must implement the data adaptor API
and pass an instance when it wishes to trigger in situ processing.

Through the data adaptor API the analysis back end can get metadata about what
the simulation can provide. This metadata is examined and then the analysis can
use the API to fetch only the data it needs to accomplish the tasks it has been
configured to do.

Finally the data adaptor is a key piece of SENSEI's in transit system. The
analysis back end can be run in a different parallel job and be given an in
transit data adaptor in place of the simulation's data adaptor. In this scenario
the in transit data adaptor helps move data needed by the analysis back end.
The data adaptor API enables this scenario to apear the same to the simulation
and the analysis back end. Neither simulaiton nor analysis need be modified for
in transit processing.


Core API
--------
Simulations need to implement the core API.




In transit API
--------------
In transit transports need to implement the in transit API.




