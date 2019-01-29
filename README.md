# Smart FoV

This repo contains a modification to the spatial partitioning FRNNs search algorithm. The algorithm only accesses bins within a 180 degree field of view, based on the agents velocity. This can greatly reduce the memory accesses and hence improve performance.

The code runs the PedestrianLOD model from FlameGPU, which consists of a random pedestrian walk model. There is no visualisation, however it validates that results are in agreement. The failures are currently in the order of 0.0001%, but these change at runtime due to the non deterministic ordering of the atomic sort used.

The code could easily be extended to support arbitrary fields of views (at a minor additional computational expense), however this has not yet been included.