# Smart FoV

This repo contains a modification to the spatial partitioning FRNNs search algorithm. The algorithm only accesses bins within a 180 degree field of view, based on the agents velocity. This can greatly reduce the memory accesses and hence improve performance.

The code runs the PedestrianLOD model from FlameGPU, which consists of a random pedestrian walk model. There is no visualisation, however it validates that results are in agreement. The failures are currently in the order of 0.0001%, but these change at runtime due to the non deterministic ordering of the atomic sort used.

The code could easily be extended to support arbitrary fields of views (at a minor additional computational expense, as calculating 90 degrees can be carried out with element swapping and negation), however this has not yet been included.

## Testing notes

In larger models there is a regular speedup under the Random walk model as high as 40% when agents are not sorted by location [I believe FLAMEGPU operates like this). This speedup is much smaller in the low occupancy configurations, this can be attributed to there being no bottleneck with such low densities and agent populations. There is a high potential to see better improvements in these same configurations on worse hardware.
I put alot of work in debugging the technique to confirm it performs identical to the control. I did find several issues, which I corrected. However, after verifying that highly variant outputs are processing the same set of neighbours (simply in a different order, due to parallel non-determinism), I’m confident that they are performing identically, despite a 0.0001% failure rate with an epsilon of 0.0001f (the number of failures changes over multiple runs, due to how the parallel sort differs ordering).


The tested Smart FOV implementation does not utilise strips or bin width. Both could probably be added, however they would add significant complexity to the code, likely impacting their ability to provide a benefit.
