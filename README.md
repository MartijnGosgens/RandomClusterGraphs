# RandomClusterGraphs

This repository implements the numeric methods that are described in *"The Erdős–Rényi random graph conditioned on every component being a clique"*.

We study the Erdős–Rényi (ER) graph conditioned on the rare event that every connected component forms a clique. We call such graphs *Cluster graphs* and refer to this conditional distribution as the *Random Cluster Graph* (RCG). Each RCG corresponds to a partition of the vertices into cliques, so that the RCG can be interpreted as a partition distribution.

The file `rcg.py` implements the main functions while `figures.py` generates the figures. The community detection experiment requires the [Hyperspherical Community Detection](https://github.com/MartijnGosgens/hyperspherical_community_detection) repository to be available in the same directory.



