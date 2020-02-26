# Learning Mixtures of Separable Dictionaries for Tensor Data: Experiments
This repo contains the code used for experiments in the [Learning Mixtures of Separable Dictionaries for Tensor Data: Analysis and Algorithm](https://arxiv.org/abs/1903.09284) paper.

All of our computation experiments were carried out on a Linux high-performance computing (HPC) cluster provided by Rutgers Office of Advanced Research Computing specifically all of our experiments were run on: 
Lenovo NextScale nx360 servers
- 2 x 12-core Intel Xeon E5-2680 v3 "Haswell" processors
- 128 GB RAM
- 1 TB local scratch disk

Almost all experiment were completed in about 3 days however some of the larger images that were denoising needed about 5 days.
All of our experiments were done using MATLAB R2019a

In the paper we conducted three main experiments to produce the plots and charts.
1. Comparision of 4 dictionary learning algorithms on synthetic data (Synthetic Experiment)
2. Analysis of online dictionary learning algorithms on denoising the House image (Online Experiment)
3. Comparision of 6 algorithms in denoising 4 different images (Real Experiment)

**Note** Running these experiments may not give you the exact numbers reported in the paper however the results obtained from this codebase do not nullify any of the conclusions and analysis reported in the paper.

+ [Reproducing Synthetic Experiments](Synthetic_Experiments/README.md)
+ [Reproducing Online Experiment](Online_Experiment/README.md)
+ [Reproducing Real Experiment](Real_Experiments/README.md)


