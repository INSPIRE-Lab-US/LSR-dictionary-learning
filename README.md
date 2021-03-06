[![DOI](https://zenodo.org/badge/243341907.svg)](https://zenodo.org/badge/latestdoi/243341907)

# Learning Mixtures of Separable Dictionaries for Tensor Data: Codebase for Numerical Experiments

## Table of Contents
<!-- MarkdownTOC -->
- [General Information](#introduction)
- [Reproducing Real-data Experiments](#real_experiments)
- [Reproducing Online-learning Experiments](#online_experiments)
- [Reproducing Synthetic-data Experiments](#synthetic_experiments)
- [Contributors](#contributors)
<!-- /MarkdownTOC -->

<a name="introduction"></a>
# General Information
This repo contains the code used for numerical experiments in the "[Learning Mixtures of Separable Dictionaries for Tensor Data: Analysis and Algorithm](https://ieeexplore.ieee.org/document/8892653)" paper.

## License and Citation
The code in this repo is being released under the GNU General Public License v3.0; please refer to the [LICENSE](./LICENSE) file in the repo for detailed legalese pertaining to the license. In particular, if you use any part of this code then you must cite both the original paper as well as this codebase as follows:

**Paper Citation:** M. Ghassemi, Z. Shakeri, A.D. Sarwate, and W.U. Bajwa, "Learning mixtures of separable dictionaries for tensor data: Analysis and algorithms," IEEE Trans. Signal Processing, vol. 68, pp. 33-48, 2020; doi: [10.1109/TSP.2019.2952046](https://doi.org/10.1109/TSP.2019.2952046).

**Codebase Citation:** J. Shenouda, M. Ghassemi, Z. Shakeri, A.D. Sarwate, and W.U. Bajwa, "Codebase---Learning mixtures of separable dictionaries for tensor data: Analysis and algorithms," GitHub Repository, 2020; doi: [10.5281/zenodo.3901852](https://doi.org/10.5281/zenodo.3901852).

## Computing Environment
All of our computational experiments were done using MATLAB R2019a. All the experiments were carried out on a Linux high-performance computing (HPC) cluster provided by the Rutgers Office of Advanced Research in Computing; specifically, all of the experiments were run on: 

Lenovo NextScale nx360 servers:
- 2 x 12-core Intel Xeon E5-2680 v3 "Haswell" processors
- 128 GB RAM
- 1 TB local scratch disk

## Summary of Experiments
In the paper, we conducted three main sets of experiments to produce all plots and tables.

1. Comparison of six different dictionary learning algorithms in denoising four different images (Real-data Experiments)
2. Performance evaluation of online dictionary learning algorithms for denoising the "House" image (Online-learning Experiments)
3. Comparison of four different dictionary learning algorithms on synthetic data (Synthetic-data Experiments)

Almost all of the experiment were completed in about 3 days; however, some of the larger images in the denoising experiments needed about 5 days.

**Note:** Precise values of some of the parameters, such as the random seeds, initially used to generate results in the paper were lost. Nonetheless, all the results obtained from this codebase are consistent with all the discussions and conclusions made in the paper.

## External Dependencies
In order to reproduce our results for image denoising with the [SeDiL](https://doi.org/10.1109/CVPR.2013.63) algorithm, you will need the source code for SeDiL; however, we do not have permission to publicize that code. In the absence of that code, you can run the alternative function `LSRImageDenoising_noSeDiL.m`. Alternatively, you can contact us with proof of express permission from the original authors of the SeDiL algorithm, after which we can provide you the codebase that includes SeDiL.

<a name="real_experiments"></a>
# Real-data Experiments
The `Real_Experiments` directory contains the code used to produce the results for the real image denoising experiments as described in the paper.

## Steps to reproduce the results
### Table II in the Paper: Performance of all Dictionary Learning Algorithms
To perform the image denoising experiments for Table II in the paper, we had one function `LSRImageDenoising.m` that was used for each image by passing in different parameters to the function. In order to speed up our computations, we ran the`LSRImageDenoising.m` function three times for each image and then concatenated our representation errors in all three `.mat` files that our function returned to give us results corresponding to a total of 25 Monte Carlo trials.

For example to perform image denoising experiments on the "House" image, we ran:
- `LSRImageDenoising(8, '../Data/rand_state1.mat','../Data/house_color.tiff', "House", "rnd1");`
- `LSRImageDenoising(8, '../Data/rand_state2.mat', '../Data/house_color.tiff', "House", "rnd2")`
- `LSRImageDenoising(9, '../Data/rand_State3.mat', '../Data/house_color.tiff', "House", "rnd3")`

as three separate jobs on our computing cluster. Each function call generated a `.mat` file in a directory pertaining to the image that was denoised. 

After the experiments for an image were done running, we ran the script in the respective image directory (e.g., `House/getHousePSNR.m`) in order to produce a table similar to the one in the paper with the PSNR obtained for each algorithm.

### Table III in the Paper: Performance of TeFDiL With Various Ranks on "Mushroom"
To reproduce Table III in the paper, we ran the `mushroomDenoisingTeFDiL.m` function three times.
- `mushroomDenoisingTeFDiL(8,'../Data/rand_state1','rnd1')`
- `mushroomDenoisingTeFDiL(8,'../Data/rand_state2','rnd2')`
- `mushroomDenoisingTeFDiL(9,'../Data/rand_state3','rnd3')`

This produces three `.mat` files under the `Real_Experiments/Mushroom` directory. Once all three functions finished running, we ran `getMushroomTeFDiLPSNR.m` to produce the PSNR values of TeFDiL at various ranks, corresponding to Table III in the paper.

## Runtime
On our servers, this job completed in three days for the House, Castle and Mushroom images; however for the Lena image, it took over five days for the job to finish completely.

<a name="online_experiments"></a>
# Online-learning Algorithm Experiments with House image
The `Online_Experiment` directory contains the code used to run the experiments for the online dictionary learning algorithms.

## Steps to reproduce the results
In order to reproduce Figure 3(b) in the paper, we ran the `HouseOnline.m` function twice; once with `Data/rand_state1` and again with `Data/rand_state2`. E.g.,
- `HouseOnline('../Data/rand_state1')`
- `HouseOnline('../Data/rand_state2')`

We split up the Monte Carlo trials over two jobs on our server for a total of 30 Monte Carlo trials.

After running the function twice (preferably at the same time as two jobs), it will save two new `.mat` files; copy those new `.mat` files to your local machine and run the  `plotsOnline.m` script, which will load in the two `.mat` files that were generated and concatenate them together before plotting the result. 

## Runtime
It took about three days for our online experiments to finish running.

<a name="synthetic_experiments"></a>
# Synthetic-data Experiments
The code for the synthetic experiments can be found in the `Synthetic_Experiments` directory.

## Steps to reproduce the results
In order to reproduce Figure 3(a) in the paper, we ran the `synthetic_experiments.m` file that returns a `.mat` file called `3D_synthetic_results_25MonteCarlo.mat` after the code has finished running. Once the code finishes execution, copy the generated `.mat` files to your local machine and run the `plot_synthetic.m` script in MATLAB. This will produce a plot of the average test error for each algorithm.

## Runtime
This set of experiments also took about three days to finish running on our computing cluster.

<a name="contributors"></a>
# Contributors

The original algorithms and experiments were developed by the authors of the paper:
- [Mohsen Ghassemi](http://eceweb1.rutgers.edu/~mg975/)
- [Zahra Shakeri](https://sites.google.com/view/zshakeri/home?authuser=1)
- [Anand D. Sarwate](https://www.ece.rutgers.edu/~asarwate/)
- [Waheed U. Bajwa](http://www.inspirelab.us/)

The reproducibility of this codebase and publicizing of it was made possible by:
- [Joseph Shenouda](https://github.com/jshen99)
