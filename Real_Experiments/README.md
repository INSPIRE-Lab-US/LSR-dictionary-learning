# Real Experiments
This directory contains the code used to produce the results from the real image denoising experiments as described in the paper.

## Steps to reproduce
### Performance of all DL Algorithms Table
To perform the image denoising experiment we had one function `LSRImageDenoising.m` which was used for each image by passing in different parameters to the function. In order to speed up our computation we ran the`LSRImageDenoising.m` function 3 times for each image then concatenated our representation errors in all three `.mat` files we received to give us a a total of 25 monte carlos.

For example to perform image denoising experiments on the House image we ran:
- `LSRImageDenoising(8, '../Data/rand_state1.mat','../Data/house_color.tiff', "House", "rnd1");`
- `LSRImageDenoising(8, '../Data/rand_state2.mat', '../Data/house_color.tiff', "House", "rnd2")`
- `LSRImageDenoising(9, '../Data/rand_State3.mat', '../Data/house_color.tiff', "House", "rnd3")`

As three separate jobs on our computing cluster. Each function call generated `.mat` file in a directory pertaining to the image that was denoised. 

After the simulations for an image are done running, run the script in each image directory 
ex. `House/getHousePSNR.m` this will print out a table similar to the one in the paper with the PSNR obtained for each algorithm.

### Performance of TeFDiL with various ranks on Mushroom
To reproduce Table 2 in the paper run the `mushroomDenoisingTeFDiL.m` function three times.
- `mushroomDenoisingTeFDiL(8,'../Data/rand_state1','rnd1')`
- `mushroomDenoisingTeFDiL(8,'../Data/rand_state2','rnd2')`
- `mushroomDenoisingTeFDiL(9,'../Data/rand_state3','rnd3')`
These will produce three `.mat` files under the `Real_Experiments/Mushroom` directory once all three functions have finished running, run `getMushroomTeFDiLPSNR.m` to produce the PSNR values of TeFDiL at various ranks.

## Runtime
On our servers this job completed in 3 days for the House,Castle and Mushroom images however for the Lena image it took over 5 days to finish completely.

## Note
In order to reproduce our results for image denoising with SeDiL you will need the source code for SeDiL. We do not have permission to publicize that code therefore if you do not have it you can run the alternate function `LSRImageDenoising_noSeDiL.m`

# Synthetic Experiment{#synthetic_experiments}
This directory contains the code used to produce the results from the synthetic experiments and the Figure 2a in the paper.

## Steps to redproduce
To obtain our results we ran the `synthetic_experiments.m` file which will produce a `.mat` file called `3D_synthetic_results_25MonteCarlo.mat` after the code has finished running. Once it is finished copy the two generated `.mat` files to your local machine and run the `plot_synthetic.m` script in MATLAB which will produce a plot of the average test error for each algorithm.

## Runtime
On our servers this job completed in 3 days 21 hours and 52 minutes. However this may vary depending on your computational power. 
