addpath('../')
%Calls function to concatenate total monte carlo runs
PSNR = getImagePSNR(["HouseDenoising_8_rnd1.mat","HouseDenoising_8_rnd2.mat","HouseDenoising_9_rnd3.mat"]);


