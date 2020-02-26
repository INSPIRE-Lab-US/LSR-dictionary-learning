%This script combines 3 mat files we generated to give us a total of 25
%montecarlos and then outputs the PSNR

mushroomTeFDiL4_rep_err_OMP = zeros(25,2);
mushroomTeFDiL8_rep_err_OMP = zeros(25,2);
mushroomTeFDiL16_rep_err_OMP = zeros(25,2);
mushroomTeFDiL32_rep_err_OMP = zeros(25,2);

%Adds first 8 monte carlos
load('MushroomDenoisingTeFDiL_8_monteCarlo_rnd1.mat')

%Combining OMP variables
mushroomTeFDiL4_rep_err_OMP(1:8,:) = Rep_err_test_TeFDiL4_OMP;
mushroomTeFDiL8_rep_err_OMP(1:8,:) = Rep_err_test_TeFDiL8_OMP;
mushroomTeFDiL16_rep_err_OMP(1:8,:) = Rep_err_test_TeFDiL16_OMP;
mushroomTeFDiL32_rep_err_OMP(1:8,:) = Rep_err_test_TeFDiL32_OMP;

%Combines next 8 monte carlos
load('MushroomDenoisingTeFDiL_8_monteCarlo_rnd2.mat')

%Combining OMP variables
mushroomTeFDiL4_rep_err_OMP(9:16,:) = Rep_err_test_TeFDiL4_OMP;
mushroomTeFDiL8_rep_err_OMP(9:16,:) = Rep_err_test_TeFDiL8_OMP;
mushroomTeFDiL16_rep_err_OMP(9:16,:) = Rep_err_test_TeFDiL16_OMP;
mushroomTeFDiL32_rep_err_OMP(9:16,:) = Rep_err_test_TeFDiL32_OMP;

%Combines last 9 monte carlos
load('MushroomDenoisingTeFDiL_9_monteCarlo_rnd3.mat')
%Combining OMP variables
mushroomTeFDiL4_rep_err_OMP(17:25,:) = Rep_err_test_TeFDiL4_OMP;
mushroomTeFDiL8_rep_err_OMP(17:25,:) = Rep_err_test_TeFDiL8_OMP;
mushroomTeFDiL16_rep_err_OMP(17:25,:) = Rep_err_test_TeFDiL16_OMP;
mushroomTeFDiL32_rep_err_OMP(17:25,:) = Rep_err_test_TeFDiL32_OMP;

%Printing out PSNR values
fprintf('r=4 \t %2.4f and %2.4f \n',-10*log10(mean(mushroomTeFDiL4_rep_err_OMP)));
fprintf('r=8 \t %2.4f and %2.4f \n',-10*log10(mean(mushroomTeFDiL8_rep_err_OMP)));
fprintf('r=16 \t %2.4f and %2.4f \n',-10*log10(mean(mushroomTeFDiL16_rep_err_OMP)));
fprintf('r=32 \t %2.4f and %2.4f \n',-10*log10(mean(mushroomTeFDiL32_rep_err_OMP)));


