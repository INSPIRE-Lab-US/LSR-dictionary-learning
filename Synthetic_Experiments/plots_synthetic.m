clc
clear all 
close all

%% Plottting results of 4 algorithms from Synthetic data

%Loading the mat-file that stored training and test errors for each algo

load('../Final Results/Synthetic/3D_synthetic_results_upds.mat');
%load('3D_synthetic_results_upds.mat');
%load('3D_synthetic_results_25MonteCarlo.mat');

%Getting mean test error for all 4 algorithms for all montecarlos
mean_test_err_KSVD = mean(Rep_err_test_KSVD);
mean_test_err_LS = mean(Rep_err_test_LS);
mean_test_err_TeFDiL1 = mean(Rep_err_test_TeFDiL1);
mean_test_err_STARK = mean(Rep_err_test_STARK);

N = [100 500 1000 5000 10000];

figure
% only plotting last 4 points b/c 100 samples was too small for KSVD
plot(N(2:end), mean_test_err_KSVD(2:end), '*b-', 'markers', 10, 'LineWidth', 2);
hold on
plot(N, mean_test_err_LS, 'dm-', 'markers', 10, 'LineWidth', 2);
hold on
plot(N, mean_test_err_STARK, 'ok-', 'markers', 10 , 'LineWidth', 2);
hold on
plot(N, mean_test_err_TeFDiL1, '*r-', 'markers', 10', 'LineWidth' ,2);

legend('K-SVD', 'BCD', 'STARK', 'TeFDiL');
title('KS-DL for Synthetic 3D Data');
xlim([100 10000]);
ylim([0 4.5]);
ylabel('Average Test Error');
xlabel('Training Sample Size');
set(gca, 'xscale', 'log');