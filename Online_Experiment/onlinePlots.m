%% Plots the data obtained from Online Denoising Algo on House
load('House_online_rnd1.mat')
PSNR_error_ODL1 = PSNR_error_ODL;
PSNR_error_OLS1 = PSNR_error_OLS;
load('House_online_rnd2.mat')
t=[50*(1:4), 100*(2:20),2000*2];
mean_PSNR_ODL = zeros(length(t),2);
for i=1:2
    PSNR_ODL = [];
    for j=1:length(PSNR_error_ODL1)
       PSNR_ODL = [PSNR_ODL; PSNR_error_ODL1{j,i}]; 
    end
    for j=1:length(PSNR_error_ODL)
       PSNR_ODL = [PSNR_ODL; PSNR_error_ODL{j,i}]; 
    end

    mean_PSNR_ODL(:,i) = mean(PSNR_ODL);
end

mean_PSNR_OLS = zeros(length(t),2);
for i=1:2
    PSNR_OLS = [];
    for j=1:length(PSNR_error_OLS1)
       PSNR_OLS = [PSNR_OLS; PSNR_error_OLS1{j,i}]; 
    end
    for j=1:length(PSNR_error_OLS)
       PSNR_OLS = [PSNR_OLS; PSNR_error_OLS{j,i}]; 
    end
    mean_PSNR_OLS(:,i) = mean(PSNR_OLS);
end

PSNR_ODL_f = -10*log10(mean_PSNR_ODL);
PSNR_OLS_f = -10*log10(mean_PSNR_OLS);

figure
plot(t(5:end),PSNR_ODL_f(5:end,1),'b', 'markers', 10, 'LineWidth', 2)
hold on
plot(t(5:end),PSNR_OLS_f(5:end,1),'r', 'markers', 10, 'LineWidth', 2)
hold on
plot(t(5:end),PSNR_ODL_f(5:end,2),'b--', 'markers', 10, 'LineWidth', 2)
hold on
plot(t(5:end),PSNR_OLS_f(5:end,2),'r--', 'markers', 10, 'LineWidth', 2)

xlim([200,4000])
xlabel('Number of Observed Samples', 'FontSize', 18, 'FontName', 'Time New Romans')
ylabel('PSNR', 'FontSize', 18, 'FontName', 'Time New Romans')
lh = legend('Online DL \sigma=10','OSubDiL \sigma=10','Online DL \sigma=50','OSubDiL \sigma=50','Location','East');
title('House Denoising using Online Algorithms')
set(gca,'fontsize',15)
print -dpng -r300 PSNR_err_compare_house_50montecarl_online.png