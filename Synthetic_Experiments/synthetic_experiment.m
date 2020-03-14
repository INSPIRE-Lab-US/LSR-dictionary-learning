% Y: observation matrix
% D: dictionary
% X: coefficient matrix
% lambda: regularization term
% K: tensor order
% gamma: Augmented Lagrangian multiplier
% Objective functions
%F_D = (1/2)*norm(Y-D*X,'fro')^2 + (lambda/N)*sum([unfold(W_1,1),unfold(W_2,2),unfold(W_3,3)]);

%% Importing and setup
clc;
clear;

%Adding path for utility functions and external algos
addpath('../_utils');
addpath(genpath('../PARAFAC'));
addpath(genpath('../FISTA-SPAMS'));

%Adding path to main algorithms
addpath('../STARK');
addpath('../TeFDiL');
addpath('../BCD');

%Loading a random seed for our random generator
S = load('../Data/rand_state1.mat');
rng(S);
%% Generating sparse coefficient matrix
m_1=4;
m_2=5;
m_3=6;

p_1=6;
p_2=8;
p_3=10;

Dictionary_sizes{1}=[m_1 m_2 m_3];%We want D to be overcomplete
Dictionary_sizes{2}=[p_1 p_2 p_3];

disp('updated')
m=prod(Dictionary_sizes{1});
p=prod(Dictionary_sizes{2});

s = ceil(p/20); %sparsity
% Sample size
N_total = 30000;

%p x N_total matrix
X_entire = [zeros(p-s,N_total);randn(s,N_total)];
for ii=1:N_total
    X_entire(:,ii) = X_entire(randperm(p),ii);
end

[Permutation_vector, Permutation_vectorT]=permutation_vec(Dictionary_sizes);
Permutation_vectors=[Permutation_vector, Permutation_vectorT];

%% Parameters
sample_sizes=[100 500 1000 5000 10000];

%Number of monte carlo simulations
N_montcarl = 25;
N_sample = length(sample_sizes);

%Training and testing error for each monte carlo simulation for each algorithm
Rep_err_train_KSVD = zeros(N_montcarl,N_sample);
Rep_err_train_LS = zeros(N_montcarl,N_sample);
Rep_err_train_STARK = zeros(N_montcarl,N_sample);
Rep_err_train_TeFDiL1 = zeros(N_montcarl,N_sample);

Rep_err_test_KSVD = zeros(N_montcarl,N_sample);
Rep_err_test_LS = zeros(N_montcarl,N_sample);
Rep_err_test_STARK = zeros(N_montcarl,N_sample);
Rep_err_test_TeFDiL1 = zeros(N_montcarl,N_sample);

% Common Dictionary Learning Parameters
Max_Iter_DL =50;
tol_DL=10^(-4);

% KSVD Parameters
ParamKSVD.L = s;   % number of elements in each linear combination.
ParamKSVD.K = p; % number of dictionary elements
ParamKSVD.numIteration = Max_Iter_DL; % number of iteration to execute the K-SVD algorithm.
ParamKSVD.memusage = 'high';
ParamKSVD.exact = 1;
ParamKSVD.errorFlag = 0;
ParamKSVD.preserveDCAtom = 0;
ParamKSVD.displayProgress = 0;

% STARK Parameters
ParamSTARK.TolADMM = 1e-4; %tolerance in ADMM update
ParamSTARK.MaxIterADMM = 5;
ParamSTARK.DicSizes=Dictionary_sizes;
ParamSTARK.Sparsity=s;
ParamSTARK.MaxIterDL=Max_Iter_DL;
ParamSTARK.TolDL=tol_DL;

% TeFDiL Parameters
ParamTeFDiL.MaxIterCP=50;
ParamTeFDiL.DicSizes=Dictionary_sizes;
ParamTeFDiL.Sparsity=s;
ParamTeFDiL.MaxIterDL=Max_Iter_DL;
ParamTeFDiL.TolDL=tol_DL;
ParamTeFDiL.epsilon=0.01; %to improve the condition number of XX^T. multiplied by its frobenious norm.

% paramSC is input to all DL algorithms for sparse coding
paramSC.s = s; %defined as ceil(p/20)
paramSC.lambdaFISTA = .1;
paramSC.MaxIterFISTA = 10;
paramSC.TolFISTA = 1e-6;
paramSC.lambdaSPAMS = 1;
paramSC.SparseCodingMethod= 'OMP'; %choose between OMP, SPAMS and FISTA.

%% Dictionary Learning Algorithm

num_test_data = 1000;

for mont_crl = 1:N_montcarl
    mont_crl
    X_entire_rand = X_entire(:,randperm(N_total));
    X_test = X_entire_rand(:,1:num_test_data);
    
    %Generating coordinate dictionaries
    D_1 = normcols(randn(m_1,p_1));
    D_2 = normcols(randn(m_2,p_2));
    D_3 = normcols(randn(m_3,p_3));
    
    D =kron(kron(D_1,D_2), D_3);
    
    Y_test = D*X_test;
    n_cnt=0;
    
    % Iterating through each sample size
    for N = sample_sizes
        n_cnt=n_cnt+1;
        X = X_entire(:,num_test_data+1:num_test_data+N);
        Y = D*X;

        D_init_k={1,3};
        
        ms = fliplr(Dictionary_sizes{1});
        ps = fliplr(Dictionary_sizes{2});
        Y_tns = reshape(Y,ms(1),ms(2),ms(3),N);
        for k=1:3
            D_initk = unfold(Y_tns,size(Y_tns),k);
            cols_k = randperm(N*prod(ms)/ms(k),ps(k));
            D_init_k{k} = normcols(D_initk(:,cols_k));
        end
        D_init = kron(D_init_k{3},kron(D_init_k{2},D_init_k{1}));
        
        %% KSVD
        if N >= p
            tic
            disp('Training unstructured dictionary using K-SVD')
            % Initial Dictionary
            ParamKSVD.InitializationMethod = 'DataElements';
            %Algorithm
            [D_KSVD,output] = KSVD(Y,ParamKSVD,paramSC);
            toc
            
            %Dictionary training Representation Error
            Rep_err_train_KSVD(mont_crl,n_cnt) = norm(Y - D_KSVD*output.CoefMatrix,'fro')^2/N;
            
            %Dictionary test Representation Error
            X_test_KSVD = SparseCoding(Y_test,D_KSVD,paramSC);
            Rep_err_test_KSVD(mont_crl,n_cnt) = norm(Y_test - D_KSVD*X_test_KSVD,'fro')^2/num_test_data;
        else
            disp('Insufficient number of training samples for K-SVD')
        end
        %% LS Updates
          tic
          disp('Training structured dictionary using LS')
          [D_LS,X_train_LS] = LS_SC_3D(Y,paramSC,Max_Iter_DL,D_init_k{1},D_init_k{2},D_init_k{3});
          toc
          %Dictionary Representation training Error with OMP
          Rep_err_train_LS(mont_crl,n_cnt) = norm(Y - D_LS*X_train_LS,'fro')^2/N;
          
          %Dictionary Representation test Error with OMP
          X_test_LS = SparseCoding(Y_test,D_LS,paramSC);
          Rep_err_test_LS(mont_crl,n_cnt) = norm(Y_test - D_LS*X_test_LS,'fro')^2/num_test_data;

       %% STARK
        tic
        disp('Training structured dictionary using STARK')
        l_cnt=0;
        lambdaADMM=norm(Y,'fro')^(1.5)/9; %[norm(Y,'fro')^(1.5)/9 9*norm(Y,'fro')^(0.5) norm(Y)^2/7]
        gammaADMM =lambdaADMM/5; % [lambda/5 lambda] %Lagrangian parameters
        l_cnt=l_cnt+1;
      
        ParamSTARK.lambdaADMM=lambdaADMM;
        ParamSTARK.gammaADMM=gammaADMM;
      
        [D_STARK,X_STARK,Reconst_error] =STARK(Y, Permutation_vectors, D_init, ParamSTARK,paramSC);
        toc
        %Dictionary training Representation Error
        Rep_err_train_STARK(mont_crl,n_cnt)=norm(Y-D_STARK*X_STARK,'fro')^2/N;
        
        %Dictionary test Representation Error
        X_test_STARK = SparseCoding(Y_test,D_STARK,paramSC);
        Rep_err_test_STARK(mont_crl,n_cnt) = norm(Y_test - D_STARK*X_test_STARK,'fro')^2/num_test_data;
      
        %% TeFDiL
         tic
         disp('Training structured dictionary using TeFDiL (rank1)')
         ParamTeFDiL.TensorRank=1;
         [D_TeFDiL, X_TeFDiL, Reconst_error_TeFDiL]= TeFDiL(Y,Permutation_vectors, D_init, ParamTeFDiL,paramSC);
         toc
         %Dictionary training Representation Error
         Rep_err_train_TeFDiL1(mont_crl,n_cnt)=norm(Y-D_TeFDiL*X_TeFDiL,'fro')^2/N;
         
         %Dictionary test Representation Error
         X_test_TeFDiL = SparseCoding(Y_test,D_TeFDiL,paramSC);
         Rep_err_test_TeFDiL1(mont_crl,n_cnt) = norm(Y_test - D_TeFDiL*X_test_TeFDiL,'fro')^2/num_test_data;
        
        %Saves all training and test error in 3D_synthetic_results_upds.mat
        save('3D_synthetic_results_25MonteCarlo',...
             'Dictionary_sizes',...
             'Rep_err_train_KSVD','Rep_err_test_KSVD',...
             'Rep_err_train_LS','Rep_err_test_LS',...
             'Rep_err_train_STARK','Rep_err_test_STARK',...
             'Rep_err_train_TeFDiL1','Rep_err_test_TeFDiL1') 
    end
end
