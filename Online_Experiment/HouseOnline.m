%% Function for online algorithm experiment on House image
% Parameters:
% path_to_rand_state: The path to the random state being used
function [Reconst_error_OLS,Reconst_error_ODL]= HouseOnline(path_to_rand_state)

    S = load(path_to_rand_state);
    
    
	if(path_to_rand_state == '../Data/rand_state1')
	   name = 'House_online_rnd1';
	else
	   name = 'House_online_rnd2';
    end
    
    %Dependencies
    addpath('../_utils');
    addpath('../_OSubDil_Algo');
    addpath(genpath('../FISTA-SPAMS'));

    rng(S);
    disp('house rand_state1_10montecarlo')
    %% Loading image and extracting overlapping patches from it
    Image=double(imread('../Data/house_color.tiff'));

    input_data = double(Image)/max(max(max(Image)));
    [a,b,c]=ind2sub(size(input_data),find(Image));
    input_data=input_data(min(a):max(a),min(b):max(b),:);

    dim1=size(input_data,1);
    dim2=size(input_data,2);

    N_freq = 3;% # of frequencies (# of features in the 3rd mode)
    patch_size = 8;

    %cropping the image for perfect tiling with no extra pixels
    input_data=input_data(1:patch_size*floor(dim1/patch_size),1:patch_size*floor(dim2/patch_size),1:N_freq);

    %new (cropped) dimensions
    dim1=size(input_data,1);
    dim2=size(input_data,2);

    step=2; %Stride
    for i=0:step:dim1-patch_size% step determines how much of the neighboring patches overlap
        for j=0:step:dim2-patch_size
            block_data{i/step+1,j/step+1}=input_data(i+1:i+patch_size, j+1:j+patch_size,:);
        end
    end

    %cropping the image to exclude the extra pixels (due to step size)
    input_data=input_data(1:i+patch_size,1:j+patch_size,1:N_freq);

    dim1_block=size(block_data,1);
    dim2_block=size(block_data,2);

    N_blocks =dim1_block*dim2_block ;%number of blocks (data points)

    obsrvtn_tensor = zeros(patch_size,patch_size,N_freq,N_blocks);
    obsrvtn_vect = zeros(patch_size^2*N_freq,N_blocks);

    k=0;
    for i = 1:size(block_data,1)
        for j = 1:size(block_data,2)
            k = k+1;
            obsrvtn_tensor(:,:,:,k) = block_data{i,j};
            obsrvtn_vect(:,k) = reshape(obsrvtn_tensor(:,:,:,k),patch_size^2*N_freq,1,1);%each point is vectorized
        end
    end
    % Noisy and noiseless data
    Y_clean = obsrvtn_vect; %clean data

    %% Dictionary Parameters
    M = [patch_size, patch_size, N_freq];
    P =[2*patch_size ,2*patch_size, N_freq];

    m=prod(M);
    p=prod(P);

    %% Experiment setup
    sample_sizes=length(Y_clean);

    N_montcarl = 15;
    
    %% Algorithm Parameters
    K = 3; %tensor order
    % Sparse Coding Parameters. Have to select sparse coding method here:
    % 'OMP', 'SPAMS', 'FISTA'
    % paramSC is input to all DL algorithms
    s = ceil(p/20); %sparsity level
    paramSC.s = s;
    paramSC.lambdaFISTA = 5;
    paramSC.MaxIterFISTA = 10;
    paramSC.TolFISTA = 1e-6;
    paramSC.lambdaSPAMS = 1;
    paramSC.SparseCodingMethod= 'FISTA';

    % SPAMS parameters
    param_ODL.K=p;  % learns a dictionary with 100 elements
    param_ODL.lambda=0.1;
    param_ODL.numThreads=-1; % number of threads
    param_ODL.batchsize=1;
    param_ODL.verbose=false;
    param_ODL.iter=sample_sizes;  % let us see what happens after 1000 iterations.

    t_eval = 50;

    sig_vals = [10,50];
    
    Reconst_error_ODL = {};
    Reconst_error_OLS = {};
    PSNR_error_ODL = {};
    PSNR_error_OLS ={};

    for mont_crl = 1:N_montcarl
        mont_crl
        for sigma_ind=1:length(sig_vals)
            sigma_ind
            sigma = sig_vals(sigma_ind);
            Y_noisy=Y_clean+sigma/max(max(max(Image)))*randn(size(Y_clean)); % noisy data

            %Generate training data
            N=4000;

            % noisy training data
            Y_train = Y_noisy(:,randperm(N_blocks,N));

            Y_train_warmstart = Y_train(:,1:P(1));

            %initializing coordinate dictionaries
            D_init_k={1,3};

            Y_tns = reshape(Y_train_warmstart,M(1),M(2),M(3),P(1));
            for k=1:3
                D_initk = unfold(Y_tns,size(Y_tns),k);
                cols_k = randperm(P(1)*prod(M)/M(k),P(k));
                D_init_k{k} = normcols(D_initk(:,cols_k));
            end
            D_init = kron(D_init_k{3},kron(D_init_k{2},D_init_k{1}));

            %% Online DL
            disp('Training unstructured dictionary using Online DL')
            [~,Reconst_err_ODL,PSNR_ODL] = DL_online(Y_train,Y_noisy,Y_clean,paramSC,D_init,t_eval,...
                dim1,dim2,input_data,N_blocks,dim2_block,step,patch_size,N_freq);

            Reconst_error_ODL{mont_crl,sigma_ind} = Reconst_err_ODL;
            PSNR_error_ODL{mont_crl,sigma_ind} = PSNR_ODL;

            %% Online LS (OSubDiL)
            disp('Training structured dictionary using OSubDil')
            [~,Reconst_err_OLS_train,Reconst_err_OLS_test,PSNR_OLS] = OSubDil(Y_train,Y_noisy,Y_clean,paramSC,D_init_k{1},D_init_k{2},D_init_k{3},t_eval,...
                dim1,dim2,input_data,N_blocks,dim2_block,step,patch_size,N_freq);

            Reconst_error_OLS{mont_crl,sigma_ind}= Reconst_err_OLS_test;
            Reconst_error_OLS_train{mont_crl,sigma_ind}= Reconst_err_OLS_train;
            PSNR_error_OLS{mont_crl,sigma_ind} = PSNR_OLS;

            %% Saving Results
           step_size = 0.00001;
           save(name,'sig_vals','M','P',...
                'Reconst_error_OLS','Reconst_error_OLS_train',...
                'Reconst_error_ODL','PSNR_error_ODL','PSNR_error_OLS','step_size')             
        end
    end
end
   