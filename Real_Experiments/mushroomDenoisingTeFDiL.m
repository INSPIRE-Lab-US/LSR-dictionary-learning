% This experiment investigates how various TeFDiL ranks can be used to
% recontruct the noisy mushroom image

%% Parameters:
%monte_carlos: Number of monte carlos to run
%path_to_random_state: Path to the random state found in /Data folder
%randState: Name of the rand state used (rand_state1, rand_state2,
%rand_state3)
function mushroomDenoisingTeFDiL(monte_carlos, path_to_random_state, randState)
    %Adding imports
    addpath('../_utils');
    addpath('../TeFDiL');
    addpath(genpath('../FISTA-SPAMS'));
    addpath(genpath('../PARAFAC'));

    %Loading a random seed
    S = load(path_to_random_state);
    rng(S);

    N_montcarl = monte_carlos;

    fprintf('Mushroom TeFDiL with various ranks with %d Monte Carlos',N_montcarl);

    %% Loading image and extracting overlapping patches to be used as training data
    Image = double(imread('../Data/mushroom.png'));

    %Normalize all pixel values
    input_data = double(Image)/max(max(max(Image)));
    %Finds coordinates of non-zero pixel values
    [a,b,c] = ind2sub(size(input_data), find(Image));

    input_data=input_data(min(a):max(a), min(b):max(b), :);

    dim1 = size(input_data,1);
    dim2 = size(input_data,2);

    N_freq=3; % # of frequencies in the 3rd mode
    patch_size=8;

    %cropping image for perfect tiling with no extra pixels
    input_data = input_data(1:patch_size*floor(dim1/patch_size), 1:patch_size*floor(dim2/patch_size), 1:N_freq);

    %new(cropped) dimensions
    dim1 = size(input_data,1);
    dim2 = size(input_data,2);

    %Populating block_data with overlapping 8x8x3 patches of our image
    step=3;  %the "step size" determines how much the patches overlap
    for i=0:step:dim1-patch_size
        for j=0:step:dim2-patch_size
            block_data{i/step+1, j/step+1} = input_data(i+1:i+patch_size, j+1:j+patch_size, :);
        end
    end

    %Cropping the image to exclude the extra pixels (due to the step size)
    input_data=input_data(1:i+patch_size,1:j+patch_size,1:N_freq);

    %Again rewriting value of dimensions after another crop
    dim1 = size(input_data,1);
    dim2 = size(input_data,2);

    dim1_block = size(block_data,1);
    dim2_block = size(block_data,2);

    N_blocks = dim1_block*dim2_block; %number of blocks (data points)

    obsrvtn_tensor = zeros(patch_size,patch_size,N_freq,N_blocks);
    %Flattened version of the observtn_tensor
    obsrvtn_vect = zeros(patch_size^2*N_freq,N_blocks);

    k=0;
    for i = 1:size(block_data,1)
        for j = 1:size(block_data,2)
            k = k+1;
            obsrvtn_tensor(:,:,:,k) = block_data{i,j};
            obsrvtn_vect(:,k) = reshape(obsrvtn_tensor(:,:,:,k), patch_size^2*N_freq,1,1);
        end
    end


    %Noiseless data
    Y_clean = obsrvtn_vect;

    %% Dictionary Parameters
    M = [patch_size, patch_size, N_freq];
    P = [2*patch_size, 2*patch_size, N_freq];

    Dictionary_sizes{1} = fliplr(M);
    Dictionary_sizes{2} = fliplr(P);

    m = prod(M);
    p = prod(P);

    %% Experiment Setup
    N_sigs = 2;

    Rep_err_train_TeFDiL4 = zeros(N_montcarl,N_sigs);
    Rep_err_test_TeFDiL4 = zeros(N_montcarl,N_sigs);
    Rep_err_test_TeFDiL4_OMP = zeros(N_montcarl,N_sigs);

    Rep_err_train_TeFDiL8 = zeros(N_montcarl,N_sigs);
    Rep_err_test_TeFDiL8 = zeros(N_montcarl,N_sigs);
    Rep_err_test_TeFDiL8_OMP = zeros(N_montcarl,N_sigs);

    Rep_err_train_TeFDiL16 = zeros(N_montcarl,N_sigs);
    Rep_err_test_TeFDiL16 = zeros(N_montcarl,N_sigs);
    Rep_err_test_TeFDiL16_OMP = zeros(N_montcarl,N_sigs);

    Rep_err_train_TeFDiL32 = zeros(N_montcarl,N_sigs);
    Rep_err_test_TeFDiL32 = zeros(N_montcarl,N_sigs);
    Rep_err_test_TeFDiL32_OMP = zeros(N_montcarl,N_sigs);


    [Permutation_vector, Permutation_vectorT]=permutation_vec(Dictionary_sizes);

    %Permutation_vector: vector containing the index of non-zero value of the permutation matrix in each row
    %Permutation_vectorT: contains those of the transpose of the permutation matrix
    Permutation_vectors=[Permutation_vector, Permutation_vectorT];

    %% Algorithm Parameters
    K = 3; %tensor order
    % Sparse Coding Parameters. Have to select sparse coding method here:
    % 'OMP', 'SPAMS', 'FISTA'
    % paramSC is input to all DL algorithms
    s = ceil(p/20); %sparsity level
    paramSC.s = s;
    paramSC.lambdaFISTA = .1;
    paramSC.MaxIterFISTA = 10;
    paramSC.TolFISTA = 1e-6;
    paramSC.lambdaSPAMS = 1;
    paramSC.SparseCodingMethod= 'FISTA';

    % Dictionary Learning Parameters
    Max_Iter_DL = 50;

    % TeFDiL Parameters
    ParamTeFDiL.MaxIterCP=50;
    ParamTeFDiL.DicSizes=Dictionary_sizes;
    ParamTeFDiL.MaxIterDL=Max_Iter_DL;
    ParamTeFDiL.TolDL=10^(-4);
    ParamTeFDiL.epsilon=0.01;%to impprove the condition number of XX^T. multiplied by its frobenious norm.

    sig_vals = [10 50];
    for mont_crl = 1:N_montcarl
        mont_crl
        for sigma_ind = 1:2
            sigma_ind
            sigma = sig_vals(sigma_ind);
            Y_noisy = Y_clean+sigma/max(max(max(Image)))*randn(size(Y_clean)); % noisy data

            %Generate training data
            N = length(Y_clean);
            %noisy training data
            Y_train = Y_noisy(:, randperm(N_blocks, N));
            Y_tns = reshape(Y_train,M(1),M(2),M(3),N);

            %% Initialization for r=4
            Rnk = 4;
            D_init_k = {Rnk,3};
            for k=1:3
                D_initk = unfold(Y_tns, size(Y_tns), k);
                for r=1:Rnk
                    cols_k = randperm(N*m/M(k),P(k));
                    D_init_k{r,k} = normcols(D_initk(:,cols_k));
                end
            end
            D_init4 = zeros(m,p);
            for r=1:Rnk
                D_init4 = D_init4 + kron(D_init_k{r,3}, kron(D_init_k{r,2},D_init_k{r,1}));
            end
            D_init4 = normc(D_init4);

            %% TefDil rank4
            disp('Training structured dictionary using TeFDiL (rank4)')
            ParamTeFDiL.TensorRank=Rnk;
            [D_TeFDiL4, X_TeFDiL4, Reconst_error_TeFDiL]= TeFDiL(Y_train,Permutation_vectors, D_init4, ParamTeFDiL,paramSC);
            
            %%%Dictionary test Representation Error
            X_test_TeFDiL4 = SparseCoding(Y_noisy,D_TeFDiL4,paramSC);
            Y_TeFDiL4=D_TeFDiL4*X_test_TeFDiL4;
            
            %%%Reconstructing the image from the overlapping patches
            [Image_out_TeFDiL4, cnt_TeFDiL4] = ImageRecon2(input_data,N_blocks,dim2_block,step,patch_size,Y_TeFDiL4,N_freq);
            Rep_err_test_TeFDiL4(mont_crl,sigma_ind)=norm(reshape(input_data -Image_out_TeFDiL4./cnt_TeFDiL4,dim1,dim2*N_freq),'fro')^2/numel(input_data);

            %with OMP
            X_test_TeFDiL4_OMP = OMP(D_TeFDiL4,Y_noisy,s);
            Y_TeFDiL4_OMP = D_TeFDiL4*X_test_TeFDiL4_OMP;
            [Image_out_TeFDiL4_OMP, cnt_TeFDiL4] = ImageRecon2(input_data,N_blocks,dim2_block,step,patch_size,Y_TeFDiL4_OMP,N_freq);
            Rep_err_test_TeFDiL4_OMP(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_TeFDiL4_OMP./cnt_TeFDiL4,dim1,dim2*N_freq),'fro')^2/numel(input_data);

            %% Initialization for r=8
            Rnk=8;
            D_init_k={Rnk,3};
            for k=1:3
                D_initk = unfold(Y_tns,size(Y_tns),k);
                for r=1:Rnk
                    cols_k = randperm(N*m/M(k),P(k));
                    D_init_k{r,k} = normcols(D_initk(:,cols_k));
                end
            end
            D_init8 = zeros(m,p);
            for r=1:Rnk
                D_init8 = D_init8 + kron(D_init_k{r,3},kron(D_init_k{r,2},D_init_k{r,1}));
            end
            D_init8 = normc(D_init8);

            %% TefDil rank8
            disp('Training structured dictionary using TeFDiL (rank8)')
            ParamTeFDiL.TensorRank=Rnk;
            [D_TeFDiL8, X_TeFDiL8, Reconst_error_TeFDiL]= TeFDiL(Y_train,Permutation_vectors, D_init8, ParamTeFDiL,paramSC);
            %%%Dictionary test Representation Error
            X_test_TeFDiL8 = SparseCoding(Y_noisy,D_TeFDiL8,paramSC);
            Y_TeFDiL8=D_TeFDiL8*X_test_TeFDiL8;
            %%%Reconstructing the image from the overlapping patches
            [Image_out_TeFDiL8, cnt_TeFDiL8] = ImageRecon2(input_data,N_blocks,dim2_block,step,patch_size,Y_TeFDiL8,N_freq);
            Rep_err_test_TeFDiL8(mont_crl,sigma_ind)=norm(reshape(input_data -Image_out_TeFDiL8./cnt_TeFDiL8,dim1,dim2*N_freq),'fro')^2/numel(input_data);

            %with OMP
            X_test_TeFDiL8_OMP = OMP(D_TeFDiL8,Y_noisy,s);
            Y_TeFDiL8_OMP = D_TeFDiL8*X_test_TeFDiL8_OMP;
            [Image_out_TeFDiL8_OMP, cnt_TeFDiL8] = ImageRecon2(input_data,N_blocks,dim2_block,step,patch_size,Y_TeFDiL8_OMP,N_freq);
            Rep_err_test_TeFDiL8_OMP(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_TeFDiL8_OMP./cnt_TeFDiL8,dim1,dim2*N_freq),'fro')^2/numel(input_data); 

            %% Initialization for r=16
            Rnk=16;
            D_init_k={Rnk,3};
            for k=1:3
                D_initk = unfold(Y_tns,size(Y_tns),k);
                for r=1:Rnk
                    cols_k = randperm(N*m/M(k),P(k));
                    D_init_k{r,k} = normcols(D_initk(:,cols_k));
                end
            end
            D_init16 = zeros(m,p);
            for r=1:Rnk
                D_init16 = D_init16 + kron(D_init_k{r,3},kron(D_init_k{r,2},D_init_k{r,1}));
            end
            D_init16 = normc(D_init16);

            %% TefDil rank16
            disp('Training structured dictionary using TeFDiL (rank16)')
            ParamTeFDiL.TensorRank=Rnk;
            [D_TeFDiL16, X_TeFDiL16, Reconst_error_TeFDiL]= TeFDiL(Y_train,Permutation_vectors, D_init16, ParamTeFDiL,paramSC);
            %%%Dictionary test Representation Error
            X_test_TeFDiL16 = SparseCoding(Y_noisy,D_TeFDiL16,paramSC);
            Y_TeFDiL16=D_TeFDiL16*X_test_TeFDiL16;
            %%%Reconstructing the image from the overlapping patches
            [Image_out_TeFDiL16, cnt_TeFDiL16] = ImageRecon2(input_data,N_blocks,dim2_block,step,patch_size,Y_TeFDiL16,N_freq);
            Rep_err_test_TeFDiL16(mont_crl,sigma_ind)=norm(reshape(input_data -Image_out_TeFDiL16./cnt_TeFDiL16,dim1,dim2*N_freq),'fro')^2/numel(input_data);

            %with OMP
            X_test_TeFDiL16_OMP = OMP(D_TeFDiL16,Y_noisy,s);
            Y_TeFDiL16_OMP = D_TeFDiL16*X_test_TeFDiL16_OMP;
            [Image_out_TeFDiL16_OMP, cnt_TeFDiL16] = ImageRecon2(input_data,N_blocks,dim2_block,step,patch_size,Y_TeFDiL16_OMP,N_freq);
            Rep_err_test_TeFDiL16_OMP(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_TeFDiL16_OMP./cnt_TeFDiL16,dim1,dim2*N_freq),'fro')^2/numel(input_data);

            % Initialization for r=32
            Rnk=32;
            D_init_k={Rnk,3};
            for k=1:3
                D_initk = unfold(Y_tns,size(Y_tns),k);
                for r=1:Rnk
                    cols_k = randperm(N*m/M(k),P(k));
                    D_init_k{r,k} = normcols(D_initk(:,cols_k));
                end
            end
            D_init32 = zeros(m,p);
            for r=1:Rnk
                D_init32 = D_init32 + kron(D_init_k{r,3},kron(D_init_k{r,2},D_init_k{r,1}));
            end
            D_init32 = normc(D_init32);

            %% TefDil rank32
            disp('Training structured dictionary using TeFDiL (rank32)')
            ParamTeFDiL.TensorRank=Rnk;
            [D_TeFDiL32, X_TeFDiL32, Reconst_error_TeFDiL]= TeFDiL(Y_train,Permutation_vectors, D_init32, ParamTeFDiL,paramSC);
            %%%Dictionary test Representation Error
            X_test_TeFDiL32 = SparseCoding(Y_noisy,D_TeFDiL32,paramSC);
            Y_TeFDiL32=D_TeFDiL32*X_test_TeFDiL32;
            %%%Reconstructing the image from the overlapping patches
            [Image_out_TeFDiL32, cnt_TeFDiL32] = ImageRecon2(input_data,N_blocks,dim2_block,step,patch_size,Y_TeFDiL32,N_freq);
            Rep_err_test_TeFDiL32(mont_crl,sigma_ind)=norm(reshape(input_data -Image_out_TeFDiL32./cnt_TeFDiL32,dim1,dim2*N_freq),'fro')^2/numel(input_data);

            %with OMP
            X_test_TeFDiL32_OMP = OMP(D_TeFDiL32,Y_noisy,s);
            Y_TeFDiL32_OMP = D_TeFDiL32*X_test_TeFDiL32_OMP;
            [Image_out_TeFDiL32_OMP, cnt_TeFDiL32] = ImageRecon2(input_data,N_blocks,dim2_block,step,patch_size,Y_TeFDiL32_OMP,N_freq);
            Rep_err_test_TeFDiL32_OMP(mont_crl,sigma_ind)=norm(reshape(input_data-Image_out_TeFDiL32_OMP./cnt_TeFDiL32,dim1,dim2*N_freq),'fro')^2/numel(input_data);

            %% Saving Results
            name = strcat('MushroomDenoisingTeFDiL_',num2str(monte_carlos),'_monteCarlo_',randState);
            
            save(name,'sig_vals','M','P',...                
                'Rep_err_test_TeFDiL4','Rep_err_test_TeFDiL8','Rep_err_test_TeFDiL16','Rep_err_test_TeFDiL32',...
                'Rep_err_test_TeFDiL4_OMP','Rep_err_test_TeFDiL8_OMP','Rep_err_test_TeFDiL16_OMP','Rep_err_test_TeFDiL32_OMP')
        end
    end
    mkdir('Mushroom');
    movefile(strcat(name,'.mat'), 'Mushroom');
end
