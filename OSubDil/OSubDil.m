%% Implementation of OSubDil: An Online LSR-DL Algorithm
function [D_upd,error_traject_train,error_traject,PSNR_traject] = OSubDil(Y,Y_test,Y_clean,paramSC,A_init,B_init,C_init,t_eval,...
    dim1,dim2,input_data,N_blocks,dim2_block,stride,patch_size,N_freq)

    m = [size(A_init,1), size(B_init,1),size(C_init,1)];
    p = [size(A_init,2), size(B_init,2),size(C_init,2)];

    N = size(Y,2); %number of samples

    step = 0.00001;
    batch_size = 1;
    A = A_init;
    B = B_init;
    C = C_init;

    D = {A,B,C};
    k=3; %tensor order


    for i=1:k
        Alpha{i} = zeros(p(i),p(i));
        Beta{i} = zeros(m(i),p(i));
    end

    %training errors
    error_traject_train = [];
    %test errors
    error_traject= [];
    PSNR_traject =[];

    for btc=1:N/batch_size
        Y_vec_btch = Y(:,(btc-1)*batch_size+1:btc*batch_size);

        Y_tns_btch = reshape(Y_vec_btch,m(1),m(2),m(3),batch_size);
        D_upd = kron(D{3},kron(D{2},D{1}));
        %Compute sparse representation of Y given updated KS dictionary
        X_vec_btch = full(SparseCoding(Y_vec_btch,D_upd,paramSC));
        X_tns_btch = reshape(X_vec_btch,p(1),p(2),p(3),batch_size);

        if btc==1
            X_out = OMP(D_upd,Y_test,paramSC.s);
            X_train = OMP(D_upd,Y,paramSC.s);

            Y_OMP = D_upd*X_out;

            error_traject = [error_traject,norm(Y_test - Y_OMP,'fro')^2/numel(Y_test)];
            error_traject_train= [error_traject_train,norm(Y - D_upd*X_train,'fro')^2/numel(Y)];

            [Image_out_OMP, cnt_LS] = ImageRecon(input_data,N_blocks,dim2_block,stride,patch_size,Y_OMP,N_freq);
            PSNR_traject = [PSNR_traject, norm(reshape(input_data-Image_out_OMP./cnt_LS,dim1,dim2*N_freq),'fro')^2/numel(input_data)];
        end

        for k=1:3
            if k==1
                A=D{1};
                B=D{2};
                C=D{3};
            elseif k==2
                A=D{2};
                B=D{1};
                C=D{3};
            elseif C==1
                break;
            else
                A=D{3};
                B=D{1};
                C=D{2};
            end

            kr_CB = kron(C,B);

            for n=1:batch_size
                X_flat = unfold(X_tns_btch(:,:,:,n),p,k);
                Yh_flat = unfold(Y_tns_btch(:,:,:,n),m,k);

                Alpha{k} = Alpha{k} + (1/batch_size)*X_flat*(kr_CB'*kr_CB)*X_flat';
                Beta{k} = Beta{k} + (1/batch_size)*Yh_flat*kr_CB*X_flat';
            end
            for pas=1:5
                for col_ind=1:p(k)
                    A(:,col_ind) = A(:,col_ind) +(Beta{k}(:,col_ind)-A*Alpha{k}(:,col_ind))*step;
                end
                D{k}=normc(A);
            end

        end
        D_upd = kron(D{3},kron(D{2},D{1}));
        if (mod(btc,t_eval) == 0 && btc<=200) || (mod(btc,100) == 0 && btc>200 && btc<2000) ...
                || (mod(btc,2000) == 0 && btc>=2000)

            X_out = OMP(D_upd,Y_test,paramSC.s);
            X_train = OMP(D_upd,Y,paramSC.s);

            Y_OMP = D_upd*X_out;

            error_traject = [error_traject,norm(Y_clean - Y_OMP,'fro')^2/numel(Y_clean)];
            error_traject_train = [error_traject_train,norm(Y - D_upd*X_train,'fro')^2/numel(Y)];

            [Image_out_OMP, cnt_LS] = ImageRecon(input_data,N_blocks,dim2_block,stride,patch_size,Y_OMP,N_freq);
            PSNR_traject = [PSNR_traject, norm(reshape(input_data-Image_out_OMP./cnt_LS,dim1,dim2*N_freq),'fro')^2/numel(input_data)];

        end
    end
end
