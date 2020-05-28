%% Online (Unstructed) Dictionary Learning
function [D,error_traject, PSNR_traject] = DL_online(Y,Y_test,Y_clean,paramSC,D,t_eval,...
    dim1,dim2,input_data,N_blocks,dim2_block,stride,patch_size,N_freq)

    N=size(Y,2);
    [m,p] = size(D);

    error_traject = [];
    PSNR_traject =[];

    A = zeros(p,p);
    B= zeros(m,p);

    for n=1:N
        if n==1
            X_out = OMP(D,Y_test,paramSC.s);
            Y_OMP = D*X_out;
            error_traject = [error_traject,norm(Y_test - Y_OMP,'fro')^2/numel(Y_test)];

            [Image_out_OMP, cnt_LS] = ImageRecon(input_data,N_blocks,dim2_block,stride,patch_size,Y_OMP,N_freq);
            PSNR_traject = [PSNR_traject, norm(reshape(input_data-Image_out_OMP./cnt_LS,dim1,dim2*N_freq),'fro')^2/numel(input_data)];

        end
        ind=n;
        y = Y(:,ind);
        x = SparseCoding(y,D,paramSC);

        A = A + x*x';
        B = B + y*x';

        for j=1:p
            if A(j,j) ~= 0
                u = (B(:,j) - D*A(:,j))/A(j,j)  + D(:,j);

                D(:,j) = u/max(1,norm(u));
            end
        end

        if (mod(n,t_eval) == 0 && n<=200) || (mod(n,100) == 0 && n>200 && n<2000) ...
                || (mod(n,2000) == 0 && n>=2000)
            X_out = OMP(D,Y_test,paramSC.s);
            Y_OMP = D*X_out;
            error_traject = [error_traject, norm(Y_clean - Y_OMP,'fro')^2/numel(Y_clean)];

            [Image_out_OMP, cnt_LS] = ImageRecon(input_data,N_blocks,dim2_block,stride,patch_size,Y_OMP,N_freq);
            PSNR_traject = [PSNR_traject, norm(reshape(input_data-Image_out_OMP./cnt_LS,dim1,dim2*N_freq),'fro')^2/numel(input_data)];

        end
    end
end
