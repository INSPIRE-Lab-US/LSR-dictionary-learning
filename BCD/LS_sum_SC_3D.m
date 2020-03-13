%% BCD Algorithm for rank > 1
function [D,X_vec] = LS_sum_SC_3D(Y_vec,r,paramSC,maxItr,D_k)
    %r=rank of rearrangement tensor, number of terms in summation
    m = [size(D_k{1,1},1), size(D_k{1,2},1),size(D_k{1,3},1)];
    p = [size(D_k{1,1},2), size(D_k{1,2},2),size(D_k{1,3},2)];
    N = size(Y_vec,2); %sample size
    Y_tns = reshape(Y_vec,m(1),m(2),m(3),N);


    for itr = 1: maxItr
        D = zeros(prod(m),prod(p));
        for i=1:r
            D = D + kron(D_k{i,3},kron(D_k{i,2},D_k{i,1}));
        end
        D = normc(D);
        X_vec = full(SparseCoding(Y_vec,D,paramSC));

        X_tns = reshape(X_vec, p(1),p(2),p(3),N);

        Y_1 =zeros(m(1),m(2)*m(3)*N);
        Y_2 =zeros(m(2),m(1)*m(3)*N);
        Y_3 =zeros(m(3),m(1)*m(2)*N);

        pinv_mtr_A1 = zeros(p(1),m(2)*m(3)*N);
        pinv_mtr_A2 = zeros(p(1),m(2)*m(3)*N);

        pinv_mtr_B1 = zeros(p(2),m(1)*m(3)*N);
        pinv_mtr_B2 = zeros(p(2),m(1)*m(3)*N);

        pinv_mtr_C1 = zeros(p(3),m(1)*m(2)*N);
        pinv_mtr_C2 = zeros(p(3),m(1)*m(2)*N);

        addit_term_A1 = zeros(m(1),m(2)*m(3)*N);
        addit_term_A2 = zeros(m(1),m(2)*m(3)*N);

        addit_term_B1 = zeros(m(2),m(1)*m(3)*N);
        addit_term_B2 = zeros(m(2),m(1)*m(3)*N);

        addit_term_C1 = zeros(m(3),m(1)*m(2)*N);
        addit_term_C2 = zeros(m(3),m(1)*m(2)*N);

        for n=1:N
            Y_1(:,m(2)*m(3)*(n-1)+1:m(2)*m(3)*n) = unfold(Y_tns(:,:,:,n),m,1);
            Y_2(:,m(1)*m(3)*(n-1)+1:m(1)*m(3)*n) = unfold(Y_tns(:,:,:,n),m,2);
            Y_3(:,m(1)*m(2)*(n-1)+1:m(1)*m(2)*n) = unfold(Y_tns(:,:,:,n),m,3);

            %r=1 
            term1 = unfold(X_tns(:,:,:,n),p,1)*(kron(D_k{1,3},D_k{1,2}))';
            term2 = unfold(X_tns(:,:,:,n),p,2)*(kron(D_k{1,3},D_k{1,1}))';
            term3 = unfold(X_tns(:,:,:,n),p,3)*(kron(D_k{1,2},D_k{1,1}))';

            pinv_mtr_A1(:,m(2)*m(3)*(n-1)+1:m(2)*m(3)*n) = term1;
            pinv_mtr_B1(:,m(1)*m(3)*(n-1)+1:m(1)*m(3)*n) = term2;
            pinv_mtr_C1(:,m(1)*m(2)*(n-1)+1:m(1)*m(2)*n) = term3;

            addit_term_A1(:,m(2)*m(3)*(n-1)+1:m(2)*m(3)*n) = D_k{1,1}*term1;
            addit_term_B1(:,m(1)*m(3)*(n-1)+1:m(1)*m(3)*n) = D_k{1,2}*term2;
            addit_term_C1(:,m(1)*m(2)*(n-1)+1:m(1)*m(2)*n) = D_k{1,3}*term3;

            %r= 2
            term12 = unfold(X_tns(:,:,:,n),p,1)*(kron(D_k{2,3},D_k{2,2}))';
            term22 = unfold(X_tns(:,:,:,n),p,2)*(kron(D_k{2,3},D_k{2,1}))';
            term32 = unfold(X_tns(:,:,:,n),p,3)*(kron(D_k{2,2},D_k{2,1}))';

            pinv_mtr_A2(:,m(2)*m(3)*(n-1)+1:m(2)*m(3)*n) = term12;
            pinv_mtr_B2(:,m(1)*m(3)*(n-1)+1:m(1)*m(3)*n) = term22;
            pinv_mtr_C2(:,m(1)*m(2)*(n-1)+1:m(1)*m(2)*n) = term32;        

            addit_term_A2(:,m(2)*m(3)*(n-1)+1:m(2)*m(3)*n) = D_k{2,1}*term12;
            addit_term_B2(:,m(1)*m(3)*(n-1)+1:m(1)*m(3)*n) = D_k{2,2}*term22;
            addit_term_C2(:,m(1)*m(2)*(n-1)+1:m(1)*m(2)*n) = D_k{2,3}*term32;

        end

        D_k{1,1} = normc((Y_1- addit_term_A2)*pinv(pinv_mtr_A1));
        D_k{2,1} = normc((Y_1- addit_term_A1)*pinv(pinv_mtr_A2));

        D_k{1,2} = normc((Y_2-addit_term_B2)*pinv(pinv_mtr_B1));
        D_k{2,2} = normc((Y_2-addit_term_B1)*pinv(pinv_mtr_B2));

        D_k{1,3} = normc((Y_3-addit_term_C2)*pinv(pinv_mtr_C1));
        D_k{2,3} = normc((Y_3-addit_term_C1)*pinv(pinv_mtr_C2));

    end

    D =zeros(prod(m),prod(p));
    for i=1:r
        D = D + kron(D_k{i,3},kron(D_k{i,2},D_k{i,1}));
    end
    D = normc(D);
end