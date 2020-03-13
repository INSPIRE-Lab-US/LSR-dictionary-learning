%% BCD Algorithm
function [D,X_vec] = LS_SC_3D(Y_vec,paramSC,maxItr,A_init,B_init,C_init)
    m = [size(A_init,1), size(B_init,1),size(C_init,1)];
    p = [size(A_init,2), size(B_init,2),size(C_init,2)];
    N = size(Y_vec,2); %sample size
    Y_tns = reshape(Y_vec,m(1),m(2),m(3),N);

    %changing order of A and B (already done for init in the function)
    A = A_init;
    B = B_init;
    C = C_init;

    for itr = 1: maxItr
        X_vec = full(SparseCoding(Y_vec,kron(C,kron(B,A)),paramSC));
        X_tns = reshape(X_vec, p(1),p(2),p(3),N);
        Y_1 =zeros(m(1),m(2)*m(3)*N);
        Y_2 =zeros(m(2),m(1)*m(3)*N);
        Y_3 =zeros(m(3),m(1)*m(2)*N);

        pinv_mtr_A = zeros(p(1),m(2)*m(3)*N);
        pinv_mtr_B = zeros(p(2),m(1)*m(3)*N);
        pinv_mtr_C = zeros(p(3),m(1)*m(2)*N);

        for n=1:N
            Y_1(:,m(2)*m(3)*(n-1)+1:m(2)*m(3)*n) = unfold(Y_tns(:,:,:,n),m,1);
            Y_2(:,m(1)*m(3)*(n-1)+1:m(1)*m(3)*n) = unfold(Y_tns(:,:,:,n),m,2);
            Y_3(:,m(1)*m(2)*(n-1)+1:m(1)*m(2)*n) = unfold(Y_tns(:,:,:,n),m,3);

            pinv_mtr_A(:,m(2)*m(3)*(n-1)+1:m(2)*m(3)*n) = unfold(X_tns(:,:,:,n),p,1)*(kron(C,B))';
            pinv_mtr_B(:,m(1)*m(3)*(n-1)+1:m(1)*m(3)*n) = unfold(X_tns(:,:,:,n),p,2)*(kron(C,A))';
            pinv_mtr_C(:,m(1)*m(2)*(n-1)+1:m(1)*m(2)*n) = unfold(X_tns(:,:,:,n),p,3)*(kron(B,A))';      
        end
        A = normc(Y_1*pinv(pinv_mtr_A));
        B = normc(Y_2*pinv(pinv_mtr_B));
        C = normc(Y_3*pinv(pinv_mtr_C));
    end
    D = kron(C,kron(B,A));
end
