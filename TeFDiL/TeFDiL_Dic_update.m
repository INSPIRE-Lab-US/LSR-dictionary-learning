function D_TeFDiL=TeFDiL_Dic_update(Y, X, Permutation_vectors, Dictionary_sizes,r,max_iter_CP,epsilon)

%% Tensor Factorization for Dictionary Learning (TeFDiL)

% Y: observation matrix
% X: coefficient matrix
% Permutation_vectors: contains the mappings from vec(D_pi) to vec(D) and vice versa.  
% Dictionary_sizes: A 1 by 2 cell containing number of rows and columns of
% factor dictionaries, respectively.
% max_iter_CP: Maximum # of iterations of the CP decomposition algorithm


%% Preliminaries
PvT=Permutation_vectors(:,2);
m = Dictionary_sizes{1};
p=  Dictionary_sizes{2};
size_dpi = fliplr(m.*p);
K=length(m);
%% Updating D using CP decomposition of T_inv(y)

% Method 1
%X_tilde = sparse(kron(X,eye(prod(m))));
%D_vec=(X_tilde*X_tilde')\(X_tilde*Y(:));

% Method 2
gram=X*X';
D= (Y*X')/(gram+epsilon*norm(gram,'fro')*speye(size(X,1)));%for addressing ill condition cases
D_vec=D(:);


D_pi_vec=D_vec(PvT);% the vectorization of T inverse.
%doroste, vali dar vaqe in Pi*D_vec hast ba tarifi ke tu journal darim.
T_inv_y=reshape(D_pi_vec,size_dpi);% unvectorizing to find T inverse
%size(T_inv_y)
options(6)=max_iter_CP;
[Dpi_Factors]=parafac(T_inv_y,r,options);

subdictionary=cell(K,r);
D=cell(1,r);
for l=1:r
   D{l}=1; 
end

for k=1:K
   for l=1:r
       subdictionary{k,l}=reshape(Dpi_Factors{K-k+1}(:,l),m(k),p(k));
      % k,size(subdictionary{k,l})
       D{l}=kron(D{l},subdictionary{k,l});
      % k,size(D{l})
   end 
end

D_TeFDiL=zeros(prod(m),prod(p));
for l=1:r
    D_TeFDiL=D_TeFDiL+D{l};
end


D_TeFDiL=normcols(D_TeFDiL);%D_TeFDiL must have unit-norm columns
