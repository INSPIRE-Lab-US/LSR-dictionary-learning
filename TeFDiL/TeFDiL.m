function [D_TeFDiL, X_TeFDiL,Reconst_error]=TeFDiL(Y, Permutation_vectors, D_init, param, paramSC)
% Tensor Factorization for Dictionary Learning 

% --Y: observation matrix
% --Permutation_vectors: containns the mappings from vec(D_pi) to vec(D) and vice versa.


%% parameters
% --Dictionary_sizes: A 1 by 2 cell containing number of rows and columns of
%factor dictionaries, respectively.
% --s: sparsity
% --MaxIter_DL: Maximum number of iterations of the DL algorithm
% --tol_DL: Reconstruction(representation) error tolerance of the DL algorithm

Dictionary_sizes=param.DicSizes;
r=param.TensorRank;
max_iter_CP=param.MaxIterCP;
Max_Iter_DL=param.MaxIterDL;
tol_DL=param.TolDL;
epsilon=param.epsilon;%regularization to address ill conditioned X
X=[];% Initialization for FISTA and SPAMS

%% Algorithm

for  iter = 1:Max_Iter_DL
    %Compressed Sensing Step
    X = SparseCoding(Y,D_init,paramSC);
   
    % TeFDiL (Dictionary Update Step)
    D_TeFDiL=TeFDiL_Dic_update(Y, X, Permutation_vectors, Dictionary_sizes,r,max_iter_CP,epsilon);
    
    Reconst_error(iter)=norm(Y-D_TeFDiL*X,'fro');
    
    if iter>1 && abs(Reconst_error(iter)-Reconst_error(iter-1))<tol_DL
        break
    end
    
    D_init=D_TeFDiL;


end
X_TeFDiL = X;