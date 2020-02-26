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
% s=param.Sparsity;
r=param.TensorRank;
max_iter_CP=param.MaxIterCP;
Max_Iter_DL=param.MaxIterDL;
tol_DL=param.TolDL;
epsilon=param.epsilon;%regularization to address ill conditioned X
% SparseCodingMethod=param.SparseCodingMethod;
X=[];% Initialization for FISTA and SPAMS

%% Algorithm

% tic
for  iter = 1:Max_Iter_DL
    %Compressed Sensing Step
    
%     if strcmp(SparseCodingMethod,'OMP')
% 
%         X=OMP(D_init,Y,s);
%         
%     elseif strcmp(SparseCodingMethod,'FISTA')
%         ParamFISTA.lambda=param.lambdaFISTA;
%         ParamFISTA.max_iter=param.MaxIterFISTA;
%         ParamFISTA.tol=param.TolFISTA;
%         
%         X = fista_lasso(Y, D_init, X, ParamFISTA);
%     elseif strcmp(SparseCodingMethod,'SPAMS')
%         ParamSPAMS.lambda     = param.lambdaSPAMS;
%         ParamSPAMS.lambda2    = 0;
%         %ParamSPAMS.numThreads = 1;
%         ParamSPAMS.mode       = 2;
%         
%         X = mexLasso(Y, D_init, ParamSPAMS);
%     %elseif strcmp(SparseCodingMethod,'SPARSA')        
%     else
%         disp('Sparse coding is performed by the default method (OMP)')
%         X=OMP(D_init,Y,s);
%     end
    
%     disp('omp:')
%     toc
    X = SparseCoding(Y,D_init,paramSC);
   
    % TeFDiL (Dictionary Update Step)
%     tic
    D_TeFDiL=TeFDiL_Dic_update(Y, X, Permutation_vectors, Dictionary_sizes,r,max_iter_CP,epsilon);
    
    Reconst_error(iter)=norm(Y-D_TeFDiL*X,'fro');
    
    if iter>1 && abs(Reconst_error(iter)-Reconst_error(iter-1))<tol_DL
        %    abs(Reconst_error(iter)-Reconst_error(iter-1))
        break
    end
    
    D_init=D_TeFDiL;

%     disp('dic Update:')
%     toc
end

% X_TeFDiL = OMP(D_TeFDiL, Y,s);
X_TeFDiL = X;