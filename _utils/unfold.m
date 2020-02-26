function [X] = unfold( T, dim, i )

order = [i,1:i-1,i+1:length(dim)];
newdata = double(permute(T,order));
X = reshape(newdata,dim(i),prod(dim([1:i-1,i+1:length(dim)])));