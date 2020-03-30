function [ v, f, n, L, M, VV, CotWeight,Laplace_Matrix,L_unweight,b_E,e2v] = cotlp( filename,K)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
if nargin==1
    K=3;
end
[v, f, n, II, JJ, SS, AA, vv, cotweight,laplace_matrix,b_E,e2v,L_unweight] = meshlp(filename,K);
if nargout==1
    v = v';
elseif nargout==3
    v=v';
    n = n';
else
    v=v';
    n = n';
    W=sparse(II, JJ, SS);
    L=W;
    A=AA;
    Atmp = sparse(1:length(A),1:length(A),1./A);
    M=sparse(1:length(A),1:length(A),A);
    %L = sparse(diag(1./ A)) * W;
    % L = Atmp * W;
    VV=vv;
    CotWeight=cotweight';
    Laplace_Matrix=laplace_matrix';
    b_E=b_E';
    for j=1:size(b_E,1)
        for i=2:size(b_E,2)
            if b_E(j,i)==b_E(j,1)
                b_E(j,i)=-1;
            end
        end
    end
    b_E=b_E';
    b_E(b_E==-1)=[];
    b_E=reshape(b_E',5,[])';
    e2v=e2v';
end
end

