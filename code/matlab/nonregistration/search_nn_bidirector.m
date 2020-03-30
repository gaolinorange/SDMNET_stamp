function [sqrDz,P,NP]=search_nn_bidirector(Z,ZF,Y,YF,NFY)
% find the nnpoint bi-directory
[sqrDz,Iz,Cz] = point_mesh_squared_distance(Z,Y,YF);
% idz=knnsearch(Y,Cz);

[sqrDy,Iy,Cy] = point_mesh_squared_distance(Y,Z,ZF);
id_out=find(sqrDy>0.0001);
id_out=[];
if ~isempty(id_out)
    on_Source = Cy(id_out,:);
    idy=knnsearch(Z,on_Source);
    y_out=Y(id_out,:);
    Cz(idy,:)=y_out;
    sqrDz(idy,:)=sqrDy(id_out,:);
    
    P=Cz;
    NP=NFY(Iz,:);
    
    for i=1:length(id_out)
        id=find(sum(YF==id_out(i),2)==1);
        if isempty(id)
            continue;
        end
        NP(idy(i),:)=mean(NFY(id,:));
    end
else
    P = Cz;
       NP = NFY(Iz,:);
end
end