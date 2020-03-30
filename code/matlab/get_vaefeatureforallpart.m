%stack vae script
function get_vaefeatureforallpart(datafolder,modelname, symmetryf, name)
partlist=dir(datafolder);
partlist(1:2)=[];
isdir1=[partlist.isdir];
partlist=partlist(isdir1);
warning off
% objlist=dir([datafolder,'\*.obj']);
% [~,i]=sort_nat({objlist.name});
% objlist = objlist(i);
vertex=[];
for i=1:length(modelname)
    for p=1:length(partlist)
        if exist(fullfile(datafolder,partlist(p).name,[modelname{i},'_',partlist(p).name,'.obj']),'file')
            [v,~]=readobjfromfile(fullfile(datafolder,partlist(p).name,[modelname{i},'_',partlist(p).name,'.obj']));
            vertex(i,p,:,:)=v;
        else
            vertex(i,p,:,:)=zeros(size(v,1),3);
        end
    end
end

firstobj=fullfile(datafolder,partlist(1).name,[modelname{1},'_',partlist(1).name,'.obj']);
[ v, f, ~, ~, L_, VVsimp, CotWeight] = cotlp(firstobj);
W = full(CotWeight);
pointnum = size(W,1);
D = zeros(pointnum);
for i = 1:pointnum
    D(i,i) = sum(W(i,:));
end
L = D-W;
recon = inv(L'*L)*L';

neighbour=zeros(size(v,1),100);
maxnum=0;
for i=1:size(VVsimp,1)
    neighbour(i,1:size(VVsimp{i,:},2))=VVsimp{i,:};
    if size(VVsimp{i,:},2)>maxnum
        maxnum=size(VVsimp{i,:},2);
    end
end
neighbour(:,maxnum+1:end)=[];

for i = 1:pointnum
    for j = 1:maxnum
        curneighbour = neighbour(i,j);
        if curneighbour == 0
            break
        end
        w = W(i,curneighbour)/2;
        vdiff(i,j,:) = w*(v(i,:)-v(curneighbour,:));
        edges1Ring(i,j) = sqrt(sum((v(i,:)-v(curneighbour,:)).^2));
        if neighbour(i,j)>0
            %                 cotweight(i,j)=CotWeight(i,neighbour2(i,j));
            cotweight(i,j)=1/length(nonzeros(neighbour(i,:)));
        end
    end
end
W1 = full(L_);
for i = 1:size(W1,1)
    for j = 1:size(W1,2)
        if W1(i,j) ~= 0
            W1(i,j)=1;
        end
    end
end

% LOGRNEW=dlmread([datafolder,'\LOGRNEW.txt']);
% S=dlmread([datafolder,'\S.txt']);
    logrling=zeros(1,pointnum,3);
    sling=zeros(1,pointnum,6);
for p=1:length(partlist)
    partfolder = fullfile(datafolder,partlist(p).name);
    objlist=dir([partfolder,'\*.obj']);
    [~,i]=sort_nat({objlist.name});
    objlist = objlist(i);
    modelname_postfix=cellfun(@(x) [x,'_',partlist(p).name,'.obj'],modelname,'UniformOutput',false);
    [~,id]=ismember(modelname_postfix,{objlist.name});
    if exist([partfolder,'\simp\FeatureMatgao.mat'],'file')
        fv=load([partfolder,'\simp\FeatureMatgao.mat']);
        LOGRNEW=fv.LOGRNEW;
        S=fv.S;
    elseif exist([partfolder,'\fv_r.mat'],'file')
        fv=load([partfolder,'\fv_r.mat']);
        LOGRNEW=fv.LOGR;
        S=fv.S;
    else
        LOGRNEW=dlmread([partfolder,'\LOGRNEW.txt']);
        S=dlmread([partfolder,'\S.txt']);
    end
    [ fmlogdr, fms ] = FeatureMap( LOGRNEW, S );
    fmlogdr=permute(reshape(fmlogdr,size(fmlogdr,1),3,pointnum),[1,3,2]);
    fms=permute(reshape(fms,size(fms,1),6,pointnum),[1,3,2]);
    fmlogdr=[fmlogdr;logrling];
    fms=[fms;sling];
    id(id==0)=size(fms,1);
    FLOGRNEW(:,p,:,:)=fmlogdr(id,:,:);
    FS(:,p,:,:)=fms(id,:,:);
end
ref_V=v;
ref_F=f';

% [ fmlogdr, fms ] = FeatureMap( fv.LOGRNEW, fv.S );
% feature = cat(2, fms, fmlogdr);
save([datafolder,'\',name,'_vaefeature.mat'],'ref_V','ref_F','FLOGRNEW','FS','L',...
    'neighbour','recon','vdiff','vertex','cotweight','W1','symmetryf','modelname','partlist','-v7.3')
end




