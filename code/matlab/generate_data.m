function generate_data(source_dir,target_dir,cate,partname,goodlist)

objlist=dir(source_dir);
objlist(1:2)=[];
isdir1=[objlist.isdir];
objlist=objlist(isdir1);
for i=1:length(objlist)
    if exist(fullfile(source_dir,objlist(i).name,['code.bad']),'file')
        
        source = fullfile(source_dir,objlist(i).name,['code.bad']);
        target = fullfile(source_dir,objlist(i).name,['code.mat']);
        movefile(source,target)
    end
end

% copy file 
if isempty(goodlist)
    goodlist={objlist.name};
else
    goodlist=file2cellArray(goodlist);
end
PrepareDataForVae(source_dir, target_dir, cate);
% compute acap
partlist=dir(target_dir);
partlist(1:2)=[];
isdir1=[partlist.isdir];
partlist=partlist(isdir1);
objnamecell=containers.Map();
maxobjnum=0;
I=0;

for i=1:length(partlist)

    list_bad=dir(fullfile(target_dir,partlist(i).name,'*.bad'));
    for k=1:length(list_bad)
    [~,badmatname]=fileparts(list_bad(k).name);  
    source = fullfile(target_dir,partlist(i).name,[badmatname,'.bad']);
    target = fullfile(target_dir,partlist(i).name,[badmatname,'.obj']);
    movefile(source,target)
    end
    list=dir(fullfile(target_dir,partlist(i).name,'*.obj'));
    [~,id]=sort_nat({list.name});
    list=list(id);
    objnamecell(partlist(i).name)=list;
   if length(list)>maxobjnum
       maxobjnum=length(list);
       I=i;
   end

end
%
% symmatlist=dir([sym_dir,'*.mat']);
    maxobjlist=objnamecell(partlist(I).name);
for i=1:length(partname)
    partlist(i).name=partname{i};
end

for i=1:maxobjnum

    splitparts = strsplit(maxobjlist(i).name, '_');
    index=splitparts{1};
    if i==1
        symmetryf(i,:,:)=zeros(length(objnamecell),2*length(objnamecell)+9);
        modelname{i}='0';
        continue;
    end
%     if ~exist(fullfile(sym_dir,[index,'.mat']),'file')
%         continue
%     end
    if ~exist(fullfile(source_dir,index,['code.mat']),'file')
        continue
    end
    symmat=load(fullfile(source_dir,index,['code.mat']));
%     symf=reshape(symmat.symmetry_feature,5,[]);
    symf=symmat.code;
    matuse=zeros(length(partlist),1);
    for fid=1:length(objnamecell)
        if exist(fullfile(target_dir,partlist(fid).name,[index,'_',partlist(fid).name,'.obj']),'file')
            matuse(fid)=1;
        end
    end
    if all(symf(:,1)==matuse) && ismember(index,goodlist)
        symmetryf(i,:,:)=symf;
        modelname{i}=index;
    else
        for k=1:length(objnamecell)
            if exist(fullfile(target_dir,partlist(k).name,[index,'_',partlist(k).name,'.obj']),'file')
                movefile(fullfile(target_dir,partlist(k).name,[index,'_',partlist(k).name,'.obj']),fullfile(target_dir,partlist(k).name,[index,'_',partlist(k).name,'.bad']))
            end            
        end
        movefile(fullfile(source_dir,index,['code.mat']),fullfile(source_dir,index,['code.bad']))
    end       
end

%compute acap
for i=1:length(partlist)
    ACAPOpt(fullfile(target_dir,partlist(i).name));  
%     get_vaefeature(fullfile(target_dir,partlist(i).name),partlist(i).name)
end
deleteid=sum(sum(abs(symmetryf),2),3)==0;
deleteid(1)=0;
symmetryf(deleteid,:,:)=[];
modelname(deleteid)=[];

get_vaefeatureforallpart(target_dir,modelname,symmetryf, cate)

end