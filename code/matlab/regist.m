function regist(Dirpath, type)
%% This process takes a long time
d = dir(Dirpath);
isub = [d(:).isdir]; %# returns logical vector
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds,{'.','..'})) = [];

part_names=getlabel(type);


for p=1:length(part_names)
    parfor i = 1:size(nameFolds, 1)
        nonrigidregis([Dirpath, '\', nameFolds{i}],part_names{p})
    end
end


end