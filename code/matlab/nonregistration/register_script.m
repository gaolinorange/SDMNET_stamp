% register
pathFolder = 'G:\wutong\sig2019\shapenetcore_segmentation\03624134_knif\divide_with_face30000'

d = dir(pathFolder);
isub = [d(:).isdir]; %# returns logical vector
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds,{'.','..'})) = [];
parfor i = 1:size(nameFolds, 1)
    nonrigidregis([pathFolder, '\', nameFolds{i}],'part1')
end
parfor i = 1:size(nameFolds, 1)
    nonrigidregis([pathFolder, '\', nameFolds{i}],'part2')
end


