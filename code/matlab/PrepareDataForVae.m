function PrepareDataForVae(source_dir, target_dir,cate)
d = dir(source_dir);
if isempty(dir(['.\cube_',cate]))
    error('error')
end
% returns logical vector
isub = [d(:).isdir];
sub_folds = {d(isub).name}';
sub_folds(ismember(sub_folds,{'.','..'})) = [];
model_normalizedobj='model_normalized.obj';

for i = 1:size(sub_folds, 1)
    disp(i);
    disp(sub_folds{i});
    [v]=dir(fullfile(source_dir,sub_folds{i},model_normalizedobj));
    if ~exist(fullfile(source_dir,sub_folds{i},model_normalizedobj),'file')
%         rmdir(fullfile(source_dir,sub_folds{i}),'s')
        continue
    end
    if v.bytes>15000000
        continue
    end
    register_dir = fullfile(source_dir, sub_folds{i});
    part_list = getlabel(cate);
    %         if length(part_list)<2
    %             a=1;
    %         end
    for j = 1:length(part_list)
        %             disp(part_list(j));
        partname = part_list{j};
        if exist( fullfile(register_dir, [partname, '_reg.obj']), 'file')
            if ~exist(fullfile(target_dir, partname),'file')
                mkdir(fullfile(target_dir, partname));
            end
            copyfile(fullfile(register_dir, [partname, '_reg.obj']), fullfile(target_dir, partname, [sub_folds{i}, '_', partname, '.obj']));
            %                    copyfile(fullfile(register_dir, [partname, '_reg.obj']), fullfile(target_dir, partname, [num2str(i+1), '.obj']));
            if ~exist(fullfile(target_dir, partname, ['0_',partname,'.obj']),'file')
                if ~exist(['.\cube_',cate,'\cube_', partname,'_new.obj'],'file')
                    assert(exist(['.\cube_',cate,'\cube_std.obj'],'file')>0)
                    copyfile(['.\cube_',cate,'\cube_std.obj'], fullfile(target_dir, partname, ['0_',partname,'.obj']));
                else
                    assert(exist(['.\cube_',cate,'\cube_', partname,'_new.obj'],'file')>0)
                    copyfile(['.\cube_',cate,'\cube_', partname,'_new.obj'], fullfile(target_dir, partname, ['0_',partname,'.obj']));
                end
                if ~exist(fullfile(target_dir, [cate,'.obj']),'file')
                    copyfile( fullfile(target_dir, partname, ['0_',partname,'.obj']), fullfile(target_dir, [cate,'.obj']))
                end
            end
        end
    end
end
end