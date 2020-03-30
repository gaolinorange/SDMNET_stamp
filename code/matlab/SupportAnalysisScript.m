function SupportAnalysisScript(divide_dir, type)
    d = dir(divide_dir);
    isub = [d(:).isdir];
    name_folds = {d(isub).name}';
    name_folds(ismember(name_folds,{'.','..'})) = [];
    
    for i = 1:size(name_folds, 1)
        SupportAnalysis(fullfile(divide_dir, name_folds{i}), type);
    end
end