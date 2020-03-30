% path = './test';
function Optimize(interpolation_dir, category, gap)
    if strcmp(category, 'plane')
        load('./libtomlab.dll', '-mat');
    elseif strcmp(category, 'chair')
        load('./libopenmp.dll', '-mat');
    end
    part_names = gelabel(category);
    for i = 1:size(part_names)
        part_dir = fullfile(interpolation_dir, part_names{i});
        if ~exist(part_dir, 'dir')
            disp('part_dir missing');
            disp(part_dir);
        end
    end
    category = 'test';
    if ~exist(category, 'dir')
        mkdir(category);
    end
    for i = 1:gap:200
       writeOBJ(fullfile(category, [num2str(i), '.obj']), all_vfs{i, 1}, all_vfs{i, 2}); 
    end
end