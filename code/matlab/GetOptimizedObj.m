% recon_inter_random_dir: contain all parts dir
% type: plane, knife and so on
% recon_inter_random: 1 represent recon, 2 represent inter, 3 represent random
% use_struct: struc_part or part
function GetOptimizedObj(recon_inter_random_dir, type, recon_inter_random, use_struct, use_origin)
    part_names = getlabel(type);
    
    mat_prefix = '';
    if 1 == recon_inter_random
        mat_prefix = 'recover';
    elseif 2 == recon_inter_random
        mat_prefix = 'inter';
    elseif 3 == recon_inter_random
        mat_prefix = 'random';
    else
        return;
    end
    
    n = size(part_names, 2);
    if 0 == use_origin && 0 == use_struct
        merge_dir = fullfile(recon_inter_random_dir, 'merge');
    elseif 0 == use_origin && 1 == use_struct
        merge_dir = fullfile(recon_inter_random_dir, 'merge_struct');
    elseif 1 == use_origin && 1 == use_struct
        merge_dir = fullfile(recon_inter_random_dir, 'merge_struct_origin');
    elseif 1 == use_origin && 0 == use_struct
        merge_dir = fullfile(recon_inter_random_dir, 'merge_origin');
    end

    if ~exist(merge_dir, 'dir')
        mkdir(merge_dir);
    end
    
    if 0 == use_struct
        part_dirs = fullfile(recon_inter_random_dir, part_names);
    else
        part_dirs = fullfile(recon_inter_random_dir, cellfun(@(x) ['struc_',x] ,part_names, 'UniformOutput',false));
    end
    
    if 0 == use_origin
        mat = load(fullfile(recon_inter_random_dir, [mat_prefix, '_sym.mat']));
    else
        % for debug
        mat = load(fullfile(recon_inter_random_dir, '..', '..', [type, '_vaefeature.mat']));
    end

    max_num = 0;
    max_dir_files = {};
    for i = 1:n
        sub_dir = fullfile(recon_inter_random_dir, part_names{i});
        sub_dir_files = dir(fullfile(sub_dir, ['*_', part_names{i}, '.obj']));
        sub_num = size(sub_dir_files, 1);
        if sub_num > max_num
            max_num = sub_num;
            max_dir_files = sub_dir_files;
        end
    end
    max_dir_names = {max_dir_files.name};
    [max_dir_names, ~] = sort_nat(max_dir_names);
    
    if 1 == use_origin
        max_dir_names = mat.modelname;
    else
        
    end


    for i = 1:length(max_dir_names)
        if i <= 2
            continue;
        end
        id = max_dir_names{i};
        splitparts = strsplit(id, '_');
        id = splitparts{1}(1:end);
        disp(id);
        if 0 == use_origin
            code = reshape(mat.symmetry_feature(i, :), n, 2*n+9);
        else
            code = squeeze(mat.symmetryf(i, :, :));
        end
        code = AdjustOutputEncode(code);
        
        part_pcs = cell(n, 1);
        part_faces = cell(n, 1);
        for j = 1:n
            obj_filename = fullfile(part_dirs{j}, [id, '_', part_names{j}, '.obj']);
            if exist(obj_filename, 'file')
                if ispc
                    [part_pc, part_face] = cotlp(obj_filename);
                else
                    [part_pc, part_face] = readOBJ(obj_filename);
                    part_face = part_face';
                end
                part_pcs{j} = part_pc;
                part_faces{j} = part_face';
            end
        end
        tic;
        [total_pc, total_face, total_optimized_pc, total_optimized_face] = ReconstructFromCodeMixIntegerReadingObjinAdvance(code, part_pcs, part_faces, type);
        toc;
        if ispc
            SaveObjT(fullfile(merge_dir, ['optimized_', id, '.obj']), total_optimized_pc', total_optimized_face');
            SaveObjT(fullfile(merge_dir, ['unoptimized_', id, '.obj']), total_pc', total_face');
        else
            SaveObj(fullfile(merge_dir, ['optimized_', id, '.obj']), total_optimized_pc', total_optimized_face');
            SaveObj(fullfile(merge_dir, ['unoptimized_', id, '.obj']), total_pc', total_face');
        end
    end
end