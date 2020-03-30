% recon_inter_random_dir: contain all parts dir
% type: plane, knife and so on
% recon_inter_random: 1 represent recon, 2 represent inter, 3 represent random
% use_struct: struc_part or part
function GetOptimizedObj(recon_inter_random_dir, type, recon_inter_random, use_struct, use_origin)
    if strcmp(type, 'chair') == 1
        part_names = {'armrest_1', 'armrest_2', 'back', 'leg1_1', 'leg1_2', 'leg2_1', 'leg2_2', 'seat'};
    elseif strcmp(type, 'knife') == 1
        part_names = {'part1', 'part2'};
    elseif strcmp(type, 'guitar') == 1
        part_names = {'part1', 'part2', 'part3'};
    elseif strcmp(type, 'skateboard')
%         part_names = {'surface', 'bearing1', 'bearing2', 'wheel1_1', 'wheel1_2', 'wheel2_1', 'wheel2_2'};
        part_names = {'surface', 'left_front_wheel', 'right_front_wheel', 'left_behind_wheel', 'right_behind_wheel'};
    elseif strcmp(type, 'cup') == 1
        part_names = {'part1', 'part2'};
    elseif strcmp(type, 'car') == 1
        part_names = {'body', 'left_front_wheel', 'right_front_wheel', 'left_behind_wheel', 'right_behind_wheel'};
    elseif strcmp(type, 'plane') == 1
        part_names = {'body', 'left_wing', 'right_wing', 'left_tail', 'right_tail', 'upper_tail', 'down_tail', 'front_landing_gear', 'left_landing_gear', 'right_landing_gear', 'left_engine_1', 'right_engine_1', 'left_engine_2', 'right_engine_2'};
    elseif strcmp(type, 'table') == 1
        part_names = {'surface', 'leg1_1', 'leg1_2', 'leg2_1', 'leg2_2', 'leg3_1', 'leg3_2', 'leg4_1', 'leg4_2'};
    end
    
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
        [status,msg,msgID] = mkdir(merge_dir);
    end
    
    if 0 == use_struct
        part_dirs = fullfile(recon_inter_random_dir, part_names);
    else
        part_dirs = fullfile(recon_inter_random_dir, cellfun(@(x) ['struc_',x] ,part_names, 'UniformOutput',false));
    end
    if 0 == use_struct && 0 == use_origin
        mat0 = load(fullfile(recon_inter_random_dir, [type, '_vaefeature.mat']));
    end
    if 0 == use_origin
        mat = load(fullfile(recon_inter_random_dir, [mat_prefix, '_sym.mat']));
    else
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


    for i = 1:33:length(max_dir_names)
%         if 1 ~= i
%             continue;
%         end
        if recon_inter_random == 2
%             if 0 ~= mod(i, 10)
%                 continue;
%             end
%               if 199 ~= i
%                   continue;
%               end
        end
%         if i < 100
%             continue;
%         end
        
        id = max_dir_names{i};
        splitparts = strsplit(id, '_');
        id = splitparts{1}(1:end);
%         if 0 == strcmp(id, 'shinxg')
%             continue;
%         end
        disp(id);
        if 0 == use_origin
            code = reshape(mat.symmetry_feature(i, :), n, 2*n+9);
        else
            code = squeeze(mat.symmetryf(i, :, :));
        end
        code = AdjustOutputEncode(code);
        structure = mat0.structure(i, :);
        part_pcs = cell(n, 1);
        part_faces = cell(n, 1);
        for j = 1:n
            obj_filename = fullfile(part_dirs{j}, [id, '_', part_names{j}, '.obj']);
%             disp(obj_filename);
            if exist(obj_filename, 'file')
                if ispc
                    [part_pc, part_face] = cotlp(obj_filename);
                else
                    [part_pc, part_face] = readObj(obj_filename);
                    part_face = part_face';
                end
                part_pcs{j} = part_pc;
                part_faces{j} = part_face';
            end
        end
        if 1 == strcmp(type, 'plane')
            part_pcs{3} = part_pcs{2} .* [-1, 1, 1];
            % for j = 11:14
            %     part_pcs{j} = part_pcs{j} + [0, 0, i/200*0.25];
            % end
            % part_pcs{13} = part_pcs{13} - [i/200*0.2, 0, 0];
%             part_pcs{14} = part_pcs{14} + [i/200*0.12, 0, 0];
            part_pcs{14} = part_pcs{14} + [0, 0, -0.02];
            if i >= 150
                part_pcs{14} = part_pcs{14} + [i/200*0.2, -0.05, 0];
            end
            part_pcs{13} = part_pcs{14} .* [-1, 1, 1];
                
        end
        % if 1 == strcmp(type, 'table')
        %     % if code(4, 1) && code(5, 1) 
        %         for j = 2:9
        %             part_pcs{j} = part_pcs{j} + power(-1, j+1)*[i/200*0.1, 0, 0];
        %         end
        %         part_pcs{4} = part_pcs{4} + [-0.2, 0, 0.2];
        %         part_pcs{5} = part_pcs{4} .* [-1, 1, 1];
        %     % end
        % end
        tic;
        [total_pc, total_face, total_optimized_pc, total_optimized_face] = ReconstructFromCode4ReconInterRandom(code, part_pcs, part_faces, type, structure);
        toc;
%         tic;
%         [total_pc, total_face, total_optimized_pc, total_optimized_face] = ReconstructFromCode4ReconInterRandomAdj(code, part_pcs, part_faces, type);
%         toc;
%         disp(merge_dir);
%         disp(id);
        writeOBJ(fullfile(merge_dir, ['optimized_', id, '.obj']), total_optimized_pc, total_optimized_face);
%         if ispc
%             writeOBJ(fullfile(merge_dir, ['optimized_', id, '.obj']), total_optimized_pc, total_optimized_face);
% %             SaveObjT(fullfile(merge_dir, ['unoptimized_', id, '.obj']), total_pc', total_face');
%         else
%             writeOBJ(fullfile(merge_dir, ['optimized_', id, '.obj']), total_optimized_pc, total_optimized_face);
% %             SaveObj(fullfile(merge_dir, ['unoptimized_', id, '.obj']), total_pc', total_face');
%         end
    end
end