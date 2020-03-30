function SupportAnalysis(part_objs_dir, type)
    part_names = getlabel(type);

    global_symmetry_plane = [1, 0, 0, 0];

    part_num = size(part_names, 2);
    code = zeros(part_num, part_num*2+9);
    code = AddPriori(code, type, part_names);

    part_pcs = cell(part_num, 1);
    part_bboxs = cell(part_num, 1);
    i = 1;
    while i <= part_num
        part_name = part_names{i};
        part_obj = fullfile(part_objs_dir, [part_name, '.obj']);
        if exist(part_obj, 'file')
            code(i, 1) = 1;
            [part_pc,~] = readobjfromfile(part_obj);

            if i+1 <= part_num
                next_part_name = part_names{i+1};
                % if symmetry
                part_name_split = strsplit(part_name, '_');
                next_part_name_split = strsplit(next_part_name, '_');
                if 1 == strcmp(part_name_split{1}, next_part_name_split{1})
                    code(i, 2*part_num+5) = 1;
                    code(i, 2*part_num+6:2*part_num+9) = global_symmetry_plane;
                    code(i+1, 2*part_num+5) = 1;
                    code(i+1, 2*part_num+6:2*part_num+9) = global_symmetry_plane;
                end
            end
            
            % fill center part
            [part_pc_bbox, ~] = GetBoundingBox4PointCloud(part_pc);
            % code(i, 2*part_num+2:2*part_num+4) = (part_pc_bbox(7, :) + part_pc_bbox(1, :))/2;
            code(i, 2*part_num+2:2*part_num+4) = mean(part_pc);
        else
            part_pc = [];
        end

        part_pcs{i} = part_pc;
        part_bbox = GetBoundingBox4PointCloud(part_pc);
        part_bboxs{i} = part_bbox;
        i=i+1;
    end

    distance = zeros(part_num, part_num);
    for i = 1:part_num
        for j = 1:part_num
            if i ~= j
                if size(part_pcs{i}, 1) == 0 || size(part_pcs{j}, 1) == 0
                    distance(i, j) = inf;
                else
                    [IDX, D] = knnsearch(part_pcs{i}, part_pcs{j});
                    dist = min(D, [], 1);
                    distance(i, j) = dist;
                end
            else
                distance(i, j) = Inf;
            end
        end    
    end
    for i = 1:part_num
        if size(part_pcs{i}, 1) == 0 || sum(code(i, 2:1+part_num)) ~= 0
            continue;
        end
        [~, j] = min(distance(i, :));
        bboxa = part_bboxs{i};
        bboxb = part_bboxs{j};
%         if IsIntersected(bboxa, bboxb)
        i_mean = mean(part_pcs{i}, 1);
        j_mean = mean(part_pcs{j}, 1);
        % i support j
        if i_mean(2) < j_mean(2)
            code(i, 1+j) = 1;
            code(j, part_num+1+i) = 1;
        % j support i
        else
            code(j, i+1) = 1;
            code(i, part_num+1+j) = 1;
        end
%         end
    end
    save(fullfile(part_objs_dir, 'code.mat'), 'code');
end