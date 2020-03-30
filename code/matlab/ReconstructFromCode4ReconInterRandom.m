%% ReconstructFromCode: reconstruct from code using quadratic optimization
function [total_pc, total_face, total_optimized_pc, total_optimized_face] = ReconstructFromCode4ReconInterRandom(code, part_pcs, part_faces, type, structure)
    if strcmp(type, 'chair') == 1
        part_names = {'armrest_1', 'armrest_2', 'back', 'leg1_1', 'leg1_2', 'leg2_1', 'leg2_2', 'seat'};
    elseif strcmp(type, 'knife') == 1
        part_names = {'part1', 'part2'};
    elseif strcmp(type, 'guitar') == 1
        part_names = {'part1', 'part2', 'part3'};
    elseif strcmp(type, 'skateboard')
        part_names = {'surface', 'bearing1', 'bearing2', 'wheel1_1', 'wheel1_2', 'wheel2_1', 'wheel2_2'};
    elseif strcmp(type, 'cup') == 1
        part_names = {'part1', 'part2'};
    elseif strcmp(type, 'car') == 1
        part_names = {'body', 'left_front_wheel', 'right_front_wheel', 'left_behind_wheel', 'right_behind_wheel'};
    elseif strcmp(type, 'plane') == 1
        part_names = {'body', 'left_wing', 'right_wing', 'left_tail', 'right_tail', 'upper_tail', 'down_tail', 'front_landing_gear', 'left_landing_gear', 'right_landing_gear', 'left_engine_1', 'right_engine_1', 'left_engine_2', 'right_engine_2'};
    elseif strcmp(type, 'table') == 1
        part_names = {'surface', 'leg1_1', 'leg1_2', 'leg2_1', 'leg2_2', 'leg3_1', 'leg3_2', 'leg4_1', 'leg4_2'};
    end

%     load(fullfile(dir, 'code.mat'));
%     is_valid = check_valid(type, num);
    n = size(code, 1);
    supporting = code(:, 2:1+n);
    supported = code(:, 2+n:1+2*n);
    center = code(:, 2+2*n:4+2*n);
    is_symmetry = code(:, 5+2*n);
    global_symmetry_plane = code(:, 6+2*n:9+2*n);

%     part_pcs = cell(n, 1);
%     part_faces = cell(n, 1);
    p0 = zeros(n, 3);
    q0 = zeros(n, 3);
    p = zeros(n, 3);
    q = zeros(n, 3);

    y_order = zeros(n, 2);
    for i = 1:n
        % part_pc_name = fullfile(dir, ['transformed_cube_', part_names{i}, '.obj']);
%         part_pc_name = fullfile(dir, [part_names{i}, '.obj']);
        part_pc = part_pcs{i};
        part_face = part_faces{i};
%         if exist(part_pc_name, 'file')
        if ~isempty(part_pc)
%             [part_pc, part_face] = cotlp(part_pc_name);
            part_bbox = GetBoundingBox4PointCloud(part_pc);
%             part_pcs{i} = part_pc;
%             part_faces{i} = part_face';
            q0(i, :) = 1/2*(part_bbox(7, :) - part_bbox(1, :));

            y_order(i, 1) = i;
            y_order(i, 2) = part_bbox(1, 2);
        else
%             part_pcs{i} = [];
            q0(i, :) = [0, 0, 0];

            y_order(i, 1) = i;
            y_order(i, 2) = inf;
        end

        p0(i, :) = code(i, 2*n+2:2*n+4);
    end
    num_parts = 0;
    for i = 1:n
        p(i, :) = p0(i, :);
        q(i, :) = q0(i, :);
    end
    % 6n*1
    x0 = zeros(6*n, 1);
    % 6n*1
    x = zeros(6*n, 1);
    % change x and x0 to one-column vector
    % 6n*1
    for i = 1:n
        x0(3*i-2:3*i, 1) = p0(i, :);
        x0(3*n+3*i-2:3*n+3*i, 1) = q0(i, :);
        x(3*i-2:3*i, 1) = p(i, :);
        x(3*n+3*i-2:3*n+3*i, 1) = q(i, :);
    end
    part_v = structure{1, 1};
    part_f = structure{1, 2};
    % 1/2
    alpha = 10000000;
    H = eye(6*n, 6*n);
    H(3*n+1:6*n, :) = alpha*H(3*n+1:6*n, :);
    f = -H'*x0;
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [];
    ub = [];
    
    % symmetry constraint
    i = 1;
    while i <= n
        % symmetry constraint
        if 1 == is_symmetry(i)
            j = i+1;
            normal = global_symmetry_plane(i, 1:3);
            d = abs(dot(center(i, :), normal));

            % symmetry constraint 1
            temp_A_for_pi_sub_pj = zeros(3, 6*n);
            temp_A_for_pi_sub_pj(:, 3*i-2:3*i) = eye(3,3);
            temp_A_for_pi_sub_pj(:, 3*j-2:3*j) = -eye(3,3);
            temp_A = normal*temp_A_for_pi_sub_pj;
            temp_b = -2*d;
            Aeq = [Aeq; temp_A];
            beq = [beq; temp_b];
            
            % symmetry constraint 2
            % convert cross product to matrix multiplication
            % https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
            normal_temp = zeros(3, 3);
            normal_temp(1,2) = normal(1,3);
            normal_temp(1,3) = -normal(1,2);
            normal_temp(2,1) = -normal(1,3);
            normal_temp(2,3) = normal(1,1);
            normal_temp(3,1) = normal(1,2);
            normal_temp(3,2) = -normal(1,1);
            temp_A = normal_temp*temp_A_for_pi_sub_pj;
            temp_b = [0, 0, 0]';
            Aeq = [Aeq; temp_A];
            beq = [beq; temp_b];

            % symmetry constraint 3
            temp_A_for_qi_sub_qj = zeros(3, 6*n);
            temp_A_for_qi_sub_qj(:, 3*n+3*i-2:3*n+3*i) = eye(3,3);
            temp_A_for_qi_sub_qj(:, 3*n+3*j-2:3*n+3*j) = -eye(3,3);
            temp_A = temp_A_for_qi_sub_qj;
            temp_b = [0, 0, 0]';
            Aeq = [Aeq; temp_A];
            beq = [beq; temp_b];

            i = i + 1;
        end
        i = i + 1;
    end
    % equal length constraint
    for i = 1:n
        i_supporting = code(i, 2:n+1);
        i_supported = code(i, n+2:2*n+1);
        supporting_group = [];
        supported_group = [];
        for j = 1:n
            if 1 == i_supporting(j)
                supporting_group = [supporting_group, j];
            end
            if 1 == i_supported(j)
                supported_group = [supported_group, j];
            end
        end
        for j = 1:size(i_supporting, 2)
            for k = j+1:size(i_supporting, 2)
                j_name = part_names{j};
                k_name = part_names{k};
                j_parts = strsplit(j_name, '_');
                k_parts = strsplit(k_name, '_');
                % same type
                if 1 == strcmp(j_parts{1}, k_parts{1})
                    temp_qj_sub_qk = zeros(1, 6*n);
                    temp_qj_sub_qk(1, 3*n+3*j-1) = 1;
                    temp_qj_sub_qk(1, 3*n+3*k-1) = -1;
                    
                    temp_A = temp_qj_sub_qk;
                    Aeq = [Aeq; temp_A];
                    temp_b = 0;
                    beq = [beq; temp_b];
                end
            end
        end
        for j = 1:size(i_supported, 2)
            for k = j+1:size(i_supported, 2)
                j_name = part_names{j};
                k_name = part_names{k};
                j_parts = strsplit(j_name, '_');
                k_parts = strsplit(k_name, '_');
                % same type
                if 1 == strcmp(j_parts{1}, k_parts{1})
                    temp_qj_sub_qk = zeros(1, 6*n);
                    temp_qj_sub_qk(1, 3*n+3*j-1) = 1;
                    temp_qj_sub_qk(1, 3*n+3*k-1) = -1;
                    
                    temp_A = temp_qj_sub_qk;
                    Aeq = [Aeq; temp_A];
                    temp_b = 0;
                    beq = [beq; temp_b];
                end
            end
        end
    end
    % Support Relationship Constraint
    for i = 1:n
        for j = 1:n
            if i == j
                continue;
            end
            % if i supports j
            if 1 == supporting(i, j)
                if q0(i, 2) > q0(j, 2)
                    temp_A_for_pj_sub_qj = zeros(1, 6*n);
                    temp_A_for_pj_sub_qj(1, 3*j-1) = 1;
                    % narrow the range of variation
                    temp_A_for_pj_sub_qj(1, 3*n+3*j-1) = -0.7;

                    temp_A_for_pi_add_qi = zeros(1, 6*n);
                    temp_A_for_pi_add_qi(1, 3*i-1) = 1;
                    temp_A_for_pi_add_qi(1, 3*n+3*i-1) = 1;

                    temp_A_for_pj_add_qj = zeros(1, 6*n);
                    temp_A_for_pj_add_qj(1, 3*j-1) = 1;
                    temp_A_for_pj_add_qj(1, 3*n+3*j-1) = 1;

                    temp_A = temp_A_for_pj_sub_qj - temp_A_for_pi_add_qi;
                    temp_b = 0;
                    A = [A; temp_A];
                    b = [b; temp_b];

                    temp_A = temp_A_for_pi_add_qi - temp_A_for_pj_add_qj;
                    temp_b = 0;
                    A = [A; temp_A];
                    b = [b; temp_b];
                else
                    temp_A_for_pi_sub_qi = zeros(1, 6*n);
                    temp_A_for_pi_sub_qi(1, 3*i-1) = 1;
                    temp_A_for_pi_sub_qi(1, 3*n+3*i-1) = -1;

                    temp_A_for_pj_sub_qj = zeros(1, 6*n);
                    temp_A_for_pj_sub_qj(1, 3*j-1) = 1;
                    temp_A_for_pj_sub_qj(1, 3*n+3*j-1) = -1;

                    temp_A_for_pi_add_qi = zeros(1, 6*n);
                    temp_A_for_pi_add_qi(1, 3*i-1) = 1;
                    % narrow the range of variation
                    temp_A_for_pi_add_qi(1, 3*n+3*i-1) = 0.7;

                    temp_A = temp_A_for_pi_sub_qi - temp_A_for_pj_sub_qj;
                    temp_b = 0;
                    A = [A; temp_A];
                    b = [b; temp_b];

                    temp_A = temp_A_for_pj_sub_qj - temp_A_for_pi_add_qi;
                    temp_b = 0;
                    A = [A; temp_A];
                    b = [b; temp_b];
                end
            end
        end
    end

    % Stable Support Constraint
    for i = 1:n
        supporting_i_items_group = [];
        for j = i:n
            if 1 == supported(i, j)
                supporting_i_items_group = [supporting_i_items_group, j];
                % x axis
                temp_A_for_pi_sub_qi = zeros(1, 6*n);
                temp_A_for_pi_sub_qi(1, 3*i-2) = 1;
                temp_A_for_pi_sub_qi(1, 3*n+3*i-2) = -1;

                temp_A_for_pj = zeros(1, 6*n);
                temp_A_for_pj(1, 3*j-2) = 1;

                temp_A_for_pi_add_qi = zeros(1, 6*n);
                temp_A_for_pi_add_qi(1, 3*i-2) = 1;
                temp_A_for_pi_add_qi(1, 3*n+3*i-2) = 1;

                temp_A = temp_A_for_pj - temp_A_for_pi_add_qi;
                temp_b = 0;
                A = [A; temp_A];
                b = [b; temp_b];

                temp_A = temp_A_for_pi_sub_qi - temp_A_for_pj;
                temp_b = 0;
                A = [A; temp_A];
                b = [b; temp_b];

                % z axis
                temp_A_for_pi_sub_qi = zeros(1, 6*n);
                temp_A_for_pi_sub_qi(1, 3*i) = 1;
                temp_A_for_pi_sub_qi(1, 3*n+3*i) = -1;

                temp_A_for_pj = zeros(1, 6*n);
                temp_A_for_pj(1, 3*j) = 1;

                temp_A_for_pi_add_qi = zeros(1, 6*n);
                temp_A_for_pi_add_qi(1, 3*i) = 1;
                temp_A_for_pi_add_qi(1, 3*n+3*i) = 1;

                temp_A = temp_A_for_pj - temp_A_for_pi_add_qi;
                temp_b = 0;
                A = [A; temp_A];
                b = [b; temp_b];

                temp_A = temp_A_for_pi_sub_qi - temp_A_for_pj;
                temp_b = 0;
                A = [A; temp_A];
                b = [b; temp_b];
            end
        end
        x_min = inf;
        z_min = inf;
        x_max = -inf;
        z_max = -inf;
        x_min_index = 0;
        z_min_index = 0;
        x_max_index = 0;
        z_max_index = 0;
        for j = 1:size(supporting_i_items_group)
            j_index = supporting_i_items_group(j);
            j_p = p0(j_index, :);
            j_q = q0(j_index, :);
            if j_p(1)-j_q(1) < x_min
                x_min = j_p(1);
                x_min_index = j_index;
            end
            if j_p(3)-j_q(3) < z_min
                z_min = j_p(3);
                z_min_index = j_index;
            end
            if j_p(1)+j_q(1) > x_max
                x_max = j_p(1);
                x_max_index = j_index;
            end
            if j_p(3)+j_q(3) > z_max
                z_max = j_p(3);
                z_max_index = j_index;
            end
        end
        if x_min_index ~= 0 && x_max_index ~= 0
            % x axis
            temp_A_for_pi_sub_qi = zeros(1, 6*n);
            temp_A_for_pi_sub_qi(1, 3*x_min_index-2) = 1;
            temp_A_for_pi_sub_qi(1, 3*n+3*x_min_index-2) = -1;
            % temp_A_for_pi_sub_qi(1, 3*n+3*x_min_index-2) = -0.1;

            temp_A_for_pj = zeros(1, 6*n);
            temp_A_for_pj(1, 3*i-2) = 1;

            temp_A_for_pi_add_qi = zeros(1, 6*n);
            temp_A_for_pi_add_qi(1, 3*x_max_index-2) = 1;
            temp_A_for_pi_add_qi(1, 3*n+3*x_max_index-2) = 1;
            % temp_A_for_pi_add_qi(1, 3*n+3*x_max_index-2) = 0.1;

            temp_A = temp_A_for_pi_sub_qi - temp_A_for_pj;
            temp_b = 0;
            A = [A; temp_A];
            b = [b; temp_b];

            temp_A = temp_A_for_pj - temp_A_for_pi_add_qi;
            temp_b = 0;
            A = [A; temp_A];
            b = [b; temp_b];
        end
        if z_min_index ~= 0 && z_max_index ~= 0
            % z axis
            temp_A_for_pi_sub_qi = zeros(1, 6*n);
            temp_A_for_pi_sub_qi(1, 3*z_min_index) = 1;
            temp_A_for_pi_sub_qi(1, 3*n+3*x_min_index) = -1;
            % temp_A_for_pi_sub_qi(1, 3*n+3*z_min_index) = -0.1;

            temp_A_for_pj = zeros(1, 6*n);
            temp_A_for_pj(1, 3*i) = 1;

            temp_A_for_pi_add_qi = zeros(1, 6*n);
            temp_A_for_pi_add_qi(1, 3*z_max_index) = 1;
            temp_A_for_pi_add_qi(1, 3*n+3*z_max_index) = 1;
            % temp_A_for_pi_add_qi(1, 3*n+3*z_max_index) = 0.1;
            

            temp_A = temp_A_for_pi_sub_qi - temp_A_for_pj;
            temp_b = 0;
            A = [A; temp_A];
            b = [b; temp_b];

            temp_A = temp_A_for_pj - temp_A_for_pi_add_qi;
            temp_b = 0;
            A = [A; temp_A];
            b = [b; temp_b];
        end
    end

    [x, fval]= quadprog(H,f, A, b);
%     fprintf('fval: %f\n', fval);
    % extract optimized p and q
    for i = 1:n
        p(i, 1:3) = x(3*i-2:3*i, 1)';
        q(i, 1:3) = x(3*n+3*i-2:3*n+3*i, 1)';
    end
    % if strcmp(type, 'chair') == 1
    %     p(1:2, 1:3) = p(1:2,1:3)*0.9;
    %     p(4:7, 1:3) = p(4:7,1:3)*0.9;
    % elseif strcmp(type, 'plane') == 1
    %     p(2, 1) = p(2, 1) + 0.035;
    %     p(3, 1) = p(3, 1) - 0.035;
    %     p(4:5, 2) = p(4, 2) + 0.005;
    %     p(4:5, 3) = p(5, 3) - 0.03;
    %     p(11:12, 3) = p(11:12, 3) + 0.05;
    %     p(13:14, 3) = p(13:14, 3) + 0.05;
    % end
    % total_optimized_pc = [];
    % total_pc = [];
    % total_face = [];
    % begin_index = 1;
    % end_index = n;
    % if strcmp(type, 'chair') == 1
    %     part_pc = part_pcs{n} - mean(part_pcs{n}, 1) + p(n, :);
    %     total_pc = [total_pc; part_pc];
    %     total_optimized_pc = [total_optimized_pc; part_pc];
    %     total_face = [total_face; part_faces{n}];
    %     begin_index = 1;
    %     end_index = n-1;
    % else
    %     part_pc = part_pcs{1} - mean(part_pcs{1}, 1) + p(1, :);
    %     total_pc = [total_pc; part_pc];
    %     total_optimized_pc = [total_optimized_pc; part_pc];
    %     total_face = [total_face; part_faces{1}];
    %     begin_index = 2;
    %     end_index = n;
    % end
    % for i = begin_index:end_index
    %     part_pc = part_pcs{i};
    %     if ~isempty(part_pc)
    %         total_pc = [total_optimized_pc; part_pc];
    %         [idx, D] = knnsearch(total_optimized_pc, part_pc);
    %         [min_d, min_index] = min(D);
    %         dxyz = total_optimized_pc(idx(min_index), :) - part_pc(min_index, :);
    %         p(i, :) = p(i, :) + dxyz;
    %         total_optimized_pc = [total_optimized_pc; part_pc];
    %         total_face = [total_face; part_faces{i}];
    %     end
    % end

    % place every part in optimized position p using optimized size q
    total_optimized_pc = [];
    total_pc = [];
    total_face = [];
    total_optimized_pc = part_v;
    total_optimized_face = part_f;
    for i = 1:num_parts
        if ~isempty(part_pcs{i})
            % if strcmp(type, 'plane') && (i == 6 || i == 11 || i == 12 || i == 13 || i == 14) && code(i, 1) == 0
            %     continue;
            % end
            if strcmp(type, 'chair') && (i == 1 || i == 2) && code(i, 1) == 0
                continue;
            end
            optimized_pc = part_pcs{i} - mean(part_pcs{i}, 1) + p(i, :);
            unoptimized_pc = part_pcs{i} - mean(part_pcs{i}, 1) + p0(i, :);
            total_pc = [total_pc; unoptimized_pc];
            total_face = AddPartFace2TotalFace(total_face, part_faces{i});
            total_optimized_pc = [total_optimized_pc; optimized_pc];
        end
    end
%     total_optimized_face = total_face;
%     if ispc
%         SaveObjT(fullfile(dir, 'unoptimized.obj'), total_pc', total_face');
%         SaveObjT(fullfile(dir, 'optimized.obj'), total_optimized_pc', total_face');
%     else
%         SaveObj(fullfile(dir, 'unoptimized.obj'), total_pc', total_faces');
%         SaveObj(fullfile(dir, 'optimized.obj'), total_optimized_pc', total_face');
%     end
end