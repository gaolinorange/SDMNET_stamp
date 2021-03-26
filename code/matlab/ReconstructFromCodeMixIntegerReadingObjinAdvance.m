function [total_pc, total_face, total_optimized_pc, total_optimized_face] = ReconstructFromCodeMixIntegerReadingObjinAdvance(code, part_pcs, part_faces, type)
    if strcmp(type, 'chair') == 1
%         part_names = {'armrest_1', 'armrest_2', 'back', 'leg1_1', 'leg1_2', 'leg2_1', 'leg2_2', 'seat'};
        support_directions = [
                                0, 0, 1, 0, 0, 0, 0, 1;
                                0, 0, 1, 0, 0, 0, 0, 1;
                                1, 1, 0, 2, 2, 2, 2, 2;
                                0, 0, 2, 0, 0, 0, 0, 2;
                                0, 0, 2, 0, 0, 0, 0, 2;
                                0, 0, 2, 0, 0, 0, 0, 2;
                                0, 0, 2, 0, 0, 0, 0, 2;
                                1, 1, 2, 2, 2, 2, 2, 0;
                            ];
    elseif strcmp(type, 'knife') == 1
%         part_names = {'part1', 'part2'};
        support_directions = [
                                0, 2;
                                2, 0;
                            ];
    elseif strcmp(type, 'guitar') == 1
%         part_names = {'part1', 'part2', 'part3'};
        support_directions = [
                                0, 2, 0;
                                2, 0, 2;
                                0, 2, 0
                            ];
    elseif strcmp(type, 'skateboard')
%         part_names = {'surface', 'bearing1', 'bearing2', 'wheel1_1', 'wheel1_2', 'wheel2_1', 'wheel2_2'};
        support_directions = [
                                0, 2, 2, 2, 2, 2, 2;
                                2, 0, 0, 1, 1, 1, 1;
                                2, 0, 0, 1, 1, 1, 1;
                                2, 1, 1, 0, 0, 0, 0;
                                2, 1, 1, 0, 0, 0, 0;
                                2, 1, 1, 0, 0, 0, 0;
                                2, 1, 1, 0, 0, 0, 0;
                            ];
    elseif strcmp(type, 'cup') == 1
%         part_names = {'part1', 'part2'};
        support_directions = [
                                0, 1;
                                1, 0;
                            ];
    elseif strcmp(type, 'car') == 1
%         part_names = {'body', 'left_front_wheel', 'right_front_wheel', 'left_behind_wheel', 'right_behind_wheel', 'left_mirror', 'right_mirror'};
        support_directions = [
                                0, 2, 2, 2, 2, 1, 1;
                                2, 0, 0, 0, 0, 0, 0;
                                2, 0, 0, 0, 0, 0, 0;
                                2, 0, 0, 0, 0, 0, 0;
                                2, 0, 0, 0, 0, 0, 0;
                                1, 0, 0, 0, 0, 0, 0;
                                1, 0, 0, 0, 0, 0, 0;
                            ];
    elseif strcmp(type, 'plane') == 1
%         part_names = {'body', 'left_wing', 'right_wing', 'left_tail', 'right_tail', 'upper_tail', 'down_tail', 'front_landing_gear', 'left_landing_gear', 'right_landing_gear', 'left_engine_1', 'right_engine_1', 'left_engine_2', 'right_engine_2'};
        support_directions = [
                                0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1;
                                1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2;
                                1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2;
                                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                                2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                                2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                                2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                                2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                                2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                                1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                                1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                                1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                                1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                            ];
    elseif strcmp(type, 'table') == 1
%         part_names = {'surface', 'leg1_1', 'leg1_2', 'leg2_1', 'leg2_2', 'leg3_1', 'leg3_2', 'leg4_1', 'leg4_2'};
        support_directions = [
                                0, 2, 2, 2, 2, 2, 2, 2;
                                2, 0, 0, 0, 0, 0, 0, 0;
                                2, 0, 0, 0, 0, 0, 0, 0;
                                2, 0, 0, 0, 0, 0, 0, 0;
                                2, 0, 0, 0, 0, 0, 0, 0;
                                2, 0, 0, 0, 0, 0, 0, 0;
                                2, 0, 0, 0, 0, 0, 0, 0;
                                2, 0, 0, 0, 0, 0, 0, 0;
                            ];
    end
    part_names = getlabel(type);
    % load(fullfile(dir, 'code.mat'));
    n = size(code, 1);
    supporting = code(:, 2:1+n);
    supported = code(:, 2+n:1+2*n);
    center = code(:, 2+2*n:4+2*n);
    is_symmetry = code(:, 5+2*n);
    global_symmetry_plane = code(:, 6+2*n:9+2*n);

    % part_pcs = cell(n, 1);
    % part_faces = cell(n, 1);
    p0 = zeros(n, 3);
    q0 = zeros(n, 3);
    p = zeros(n, 3);
    q = zeros(n, 3);
    delta1 = zeros(n, n);
    delta2 = zeros(n, n);

    y_order = zeros(n, 2);
    for i = 1:n
        % part_pc_name = fullfile(dir, [part_names{i}, '.obj']);
        part_pc = part_pcs{i};
        part_face = part_faces{i};
        if ~isempty(part_pc)
            % [part_pc, part_face] = cotlp(part_pc_name);
            part_bbox = GetBoundingBox4PointCloud(part_pc);
            % part_pcs{i} = part_pc;
            % part_faces{i} = part_face';
            q0(i, :) = 1/2*(part_bbox(7, :) - part_bbox(1, :));
            p0(i, :) = 1/2*(part_bbox(7, :) + part_bbox(1, :));
            y_order(i, 1) = i;
            y_order(i, 2) = part_bbox(1, 2);
        else
            % part_pcs{i} = [];
            q0(i, :) = [0, 0, 0];
            p0(i, :) = [0, 0, 0];
            y_order(i, 1) = i;
            y_order(i, 2) = inf;
        end
    end
    for i = 1:n
        if strcmp('chair', type) == 1 && i <= 7
            if ~isempty(part_pcs{i})
                part_pcs{i} = part_pcs{i} + 1.5*get_direction(part_pcs{8}, part_pcs{i});
                part_bbox = GetBoundingBox4PointCloud(part_pcs{i});
                q0(i, :) = 1/2*(part_bbox(7, :) - part_bbox(1, :));
                p0(i, :) = 1/2*(part_bbox(7, :) + part_bbox(1, :));
                y_order(i, 1) = i;
                y_order(i, 2) = part_bbox(1, 2);
            end
        end
    end
    
    N = 6*n+2*n*n;
    x0 = zeros(N, 1);
    x = zeros(N, 1);
    for i = 1:n
        x0(3*i-2:3*i, 1) = p0(i, :);
        x0(3*n+3*i-2:3*n+3*i, 1) = q0(i, :);
        x0(6*n+(i-1)*n+1:6*n+i*n, 1) = delta1(i, :);
        x0(6*n+n*n+(i-1)*n+1:6*n+n*n+i*n, 1) = delta2(i, :);

        x(3*i-2:3*i, 1) = p(i, :);
        x(3*n+3*i-2:3*n+3*i, 1) = q(i, :);
        x(6*n+(i-1)*n+1:6*n+i*n, 1) = delta1(i, :);
        x(6*n+n*n+(i-1)*n+1:6*n+n*n+i*n, 1) = delta2(i, :);
    end
    epsilon1 = 0.1;
    epsilon2 = 0.1;
    alpha = 100;
    H = eye(N, N);
    H(3*n+1:6*n, :) = alpha*H(3*n+1:6*n, :);
    H(6*n+1:N, :) = 0*H(6*n+1:N, :);

    f = -H'*x0;
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [];
    ub = [];
    
    i = 1;
    while i <= n
        if 1 == is_symmetry(i) && code(i, 1) == 1
            j = i+1;
            normal = global_symmetry_plane(i, 1:3);
            d = abs(dot(center(i, :), normal));

            temp_A_for_pi_sub_pj = zeros(3, N);
            temp_A_for_pi_sub_pj(:, 3*i-2:3*i) = eye(3,3);
            temp_A_for_pi_sub_pj(:, 3*j-2:3*j) = -eye(3,3);
            temp_A = normal*temp_A_for_pi_sub_pj;
            temp_b = -2*d;
            Aeq = [Aeq; temp_A];
            beq = [beq; temp_b];
            
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

            temp_A_for_qi_sub_qj = zeros(3, N);
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
    A = [A; Aeq];
    b = [b; beq];
    lb = [lb; beq];

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
        if size(i_supporting, 2) > 1
            for j = 1:size(i_supporting, 2)
                for k = j+1:size(i_supporting, 2)
                    j_name = part_names{j};
                    k_name = part_names{k};
                    j_parts = strsplit(j_name, '_');
                    k_parts = strsplit(k_name, '_');
                    if 1 == strcmp(j_parts{1}, k_parts{1})
                        temp_qj_sub_qk = zeros(1, N);
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
        if size(i_supported, 2) > 1
            for j = 1:size(i_supported, 2)
                for k = j+1:size(i_supported, 2)
                    j_name = part_names{j};
                    k_name = part_names{k};
                    j_parts = strsplit(j_name, '_');
                    k_parts = strsplit(k_name, '_');
                    if 1 == strcmp(j_parts{1}, k_parts{1})
                        temp_qj_sub_qk = zeros(1, N);
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
    end
    M = 200;
    for loopi = 1:n
        for loopj = 1:n
            i = loopi;
            j = loopj;
            if 1 == supporting(i, j)
                t = support_directions(i, j);

                if t == 1
                    t1 = 2;
                    t2 = 3;
                elseif t == 2
                    t1 = 1;
                    t2 = 3;
                elseif t == 3
                    t1 = 1;
                    t2 = 2;
                else
                    continue;
                end
                if(p0(i, t) > p0(j, t))
                    temp_i = i;
                    i = j;
                    j = temp_i;
                end
                disp([i, j, t]);         

                temp_A = zeros(1, N);
                temp_A(1, 3*j-2+t-1) = 1;
                temp_A(1, 3*n+3*j-2+t-1) = epsilon1 - 1;
                temp_A(1, 3*i-2+t-1) = -1;
                temp_A(1, 3*n+3*i-2+t-1) = -1;
                temp_b = 0;
                A = [A; temp_A];
                b = [b; temp_b];
                lb = [lb; -inf];

                temp_A = zeros(1, N);
                temp_A(1, 3*i-2+t-1) = 1;
                temp_A(1, 3*n+3*i-2+t-1) = 1;
                temp_A(1, 3*j-2+t-1) = -1;
                temp_A(1, 3*n+3*j-2+t-1) = epsilon2-1;
                temp_b = 0;
                A = [A; temp_A];
                b = [b; temp_b];
                lb = [lb; -inf];

                if i == 8 && j == 3 && t == 2 && strcmp('chair', type) == 1
                    temp_A = zeros(1, N);
                    temp_A(1, 3*j-2+3-1) = 1;
                    temp_A(1, 3*n+3*j-2+3-1) = epsilon1 - 1;
                    temp_A(1, 3*i-2+3-1) = -1;
                    temp_A(1, 3*n+3*i-2+3-1) = -1;
                    temp_b = 0;
                    A = [A; temp_A];
                    b = [b; temp_b];
                    lb = [lb; -inf];

                    temp_A = zeros(1, N);
                    temp_A(1, 3*i-2+3-1) = 1;
                    temp_A(1, 3*n+3*i-2+3-1) = 1;
                    temp_A(1, 3*j-2+3-1) = -1;
                    temp_A(1, 3*n+3*j-2+3-1) = epsilon2-1;
                    temp_b = 0;
                    A = [A; temp_A];
                    b = [b; temp_b];
                    lb = [lb; -inf];
                    continue;
                end

                if ((i == 1 && j == 5 && t == 1) || (i == 1 && j == 6 && t == 2)) && strcmp('plane', type) == 1
                    temp_A = zeros(1, N);
                    temp_A(1, 3*j-2+3-1) = 1;
                    temp_A(1, 3*n+3*j-2+3-1) = epsilon1 - 1;
                    temp_A(1, 3*i-2+3-1) = -1;
                    temp_A(1, 3*n+3*i-2+3-1) = -1;
                    temp_b = 0;
                    A = [A; temp_A];
                    b = [b; temp_b];
                    lb = [lb; -inf];

                    temp_A = zeros(1, N);
                    temp_A(1, 3*i-2+3-1) = 1;
                    temp_A(1, 3*n+3*i-2+3-1) = 1;
                    temp_A(1, 3*j-2+3-1) = -1;
                    temp_A(1, 3*n+3*j-2+3-1) = epsilon2-1;
                    temp_b = 0;
                    A = [A; temp_A];
                    b = [b; temp_b];
                    lb = [lb; -inf];
                    continue;
                end
                temp_A = zeros(1, N);
                temp_A(1, 3*j-2+t1-1) = 1;
                temp_A(1, 3*n+3*j-2+t1-1) = -1;
                temp_A(1, 3*i-2+t1-1) = -1;
                temp_A(1, 3*n+3*i-2+t1-1) = 1;
                temp_A(1, 6*n+n*(i-1)+j) = -M;
                temp_b = 0;

                A = [A; temp_A];
                b = [b; temp_b];
                lb = [lb; -inf];

                temp_A = zeros(1, N);
                temp_A(1, 3*i-2+t1-1) = 1;
                temp_A(1, 3*n+3*i-2+t1-1) = 1;
                temp_A(1, 3*j-2+t1-1) = -1;
                temp_A(1, 3*n+3*j-2+t1-1) = -1;
                temp_A(1, 6*n+n*(i-1)+j) = -M;
                temp_b = 0;

                A = [A; temp_A];
                b = [b; temp_b];
                lb = [lb; -inf];

                temp_A = zeros(1, N);
                temp_A(1, 3*j-2+t2-1) = 1;
                temp_A(1, 3*n+3*j-2+t2-1) = -1;
                temp_A(1, 3*i-2+t2-1) = -1;
                temp_A(1, 3*n+3*i-2+t2-1) = 1;
                temp_A(1, 6*n+n*(i-1)+j) = -M;
                temp_b = 0;

                A = [A; temp_A];
                b = [b; temp_b];
                lb = [lb; -inf];

                temp_A = zeros(1, N);
                temp_A(1, 3*i-2+t2-1) = 1;
                temp_A(1, 3*n+3*i-2+t2-1) = 1;
                temp_A(1, 3*j-2+t2-1) = -1;
                temp_A(1, 3*n+3*j-2+t2-1) = -1;
                temp_A(1, 6*n+n*(i-1)+j) = -M;
                temp_b = 0;

                A = [A; temp_A];
                b = [b; temp_b];
                lb = [lb; -inf];

                temp_A = zeros(1, N);
                temp_A(1, 3*j-2+t1-1) = 1;
                temp_A(1, 3*n+3*j-2+t1-1) = -1;
                temp_A(1, 3*i-2+t1-1) = -1;
                temp_A(1, 3*n+3*i-2+t1-1) = 1;
                temp_A(1, 6*n+n*n+n*(i-1)+j) = -M;
                temp_b = 0;

                A = [A; temp_A];
                b = [b; temp_b];
                lb = [lb; -inf];

                temp_A = zeros(1, N);
                temp_A(1, 3*i-2+t1-1) = 1;
                temp_A(1, 3*n+3*i-2+t1-1) = 1;
                temp_A(1, 3*j-2+t1-1) = -1;
                temp_A(1, 3*n+3*j-2+t1-1) = -1;
                temp_A(1, 6*n+n*n+n*(i-1)+j) = -M;
                temp_b = 0;

                A = [A; temp_A];
                b = [b; temp_b];
                lb = [lb; -inf];

                temp_A = zeros(1, N);
                temp_A(1, 3*j-2+t2-1) = 1;
                temp_A(1, 3*n+3*j-2+t2-1) = -1;
                temp_A(1, 3*i-2+t2-1) = -1;
                temp_A(1, 3*n+3*i-2+t2-1) = 1;
                temp_A(1, 6*n+n*n+n*(i-1)+j) = -M;
                temp_b = 0;

                A = [A; temp_A];
                b = [b; temp_b];
                lb = [lb; -inf];

                temp_A = zeros(1, N);
                temp_A(1, 3*i-2+t2-1) = 1;
                temp_A(1, 3*n+3*i-2+t2-1) = 1;
                temp_A(1, 3*j-2+t2-1) = -1;
                temp_A(1, 3*n+3*j-2+t2-1) = -1;
                temp_A(1, 6*n+n*n+n*(i-1)+j) = -M;
                temp_b = 0;

                A = [A; temp_A];
                b = [b; temp_b];
                lb = [lb; -inf];

                temp_A = zeros(1, N);
                temp_A(1, 6*n+n*(i-1)+j) = 1;
                temp_b = 1;
                A = [A; temp_A];
                b = [b; temp_b];
                lb = [lb; 0];

                temp_A = zeros(1, N);
                temp_A(1, 6*n+n*n+n*(i-1)+j) = 1;
                temp_b = 1;
                A = [A; temp_A];
                b = [b; temp_b];
                lb = [lb; 0];

                temp_A = zeros(1, N);
                temp_A(1, 6*n+n*(i-1)+j) = 1;
                temp_A(1, 6*n+n*n+n*(i-1)+j) = 1;
                temp_b = 1;
                A = [A; temp_A];
                b = [b; temp_b];
                lb = [lb; 0];                
            end
        end
    end

    for i = 1:n
        supporting_i_items_group = [];
        for j = 1:n
            if 1 == supported(i, j)
                supporting_i_items_group = [supporting_i_items_group, j];
            end
        end
        if size(supporting_i_items_group, 2) <= 1
            continue;
        end
        x_min = inf;
        z_min = inf;
        x_max = -inf;
        z_max = -inf;
        x_min_index = 0;
        z_min_index = 0;
        x_max_index = 0;
        z_max_index = 0;
        for j = 1:size(supporting_i_items_group, 2)
            j_index = supporting_i_items_group(j);
            j_p = p0(j_index, :);
            j_q = q0(j_index, :);
            if j_p(1)-j_q(1) < x_min
                x_min = j_p(1)-j_q(1);
                x_min_index = j_index;
            end
            if j_p(3)-j_q(3) < z_min
                z_min = j_p(3)-j_q(3);
                z_min_index = j_index;
            end
            if j_p(1)+j_q(1) > x_max
                x_max = j_p(1)+j_q(1);
                x_max_index = j_index;
            end
            if j_p(3)+j_q(3) > z_max
                z_max = j_p(3)+j_q(3);
                z_max_index = j_index;
            end
        end
        if x_min_index ~= 0 && x_max_index ~= 0
            temp_A_for_pi_sub_qi = zeros(1, N);
            temp_A_for_pi_sub_qi(1, 3*x_min_index-2) = 1;
            temp_A_for_pi_sub_qi(1, 3*n+3*x_min_index-2) = -1;

            temp_A_for_pj = zeros(1, N);
            temp_A_for_pj(1, 3*x_min_index-2) = 1;

            temp_A_for_pi_add_qi = zeros(1, N);
            temp_A_for_pi_add_qi(1, 3*x_min_index-2) = 1;
            temp_A_for_pi_add_qi(1, 3*n+3*x_min_index-2) = 1;

            temp_A = temp_A_for_pi_sub_qi - temp_A_for_pj;
            temp_b = 0;
            A = [A; temp_A];
            b = [b; temp_b];
            lb = [lb; -inf];

            temp_A = temp_A_for_pj - temp_A_for_pi_add_qi;
            temp_b = 0;
            A = [A; temp_A];
            b = [b; temp_b];
            lb = [lb; -inf];
        end
        if z_min_index ~= 0 && z_max_index ~= 0
            temp_A_for_pi_sub_qi = zeros(1, N);
            temp_A_for_pi_sub_qi(1, 3*x_min_index) = 1;
            temp_A_for_pi_sub_qi(1, 3*n+3*x_min_index) = -1;

            temp_A_for_pj = zeros(1, N);
            temp_A_for_pj(1, 3*x_min_index) = 1;

            temp_A_for_pi_add_qi = zeros(1, N);
            temp_A_for_pi_add_qi(1, 3*x_min_index) = 1;
            temp_A_for_pi_add_qi(1, 3*n+3*x_min_index) = 1;

            temp_A = temp_A_for_pi_sub_qi - temp_A_for_pj;
            temp_b = 0;
            A = [A; temp_A];
            b = [b; temp_b];
            lb = [lb; -inf];

            temp_A = temp_A_for_pj - temp_A_for_pi_add_qi;
            temp_b = 0;
            A = [A; temp_A];
            b = [b; temp_b];
            lb = [lb; -inf];
        end
    end
 
    xtype = repmat('C', 1, 6*n+2*n*n);
    for i = 6*n+1:6*n+2*n*n
        xtype(1, i) = 'I';
    end

    IntVars = zeros(1, 6*n+2*n*n);
    for i = 6*n+1:6*n+2*n*n
        IntVars(1, i) = 1;
    end
    % A = [A; Aeq];
    % b = [b; beq];
    
%     tic;
    [x, fval]= quadprog(H,f, A, b,Aeq,beq);
    % Opt = opti('qp', H, f, 'ineq', A, b, 'xtype', xtype);
    % [x,fval,exitflag,info] = solve(Opt);
%     Prob = miqpAssign(H, f, A, lb, b, [], [], x0, ...
%         IntVars, [], [], [], ...
%         'name', [], [], ...
%         [], [], [], []);
%     Result = tomRun('gurobi', Prob, 1);
%     x = Result.x_k;
%     toc;
    
    for i = 1:n
        if i ~= 3 && i~=8
            p(i, 1:3) = x(3*i-2:3*i, 1)';
            q(i, 1:3) = x(3*n+3*i-2:3*n+3*i, 1)';
        end
        if strcmp('chair', type) == 1 && i >= 4 && i <= 7
            [local_bbox, local_bbox_face] = GetBoundingBox4PointCloud(part_pcs{8});
%             [local_bboxi,local_bbox_facei] = GetBoundingBox4PointCloud(part_pcs{8});
            dxyz = local_bbox(7, :) - local_bbox(1, :);
            
            if i == 4
                p(i, 1) = local_bbox(1, 1)*0.9;
%                 p(i, 2) = local_bbox(1, 2) - dxyz(2)/1.5;
                p(i, 3) = local_bbox(1, 3)*0.9;
            elseif i == 5
                p(i, 1) = local_bbox(2, 1)*0.9;
%                 p(i, 2) = local_bbox(1, 2) - dxyz(2)/1.5;
                p(i, 3) = local_bbox(2, 3)*0.9;
            elseif i == 6
                p(i, 1) = local_bbox(4, 1)*0.9;
%                 p(i, 2) = local_bbox(1, 2) - dxyz(2)/1.5;
                p(i, 3) = local_bbox(4, 3)*0.9;
            elseif i == 7
                p(i, 1) = local_bbox(3, 1)*0.9;
%                 p(i, 2) = local_bbox(1, 2) - dxyz(2)/1.5;
                p(i, 3) = local_bbox(3, 3)*0.9;
            end
%             p(i, :) = p(i, :) + get_direction(part_pcs{8}, part_pcs{i});
        end
    end
    
    total_optimized_pc = [];
    total_pc = [];
    total_face = [];
    for i = 1:n
        
        if ~isempty(part_pcs{i})
            if strcmp('chair', type) == 1 && i~=3 && i ~= 8
                optimized_pc = part_pcs{i} - mean(part_pcs{i}, 1) + p(i, :);
            else
                optimized_pc = part_pcs{i};
            end
            total_pc = [total_pc; part_pcs{i}];
            total_face = AddPartFace2TotalFace(total_face, part_faces{i});
            total_optimized_pc = [total_optimized_pc; optimized_pc];
        
        end
    end
    total_optimized_face = total_face;
    % if ispc
    %     SaveObjT(fullfile(dir, 'unoptimized.obj'), total_pc', total_face');
    %     SaveObjT(fullfile(dir, 'optimized.obj'), total_optimized_pc', total_face');
    % else
    %     SaveObj(fullfile(dir, 'unoptimized.obj'), total_pc', total_faces');
    %     SaveObj(fullfile(dir, 'optimized.obj'), total_optimized_pc', total_face');
    % end
end
