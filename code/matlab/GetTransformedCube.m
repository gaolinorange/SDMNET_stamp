function GetTransformedCube(pts_labels_dir, dir_postfix, type)
%     if use_postfix == 0
%         dir_postfix = 2000;
%         pts_dir = fullfile(pts_labels_dir, 'points');
%     else
%         pts_dir = fullfile(pts_labels_dir, ['points', num2str(dir_postfix)]);
%     end
    pts_dir =  pts_labels_dir;
    model_normalizedobj='model_normalized.obj';
    pts_list = dir([pts_dir, '\*']);
    pts_list(1:2)=[];
    
    divide_with_face_dir =  fullfile(pts_labels_dir, ['../box', num2str(dir_postfix)]);
    if ~exist(divide_with_face_dir, 'dir')
        mkdir(divide_with_face_dir);
    end
    part_names = getlabel(type);
    
    origin_cube_dir = ['cube_', type];
    cube_vs = {};
    cube_fs = {};
    for i = 1:size(part_names, 2)
        cubename=fullfile(origin_cube_dir, ['cube_', part_names{i}, '.obj']);
        if ~exist(cubename,'file')
            cubename=fullfile(origin_cube_dir, ['cube_std.obj']);
        end
        [v, f] = readobjfromfile(cubename);
        cube_vs = [cube_vs, v];
        cube_fs = [cube_fs, f+1];
    end
    
    specified_id = '';
    for i=1:size(pts_list, 1)
        disp(i);
        splitparts = strsplit(pts_list(i).name, '.');
        pts_numstr = splitparts{1};
        disp(pts_numstr);
        mesh_filename = fullfile(pts_labels_dir, pts_numstr,'models',model_normalizedobj);
        
        if strcmp(specified_id,'')==0&&0 == strcmp(pts_numstr, specified_id)
            continue;
        end
        [vertex, ~] = readobjfromfile(mesh_filename);
        if size(vertex,1)>400000
            continue
        end

        % used to store divide point cloud
        sub_divide_without_face_dir = fullfile(pts_labels_dir, pts_numstr,'models');
        if ~exist(sub_divide_without_face_dir, 'dir')
            continue;
        end
        
        sub_divide_with_face_dir = fullfile(divide_with_face_dir, pts_numstr);
        if ~exist(sub_divide_with_face_dir, 'dir')
            mkdir(sub_divide_with_face_dir);
        end
        
        copyfile(mesh_filename, fullfile(sub_divide_with_face_dir, model_normalizedobj))
        for j = 1:size(part_names, 2)
            if exist(fullfile(divide_with_face_dir, ['transformed_cube_', part_names{j}, '.obj']), 'file')
                continue;
            end
            if ~exist(fullfile(sub_divide_without_face_dir, [part_names{j}, '.obj']), 'file')
                continue;
            end
            try
                [part_points,~] = readobjfromfile(fullfile(sub_divide_without_face_dir, [part_names{j}, '.obj']));
            catch
                continue;
            end
%             part_points=removeoutliner(part_points);
%             if size(part_points, 1) == 0
%                 continue;
%             end

            transformed_box = zeros(8, 3);
            try
                [~,cornerpoints,h,~,~] = boundbox(part_points(:,1),part_points(:,2),part_points(:,3),'v',1);
                transformed_box=changebbxvert(cornerpoints);
            catch
                max_point = max(part_points, [], 1);
                maxx = max_point(1);
                maxy = max_point(2);
                maxz = max_point(3);
                
                min_point = min(part_points, [], 1);
                minx = min_point(1);
                miny = min_point(2);
                minz = min_point(3);
                
                x_diff = maxx - minx;
                y_diff = maxy - miny;
                z_diff = maxz - minz;
                transformed_box(1, :) = [minx, miny, minz];
                transformed_box(2, :) = [minx+x_diff, miny, minz];
                transformed_box(3, :) = [minx+x_diff, miny, minz+z_diff];
                transformed_box(4, :) = [minx, miny, minz+z_diff];
                
                transformed_box(5, :) = [minx, miny+y_diff, minz];
                transformed_box(6, :) = [minx+x_diff, miny+y_diff, minz];
                transformed_box(7, :) = [minx+x_diff, miny+y_diff, minz+z_diff];
                transformed_box(8, :) = [minx, miny+y_diff, minz+z_diff];
            end
            
            transformed_box(:, 4) = 1;
            
            origin_box = zeros(8, 3);
            origin_box(1, :) = [-0.5, -0.5, -0.5]+[0.5, 0.5, 0.5];
            origin_box(2, :) = [0.5, -0.5, -0.5]+[0.5, 0.5, 0.5];
            origin_box(3, :) = [0.5, -0.5, 0.5]+[0.5, 0.5, 0.5];
            origin_box(4, :) = [-0.5, -0.5, 0.5]+[0.5, 0.5, 0.5];
            
            origin_box(5, :) = [-0.5, 0.5, -0.5]+[0.5, 0.5, 0.5];
            origin_box(6, :) = [0.5, 0.5, -0.5]+[0.5, 0.5, 0.5];
            origin_box(7, :) = [0.5, 0.5, 0.5]+[0.5, 0.5, 0.5];
            origin_box(8, :) = [-0.5, 0.5, 0.5]+[0.5, 0.5, 0.5];
            origin_box(:, 4) = 1;
            
            % transpose to 4 * N
            transformed_box = transformed_box';
            origin_box = origin_box';
            transform_matrix = transformed_box/origin_box;            
            disp(transform_matrix);

            % save objs
            local_cubeid_v = cube_vs{j};
            local_cubeid_f = cube_fs{j};
            tempV = local_cubeid_v;
            tempV(:, 4) = 1;
            transformed_v = transform_matrix*tempV';
            
            SaveObjT(fullfile(sub_divide_with_face_dir, ['transformed_cube_', part_names{j}, '.obj']), transformed_v(1:3, :), local_cubeid_f');
            copyfile(fullfile(sub_divide_without_face_dir, [part_names{j}, '.obj']), fullfile(sub_divide_with_face_dir, [part_names{j}, '.obj']))
        end
    end
end