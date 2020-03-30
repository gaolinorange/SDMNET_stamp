function [local_bbox,local_bbox_face] = GetBoundingBox4PointCloud(pc)
    if size(pc, 1) == 0
        local_bbox = [];
        return;
    end
    % get bounding box
    max_point = max(pc, [], 1);
    maxx = max_point(1)+0.001;
    maxy = max_point(2)+0.001;
    maxz = max_point(3)+0.001;
    
    min_point = min(pc, [], 1);
    minx = min_point(1)-0.001;
    miny = min_point(2)-0.001;
    minz = min_point(3)-0.001;
    
    x_diff = maxx - minx;
    y_diff = maxy - miny;
    z_diff = maxz - minz;
    local_bbox(1, :) = [minx, miny, minz];
    local_bbox(2, :) = [minx+x_diff, miny, minz];
    local_bbox(3, :) = [minx+x_diff, miny, minz+z_diff];
    local_bbox(4, :) = [minx, miny, minz+z_diff];
    local_bbox(5, :) = [minx, miny+y_diff, minz];
    local_bbox(6, :) = [minx+x_diff, miny+y_diff, minz];
    local_bbox(7, :) = [minx+x_diff, miny+y_diff, minz+z_diff];
    local_bbox(8, :) = [minx, miny+y_diff, minz+z_diff];
    local_bbox_face=[1 6 2;5 6 1;7 3 2 ;7 2 6 ;4 3 7; 8 4 7;1 4 8;1 8 5;5 7 6;5 8 7 ;1 3 4; 1 2 3];
end