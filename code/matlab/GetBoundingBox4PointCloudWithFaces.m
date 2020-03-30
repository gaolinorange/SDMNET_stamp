function [v, f] = GetBoundingBox4PointCloudWithFaces(pc)
    if size(pc, 1) == 0
        v = [];
        f = [];
        return;
    end
    % get bounding box
    max_point = max(pc, [], 1);
    maxx = max_point(1);
    maxy = max_point(2);
    maxz = max_point(3);
    
    min_point = min(pc, [], 1);
    minx = min_point(1);
    miny = min_point(2);
    minz = min_point(3);
    
    x_diff = maxx - minx;
    y_diff = maxy - miny;
    z_diff = maxz - minz;
    v(1, :) = [minx, miny, minz];
    v(2, :) = [minx+x_diff, miny, minz];
    v(3, :) = [minx+x_diff, miny, minz+z_diff];
    v(4, :) = [minx, miny, minz+z_diff];
    v(5, :) = [minx, miny+y_diff, minz];
    v(6, :) = [minx+x_diff, miny+y_diff, minz];
    v(7, :) = [minx+x_diff, miny+y_diff, minz+z_diff];
    v(8, :) = [minx, miny+y_diff, minz+z_diff];

    f = [1,2,3,4;5,6,7,8;1,2,6,5;3,4,8,7;1,4,8,5;2,3,7,6];
end