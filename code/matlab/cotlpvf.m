function [ v, f, n, fn ] = cotlpvf( filename )

[v, f, n, fn] = meshlpvf(filename);
v = v';
n = n';
fn = fn';

end

