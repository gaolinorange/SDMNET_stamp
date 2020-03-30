function CA = file2cellArray(fname)
% fname is a string that names a .txt file in the current directory.
% CA is a cell array with CA{k} being the k-th line in the file.

fid= fopen(fname, 'r');
ik= 0;
while ~feof(fid)
   ik= ik+1;
   CA{ik}= fgetl(fid);
end
fclose(fid);