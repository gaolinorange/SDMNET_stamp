function nonrigidregis(folder,name)

% w1 plane s2 point w3 global rigid w4 local rigid

if nargin>1
    numpart=1;
else
    samplelist=dir([folder,'\transformed_cube_*.obj']);
    numpart=length(samplelist);
end
for i=1:numpart
    
    if nargin<1
        samplefile = samplelist(i).name;
        splitparts = strsplit(samplefile, '_');
        [~,name]=fileparts(splitparts{3});       
    end

    srcfile = fullfile(folder, ['transformed_cube_',name, '.obj']);
    tarfile = fullfile(folder, [name,'.obj']);
    savefile = fullfile(folder, [name, '_reg.obj']);
    
    %     srcfile=[folder,'\part',num2str(i),'.obj'];
    %     tarfile=[folder,'\sample_part',num2str(i),'_pc.obj'];
    %     savefile =[folder,'\part',num2str(i),'_reg.obj'];
    w1 = 7;
    w2 = 0.0;
    w3 = 0;
    w4 = 40;
    %         w1 = 50.0;
    %     w2 = 100;
    %     w3 = 1500;
    %     w4 = 90;
    iter = 40;
    % regis_union(srcfile, tarfile, savefile);
    if exist(srcfile,'file')&&exist(tarfile,'file')
        
        
        if ~exist(savefile,'file')
            try
                NonRigidAlignment3Dnew(srcfile,tarfile, iter,savefile,w1,w2,w3,w4);
                %         denoise_mesh1(tmpfile, savefile)
                %         end
            catch
                continue
            end
        end
    end
    
end
end