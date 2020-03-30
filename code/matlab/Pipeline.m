% preprocess total pipeline: divide, get transformed cube, get sub mesh, resigter, get structure code, post regist

% Car

addpath('./nonregistration')
cate='car';
postfix=50;
partname={};

categarydir = './DATA/Car/Car';
box_dir =[categarydir,'\..\box',num2str(postfix)];
vaedir=[categarydir,'\..\vaenew',num2str(postfix)];

GetTransformedCube(categarydir, postfix, cate);
regist(box_dir, cate)
SupportAnalysisScript(box_dir, cate);
generate_data(box_dir,vaedir,cate,partname,'');

% next steps:
% 1. train sdm-net
% 2. obtain the merged obj by "GetOptimizedObj.m"
% for the post-processing in our paper, we have some demos in "DEMO.m"
