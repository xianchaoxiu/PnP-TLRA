clear all;
close all;

addpath(genpath('./proximal_operator'));
addpath(genpath('./tSVD'));

load('FFDNet_gray.mat');
%load('pavia_car');
%load Sandiego_new
%load Sandiego_gt
load abu-beach-2

%data = hsi;
%mask = hsi_gt;

mask=map;
map=mask;

% K=16; % number of clusters, pavia car
% beta=0.0004; % beta
% lamda=0.04; % lambda

% f_show=data(:,:,[37,18,8]);
% for i=1:3
%     max_f=max(max(f_show(:,:,i)));
%     min_f=min(min(f_show(:,:,i)));
%     f_show(:,:,i)=(f_show(:,:,i)-min_f)/(max_f-min_f);
% end

% figure,imshow(f_show);
% figure,imshow(mask,[]);

DataTest=data;
[H,W,Dim]=size(DataTest);
num=H*W;
for i=1:Dim
    DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
end

K=2; % number of clusters, pavia car
beta=0.01; % beta
lamda=0.01; % lambda

Y=reshape(DataTest, num, Dim)';

E=DeCNNAD(Y,H,W,K,beta,lamda,net);

E=reshape(E, num, Dim)';

% numb_dimension = 4;%airport-1  
%  numb_dimension =5;%airport-2 
%   numb_dimension =17;%airport-3
% numb_dimension = 17;%airport-4

% numb_dimension = 19;%ubran1
% numb_dimension = 21;%ubran2
%  numb_dimension =15;%ubran3 
%   numb_dimension =6;%ubran4
% numb_dimension = 10;%ubran5

numb_dimension = 15;
% numb_dimension = 15;%beach-1  
%  numb_dimension =15;%beach-2 
%  numb_dimension =12;%beach-3
% numb_dimension = 4;%beach-4
 
% numb_dimension = 6;%San Diego
% numb_dimension = 15;%HY-urban

DataTest = PCA_img(DataTest, numb_dimension);

[H,W,Dim]=size(DataTest);
for i=1:Dim 
    DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
end 

X=DataTest;  
[n1,n2,n3]=size(X);

% opts.lambda =0.06;%airport-1 
% opts.lambda =0.06; %airport-2 
% opts.lambda =0.06; %airport-3
% opts.lambda =0.06; %airport-4

% opts.lambda =0.2;%beach-1 
%opts.lambda =0.3;
 opts.lambda =0.3; %beach-2  
% opts.lambda =0.3; %beach-3
% opts.lambda =0.03; %beach-4

% opts.lambda =0.07;%urban-1 
% opts.lambda =0.06; %urban-2 
% opts.lambda =0.8; %urban-3
%opts.lambda =0.08;
% opts.lambda =0.02; %urban-4
%opts.lambda =0.08; %urban-5

% opts.lambda =0.02;%san
% opts.lambda =0.06;%Urban

opts.mu = 1e-4;
opts.tol = 1e-8;    
opts.rho = 1.5;
opts.max_iter = 100;
opts.DEBUG = 0;
% learn-dictionary
tic;
[L,S,rank] = dictionary_learning_tlrr1( X, opts);
max_iter=100;
Debug = 0;

% lambda=0.01;%airport-1  
% lambda=0.01;%airport-2 
% lambda=0.05;%airport-3 
% lambda=0.05;%airport-4 % :)

% lambda=0.04;%beach-1  
% lambda=0.05;%beach-2 
% lambda=0.05;%beach-3 
% lambda=0.0006;%beach-4 % :)

% lambda=0.2;%urban-1  
% lambda=0.03;%urban-2 
% lambda=0.03;%urban-3 
lambda=0.05;
% lambda=0.001;%urban-4 % :)
%lambda=0.02;%urban-5 % :)

% lambda=0.01;%HY-Urban
% lambda=0.01;%San :)

[Z,tlrr_E,Z_rank,err_va ] = TLRA(X,L,max_iter,lambda,Debug);

E=reshape(tlrr_E, num, Dim)';

mask_reshape = reshape(mask, 1, num);
anomaly_map = logical(double(mask_reshape)>0);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
normal_map = logical(double(mask_reshape)==0);

r_new=sqrt(sum(E.^2,1));
r_max = max(r_new(:));
taus = linspace(0, r_max, 5000);
PF10=zeros(1,5000);
PD10=zeros(1,5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r_new> tau);
  PF10(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD10(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area_TLRR = sum((PF10(1:end-1)-PF10(2:end)).*(PD10(2:end)+PD10(1:end-1))/2)
area_TLRR_taus = sum((taus(1:end-1)-taus(2:end)).*(PF10(2:end)+PF10(1:end-1))/2)

semilogx(PF10, PD10, 'g-', 'LineWidth', 4);

% r_new=sqrt(sum(E.^2,1));
% AUC=ROC(r_new,map,1) % AUC
f_anomaly=reshape(r_new,[H,W]);
f_anomaly=(f_anomaly-min(f_anomaly(:)))/(max(f_anomaly(:))-min(f_anomaly(:)));
imwrite(f_anomaly,'beach.jpg');