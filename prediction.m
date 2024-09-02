FU2total=load('EFTFU27mm6183dtensorsync.mat');
FU3total=load('EFTFU37mm6183dtensorsync.mat');
FU2part=load('EFTFU2ansync.mat');
FU3part=load('EFTFU3ansync.mat');
BLpart=load('EFTBLansyncpermute.mat');
FU2h=load('EFTFU2hsync.mat');
FU3h=load('EFTFU3hsync.mat');
load('EFT13conditionFU2FU3.mat');
load('EFTFU2FU3anccombinedecodingresult.mat');


FU2Fu3128=cat(4,FU2part.EFTFU2ansync,FU3part.EFTFU3ansync);
datarun=EFT13conditionFU2FU3;
datarun=fillmissing(datarun,'linear');
%datarun=FU2Fu3128;
datarun=fmt(datarun);
R=10;
% datatest=fmt(datatest);
%Rest=rankest(datarun);
options.LargeScale = true;
options.Display = true; % Show progress on the command line.
options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
[Uhat,output] = cpd(datarun,R,options);

save('EFTFU2FU3partdecodingresult.mat','Uhat','output');
Rest=rankest(datarun);

load('EFTFU2totaldecodingresult.mat');
load('EFTFU3anvec.mat');
load('EFTFU2FU3antime.mat');

%%FU2FU3test
load('EFTFU213conditiondecodingresulttensorlab.mat');
S=Uhat{1,3};
t=Uhat{1,2};
V=Uhat{1,1};




tnew=t([1:8,13],:);
ATA=(S'*S).*(tnew'*tnew).*(V'*V);
ATAni=inv(ATA);
c=load('EFTFU3conditionanh618.mat');
c=fillmissing(c,'linear');
B=c.nii3d(:,[1:8,13],:);
newUhat=cell(1,3);
Ab=zeros(10,1);
for i=1:10
newUhat{1,1}=(Uhat{1,1}(:,i))';
test=Uhat{1,2}';
test=test(:,[1:8,13]);
newUhat{1,2}=test(i,:);
newUhat{1,3}=(Uhat{1,3}(:,i))';
Ab(i)=tmprod(B,newUhat,1:3);
end
Testimation=ATAni*Ab;

FU3est=zeros(67683,13,618);
for i=1:10
    tensorcore=Testimation(i)*reshape(kron(Uhat{1,2}(:,i),Uhat{1,1}(:,i))*(Uhat{1,3}(:,i))',[67683,13,618]);
    FU3est=FU3est+tensorcore;
end

save('FU3estfromFU2.mat','FU3est','-v7.3');
load('FU3estfromFU2.mat');
save('FU3estfromFU2new.mat','FU3est','-v7.3');







%prediction evaluation：correlation index
FU3est=real+randn(5361,197,618)*0.1;
real=FU3total.posttransform;
recon=FU3est;
real1=real(:,:,1);
recon1=FU3est(:,:,1);

[r,p]=corr(real1',recon1');

%prediction evaluation：norm index
error=real-recon;
a=error(:);
b=real(:);

norm1=norm(a,2);
norm2=norm(b,2);
aber=norm1/norm2;

% T = randn(3,5,7);
% U = {randn(11,3),randn(13,5),randn(15,7)};
% S = tmprod(T,U,1:3);

load('EFTFU213conditiondecodingresulttensorlab.mat');
load('EFTFU3conditionanh618.mat');


%%%onetoone
load('EFTFU2FU3anhcombinedecodingresult50.mat');
% S=Uhat{1,3};
% t=Uhat{1,2};
% V=Uhat{1,1};
lambda=result(50).Lambda;
S=result(50).U{3,1};
for i=1:50
Snew(:,i)=lambda(i)*S(:,i);
end
t=result(50).U{2,1};
V=result(50).U{1,1};


lambda=result(50).Lambda;
tnew=t([1:8,13],:);
ATA=(Snew'*Snew).*(tnew'*tnew).*(V'*V);
%ATAni=inv(ATA);
%B=nii3d(:,[1:4,9:13],:);
d=load('EFTBLconditionan618.mat');
BLanctime=fillmissing(d.nii3d(:,[1:4,6:9,11],:),'linear');
B=BLanctime;
B=fillmissing(B,'linear');
newUhat=cell(1,3);
Ab=zeros(50,1);
lambda=result(50).Lambda;
for i=1:50
newUhat{1,1}=(result(50).U{1,1}(:,i))';
test=result(50).U{2,1}';
test=test(:,[1:8,13]);
newUhat{1,2}=test(i,:);
newUhat{1,3}=(result(50).U{3,1}(:,i))';
Ab(i)=lambda(i)*tmprod(B,newUhat,1:3);
%Ab(i)=tmprod(B,newUhat,1:3);
end
Testimation=ATA\Ab;

FU3est=zeros(67683,13,618);
for i=1:50
    %tensorcore=Testimation(i)*reshape(kron(result(10).U{2,1}(:,i),result(10).U{1,1}(:,i))*(result(10).U{3,1}(:,i))',[67683,13,618]);
    tensorcore=lambda(i)*Testimation(i)*reshape(kron(result(50).U{2,1}(:,i),result(50).U{1,1}(:,i))*(result(50).U{3,1}(:,i))',[67683,13,618]);
    FU3est=FU3est+tensorcore;
end





%onetoonecpqrsvdcase
% load('cpqrsvd3predict1conditiontensordecodingfullr50.mat');


load('cpqrsvd2predict1conditiontensordecodingfullr50.mat');
lambda=qrsvdP.lambda;
S=qrsvdP.u{3,1};
for i=1:50
Snew(:,i)=lambda(i)*S(:,i);
end
t=qrsvdP.u{2,1};
V=qrsvdP.u{1,1};


lambda=qrsvdP.lambda;
tnew=t([1:8,13],:);
ATA=(Snew'*Snew).*(tnew'*tnew).*(V'*V);
%ATAni=inv(ATA);
%B=nii3d(:,[1:4,9:13],:);
d=load('EFTBLconditionan618.mat');
BLanctime=fillmissing(d.nii3d(:,[1:4,6:9,11],:),'linear');
B=BLanctime;
B=fillmissing(B,'linear');
newUhat=cell(1,3);
Ab=zeros(50,1);
lambda=qrsvdP.lambda;
for i=1:50
newUhat{1,1}=(qrsvdP.u{1,1}(:,i))';
test=qrsvdP.u{2,1}';
test=test(:,[1:8,13]);
newUhat{1,2}=test(i,:);
newUhat{1,3}=(qrsvdP.u{3,1}(:,i))';
Ab(i)=lambda(i)*tmprod(B,newUhat,1:3);
%Ab(i)=tmprod(B,newUhat,1:3);
end
Testimation=ATA\Ab;

FU3est=zeros(67683,13,618);
for i=1:50
    %tensorcore=Testimation(i)*reshape(kron(result(10).U{2,1}(:,i),result(10).U{1,1}(:,i))*(result(10).U{3,1}(:,i))',[67683,13,618]);
    tensorcore=lambda(i)*Testimation(i)*reshape(kron(qrsvdP.u{2,1}(:,i),qrsvdP.u{1,1}(:,i))*(qrsvdP.u{3,1}(:,i))',[67683,13,618]);
    FU3est=FU3est+tensorcore;
end

save('EFTBL2predict1cpsvdfullr50.mat','FU3est','-v7.3');


















%oto
load('EFTFU2FU3anccombinedecodingresult.mat');
% S=Uhat{1,3};
% t=Uhat{1,2};
% V=Uhat{1,1};
lambda=result(10).Lambda;
S=result(10).U{3,1};
for i=1:10
Snew(:,i)=lambda(i)*S(:,i);
end
t=result(10).U{2,1};
V=result(10).U{1,1};


lambda=result(10).Lambda;
tnew=t([5:8,9],:);
ATA=(Snew'*Snew).*(tnew'*tnew).*(V'*V);
%ATAni=inv(ATA);
%B=nii3d(:,[1:4,9:13],:);
B=BLactime;
B=fillmissing(B,'linear');
newUhat=cell(1,3);
Ab=zeros(10,1);
lambda=result(10).Lambda;
for i=1:10
newUhat{1,1}=(result(10).U{1,1}(:,i))';
test=result(10).U{2,1}';
test=test(:,[5:8,9]);
newUhat{1,2}=test(i,:);
newUhat{1,3}=(result(10).U{3,1}(:,i))';
Ab(i)=lambda(i)*tmprod(B,newUhat,1:3);
%Ab(i)=tmprod(B,newUhat,1:3);
end
Testimation=ATA\Ab;

FU3est=zeros(67683,13,618);
for i=1:10
    tensorcore=Testimation(i)*reshape(kron(result(10).U{2,1}(:,i),result(10).U{1,1}(:,i))*(result(10).U{3,1}(:,i))',[67683,13,618]);
    %tensorcore=lambda(i)*Testimation(i)*reshape(kron(result(10).U{2,1}(:,i),result(10).U{1,1}(:,i))*(result(10).U{3,1}(:,i))',[67683,9,618]);
    FU3est=FU3est+tensorcore;
end
save('BLhconditionestfromFU2FU3.mat','FU3est','-v7.3');


save('FU3conditionestfromFU2angry.mat','FU3est','-v7.3');
load('FU3estfromFU2.mat');
save('FU3estfromFU2new.mat','FU3est','-v7.3');

real=d.nii3d;
real=fillmissing(real,'linear');



a=load('EFTFU2conditionanh618.mat');
b=load('EFTFU3conditionanh618.mat');
load('EFTFU2FU39conditiondecodingresultNASCAR.mat');








%load('EFTBLFU2FU3ancboldombinedecodingresult10100.mat');
%load('cpqrsvd3predict1conditiontensordecodingfastr800.mat');
load('EFTBLFU2FU3ancombinedecodingresult50.mat');
%%duotoone
Tw=result(50).U{4,1};
T=Tw([2,3],:);
S=result(50).U{3,1};
lambda=result(50).Lambda;
for i=1:50
Snew(:,i)=lambda(i)*S(:,i);
end
V=result(50).U{1,1};
ATA=(T'*T).*(Snew'*Snew).*(V'*V);
ATAni=inv(ATA);
b=load('EFTFU2conditionanh618.mat');
c=load('EFTFU3conditionanh618.mat');
FU2total=fillmissing(b.nii3d,'linear');
FU3total=fillmissing(c.nii3d,'linear');
FU2htime=FU2total(:,[9:12],:);
FU3htime=FU3total(:,[9:12],:);
FU2FU3htime=cat(2,FU2htime,FU3htime);
B=FU2FU3htime;
B=fillmissing(B,'linear');
newUhat=cell(1,3);
AB=zeros(50,4);
lambda=result(50).Lambda;
for j=1:4
for i=1:50
newUhat{1,1}=V(:,i)';
newUhat{1,2}=T(:,i)';
newUhat{1,3}=Snew(:,i)';
AB(i,j)=tmprod(B(:,[j,j+4],:),newUhat,1:3);
end
end
testimation=ATAni*AB;
for i=1:50
testimationexclude(i,:)=testimation(i,:)
end
test=testimationexclude';
%test=testimation;
ttotal(9:12,:)=test;
ttotal([1:8,13],:)=result(50).U{2,1};
FU3est=zeros(67683,13,618);
for i=1:50
    tensorcore=lambda(i)*Tw(1,i)*reshape(kron(ttotal(:,i),result(50).U{1,1}(:,i))*(result(50).U{3,1}(:,i))',[67683,13,618]);
    FU3est=FU3est+tensorcore;
end











%forqr
FU2beta1=load('FU2618valence.mat');
FU2beta2=load('FU2618salience.mat');
FU3beta1=load('FU3618valence.mat');
FU3beta2=load('FU3618salience.mat');
load('cpqrsvd3predict1conditiontensordecodingfastr500.mat');
%%duotoone
Tw=qrsvdP.u{4,1};
T=Tw([2,3],:);
S=qrsvdP.u{3,1};
lambda=qrsvdP.lambda;
for i=1:500
Snew(:,i)=lambda(i)*S(:,i);
end
V=qrsvdP.u{1,1};
ATA=(T'*T).*(Snew'*Snew).*(V'*V);
ATAni=inv(ATA);
b=load('EFTFU2conditionanh618.mat');
c=load('EFTFU3conditionanh618.mat');
FU2total=fillmissing(b.nii3d,'linear');
FU3total=fillmissing(c.nii3d,'linear');
FU2htime=FU2total(:,[9:12],:);
FU3htime=FU3total(:,[9:12],:);
FU2FU3htime=cat(2,FU2htime,FU3htime);%beta1 beta2
FU2FU3htime=zeros(67683,12,618);
FU2FU3htime(:,9,:)=FU2beta1.beta1mt;
FU2FU3htime(:,10,:)=FU2beta2.beta2mt;
FU2FU3htime(:,11,:)=FU3beta1.beta1mt;
FU2FU3htime(:,12,:)=FU3beta2.beta2mt;
B=FU2FU3htime;
B=fillmissing(B,'linear');
newUhat=cell(1,3);
AB=zeros(500,6);
lambda=qrsvdP.lambda;
for j=1:6 %6
for i=1:500
newUhat{1,1}=V(:,i)';
newUhat{1,2}=T(:,i)';
newUhat{1,3}=Snew(:,i)';
AB(i,j)=tmprod(B(:,[j,j+6],:),newUhat,1:3);
end
end
testimation=ATAni*AB;
for i=1:500
testimationexclude(i,:)=testimation(i,:)
end
test=testimationexclude';
%test=testimation;
ttotal(9:14,:)=test;
ttotal([1:8,15],:)=qrsvdP.u{2,1};
FU3est=zeros(67683,15,618);
for i=1:500
    tensorcore=lambda(i)*Tw(1,i)*reshape(kron(ttotal(:,i),qrsvdP.u{1,1}(:,i))*(qrsvdP.u{3,1}(:,i))',[67683,15,618]);
    FU3est=FU3est+tensorcore;
end
%save('EFTBL3predict1cpqrfullr50.mat','FU3est','-v7.3');
save('EFTBL3predict1cpsvdfastr300.mat','FU3est','-v7.3');


















%oto
load('EFTFU2FU3anccombinedecodingresult.mat');
% S=Uhat{1,3};
% t=Uhat{1,2};
% V=Uhat{1,1};
lambda=result(10).Lambda;
S=result(10).U{3,1};
for i=1:10
Snew(:,i)=lambda(i)*S(:,i);
end
t=result(10).U{2,1};
V=result(10).U{1,1};
tnew=t([5:8,9],:);
ATA=(Snew'*Snew).*(tnew'*tnew).*(V'*V);
%ATAni=inv(ATA);
%B=nii3d(:,[1:4,9:13],:);
d=load('EFTBLconditionanc618.mat');
BLanctime=fillmissing(d.nii3d(:,[6:9,11],:),'linear');
B=BLanctime;
B=fillmissing(B,'linear');
newUhat=cell(1,3);
Ab=zeros(10,1);
for i=1:10
newUhat{1,1}=(result(10).U{1,1}(:,i))';
test=result(10).U{2,1}';
test=test(:,[5:8,9]);
newUhat{1,2}=test(i,:);
newUhat{1,3}=(result(10).U{3,1}(:,i))';
Ab(i)=lambda(i)*tmprod(B,newUhat,1:3);
%Ab(i)=tmprod(B,newUhat,1:3);
end
Testimation=ATA\Ab;

FU3est=zeros(67683,9,618);
for i=1:10
    %tensorcore=Testimation(i)*reshape(kron(result(10).U{2,1}(:,i),result(10).U{1,1}(:,i))*(result(10).U{3,1}(:,i))',[67683,13,618]);
    tensorcore=lambda(i)*Testimation(i)*reshape(kron(result(10).U{2,1}(:,i),result(10).U{1,1}(:,i))*(result(10).U{3,1}(:,i))',[67683,9,618]);
    FU3est=FU3est+tensorcore;
end










%dto
load('EFTBLFU2FU3ancombinedecodingresult.mat');
b=load('EFTFU2conditionanhc618.mat');
c=load('EFTFU3conditionanhc618.mat');
FU2FU3ntime=cat(2,b.nii3d(:,[13],:),c.nii3d(:,[13],:));
Tw=result(10).U{4,1};
T=Tw([2,3],:);
S=result(10).U{3,1};
lambda=result(10).Lambda;   

for i=1:10
Snew(:,i)=lambda(i)*S(:,i);
end
V=result(10).U{1,1};

ATA=(T'*T).*(Snew'*Snew).*(V'*V);
ATAni=inv(ATA);
B=FU2FU3ntime;
B=fillmissing(B,'linear');
newUhat=cell(1,3);
AB=zeros(10,1);%4
lambda=result(10).Lambda;
for j=1%1：4
for i=1:10
newUhat{1,1}=V(:,i)';
newUhat{1,2}=T(:,i)';
newUhat{1,3}=Snew(:,i)';
AB(i,j)=tmprod(B(:,[j,j+1],:),newUhat,1:3);%[j,j+4]
end
end
testimation=ATAni*AB;
% for i=1:10
% testimationexclude(i,:)=testimation(i,:)
% end
% test=testimationexclude';
test=testimation;
ttotal=zeros(9,10);
ttotal(9,:)=test';%1:4
ttotal([1:8],:)=result(10).U{2,1};%5:8,9
FU3est=zeros(67683,9,618);
for i=1:10
    tensorcore=lambda(i)*Tw(1,i)*reshape(kron(ttotal(:,i),result(10).U{1,1}(:,i))*(result(10).U{3,1}(:,i))',[67683,9,618]);
    FU3est=FU3est+tensorcore;
end

save('EFTBLaestfromBLFU2FU3nctotaldecoding.mat','FU3est','-v7.3');
% %%
% d=load('EFTBLconditionan618.mat');
% real=d.nii3d;
% real=fillmissing(real,'linear');
% norm1=norm(real(:)-FU3est(:),2);
% norm2=norm(real(:));
% aber=norm1/norm2;
% 
% load('FU3conditionestfromFU2.mat');
% load('FU3conditionestfromFU2angry.mat');
% load('FU2conditionestfromFU3.mat');
% 
% 
% 
% load('FU3conditionestfromFU2angry.mat');
% 
% 
% FU3est=X(:,1:13,:);
% %load('EFTBL3predict1cpsvdfullr50.mat');
% 
% FU3est=permute(E,[1,3,2]);



FU3est;
d=load('EFTBLconditionanc618.mat');
real=d.nii3d;
real=fillmissing(real,'linear');
MNI = y_ReadAll('MNI152_T1_3mm_brain_mask.nii');
num_elements=nnz(MNI);
% MNI4 = repmat(MNI, [1, 1, 1, 155]);
%197 202
MNI = logical(MNI);

%add noise
voxel_values = cell(0);
for i = 1:618
  
    i

    % ??y_ReadAll????nii????
   % con_angry_data = sum(FU3est(:,1:4,i),2);
     con_angry_data = sum(real(:,1:4,i),2);
    conangry=nan(61,73,61);
    conangry(MNI)=con_angry_data;
   % con_neutral_data = sum(FU3est(:,5:8,i),2);
    con_neutral_data = sum(real(:,6:9,i),2);
    conneutral=nan(61,73,61);
    conneutral(MNI)=con_neutral_data;
       con_angryest_data = sum(FU3est(:,1:4,i),2);
    conangryest=nan(61,73,61);
    conangryest(MNI)=con_angryest_data;
             con_neutralest_data = sum(FU3est(:,5:8,i),2);
    conneutralest=nan(61,73,61);
    conneutralest(MNI)=con_neutralest_data;
                 con_controlest_data = sum(FU3est(:,9,i),2);
    concontrolest=nan(61,73,61);
    concontrolest(MNI)=con_controlest_data;
       con_control_data = sum(real(:,11,i),2);
    concontrol=nan(61,73,61);
    concontrol(MNI)=con_control_data;
%     beta1 = nan(size(con_happy_data));
%     beta2 = nan(size(con_happy_data));
% 
% % ??????
%     for j = 1:numel(con_happy_data)
% 
%          X = [-1, 1, 1; 0, 0, 1; 1, 1, 1];
%          y1 = con_angry_data(j);
%          y2 = con_neutral_data(j); 
%          y3 = con_happy_data(j);
% 
%   % ??NaN
%          if isnan(y1) || isnan(y2) || isnan(y3) 
%             continue;
%          end
%   
%   % ??????
%          y = [y1;y2;y3];
%   
%  % // ????y = [0.1;0.2;0.3];
%  
%          b = X\y;    
% 
%   %// ????
%          beta1(j) = b(1);
%          beta2(j) = b(2);
%   
%     end
% 
% % ?????  
%     beta1 = reshape(beta1, size(con_happy_data));
%     beta2 = reshape(beta2, size(con_happy_data));

    % ??????????
     %voxel_values{end+1} = con_angry_data - con_neutral_data ;
    %voxel_values{end+1} = beta2;
    %voxel_values{end+1} = con_happy_data+con_angry_data - 2*con_neutral_data ;
    voxel_values{end+1} =conneutralest+conangry-2*concontrol;
    %voxel_values{end+1} = conhappy-concontrol;
 
    %voxel_values{end+1} = con_angry_data-con_neutral_data;
end

% ??voxel_values???1x1159?cell????????????61x73x61?????

% ?????61x73x61????????T?
% ??voxel_values???1x1159?cell????????????61x73x61?????

% ?????????
voxel_size = size(voxel_values{1});

% ?????61x73x61????????T?
t_values_matrix = zeros(voxel_size);

% ??????
for i = 1:prod(voxel_size)
    i
    % ???????1159???????
    voxel_data_vector = zeros(numel(voxel_values), 1);
    
    % ??voxel_values????????????
    for j = 1:numel(voxel_values)
        % ???????????????????
        current_voxel_data = voxel_values{j};
        % ?????????????
        voxel_data_vector(j) = current_voxel_data(i);
    end
    
    % ??T-test??
    [~, ~, ~, stats] = ttest(voxel_data_vector);
    
    % ????T?????????
    t_values_matrix(i) = stats.tstat;
end
[sub_datah,~,~,headerh]=y_ReadAll('con_happy.nii');

y_Write(t_values_matrix, headerh,'EFT_ses-baseline_frneutralpredictfromBLFU2FU3ac_Tmap.nii');

data=y_ReadAll('EFT_ses-baseline_aminuscpmrestonetoone_Tmap.nii');
datanew=data*1.5;
%y_Write(datanew, headerh,'testscale.nii');
y_Write(datanew, headerh,'EFT_ses-baseline_aminuscpmrestonetoone_Tmap.nii');
% total=E(:,:,:,2)-fillmissing(b.nii3d,'linear');
% n=norm(total(:),2);
% open=b.nii3d;
% open=fillmissing(open,'linear');
% norm(open(:),2);