clc
clear all
% close all
%% Simulate the scenario and return signal

alg = 'glrt';   % This sets the type of adaptive detector that will be applied.
type = 'K';
sim_num = 1e5;  % The number of scenarios to simulate.
gen_data = true;% Flag to decide whether or not to generate new data.

PRI = 1e-8;     % Pulse repetition interval.
f_d = 2e7;      % Doppler frequency.
N = 5;         % Number of pulses per CPI.
K = 5*N;        % Number of secondary data samples used for parameter estimation.
i = 1:N;        % Pulse number.
P_fa = 1e-3;    % Desired probability of false alarm.
% SNIR = 25;      % Target SNIR in dB
SNIR = [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35];      % Target SNIR in dB
CT = 'constant';% Clutter texture type. Either 'constant' for Guassian clutter or 'gamma' for K-distributed clutter.
v = 0.1;        % Texture order parameter. Currently only used of texture is gamma distributed.
rho = 0.999;%0.9;      % The one-lag correlation coefficient.
tau = 1.0;        % Clutter texture
% a = 0.5;%100;%4;
% a = 1.5;
a = 0.5;%1.5;
% b = 4;%1/3;
b = 1;
theta_ma = 0.20;%0.035; % Target misalignment


if gen_data
    % Generate clutter data
%     [d,d_train,M] = ClutterSim(sim_num,K,N,rho,tau);
    [d] = ClutterSimNG(type,sim_num,K,N,rho,a,b);
    d_train = d(1:K,:);
    
%     d = zeros(sim_num,1,N);
%     d_train = zeros(sim_num,K,N);
%     for jj = 1:sim_num
%         [d(jj,:,:),d_train(jj,:,:)] = ClutterSim(K,N,rho);
%     end
else
    temp = load("GE_Data_rho_0_9_K_100_N_16.mat");
    d = temp.d;
    clear temp
    temp = load("GE_Data_rho_0_9_K_100_N_16_training_samples.mat");
    d_train = temp.d_train;
    clear temp
end


% Estimate the clutter mean and covariance
mu_est = sum(d_train,'all')/(K*N);                                      % Estimated clutter mean
% mu_est = sum(d_train,[2,3])/(K*N);                                      % Estimated clutter mean
cov_est = zeros(K,N,N);
% for ll = 1:length(d_train)
%     z_train = squeeze(d_train(ll,:,:));
for kk = 1:K
%         cov_est(ll,kk,:,:) = (d_train(kk,:)'-mu_est)*(d_train(kk,:)-mu_est);
    cov_est(kk,:,:) = (d_train(kk,:)'-mu_est)*(d_train(kk,:)-mu_est);
%     cov_est(kk,:,:) = d_train(kk,:)'*d_train(kk,:);
end
%     ll
% end
M_est = squeeze(sum(cov_est,1))/K;                                      % Estimated clutter covariance
S = K*M_est;                                                            % Parameter S = KM
% S = 2*K*M;
clear cov_est;

% Model the target return signal based on steering vector p and amplitude alpha.
p = exp(-1j*2*pi*i*f_d*PRI);                            % Temporal steering vector.
p_ma = exp(-1j*2*pi*i*(f_d*PRI+theta_ma));              % Misaligned target steering vector

z_target = zeros(length(SNIR),sim_num,N);
for ii = 1:length(SNIR)
    alpha = sqrt(db2mag(SNIR(ii))/(conj(p)/(d'*d/sim_num)*p.'));% Unknown amplitude. Scale based on desired SNIR. Constant for non-fluctuating targets, circular complex gaussian for Swerling 1.
    s = alpha.*p;                                           % Target return signal.
    s_ma = alpha.*p_ma;                                     % Mismatched target return signal.

    % Model the signal under test. Here I model both the scenario in which the
    % target is present and the one in which the target is absent.
    z_target(ii,:,:) = squeeze(s_ma)+sqrt(tau)*d; %Signal under test if target is present.
    % z_target = squeeze(s)+sqrt(tau)*d; %Signal under test if target is present.
end
z_notarget = sqrt(tau)*d; %Signal under test if no target is present.

%Calculate the angle between p and p_ma
num = abs(conj(p)*inv(d'*d/sim_num)*p_ma.')^2;
den = ((conj(p)*inv(d'*d/sim_num)*p.')*(conj(p_ma)*inv(d'*d/sim_num)*p_ma.'));
cos2theta = num/den;

% Calculate the SNIR of the signal to make verify it.
%     SNR = db((abs(alpha)^2)*conj(p)*inv(M_est)*p.');


%% Calculate the test statistic and set the threshold
tally_glrt = zeros(length(SNIR),1); 
tally_amf = zeros(length(SNIR),1); 
tally_ace = zeros(length(SNIR),1); 
FA_glrt = zeros(length(SNIR),1); 
FA_amf = zeros(length(SNIR),1); 
FA_ace = zeros(length(SNIR),1); 
for jj=1:length(SNIR)
    for zz = 1:length(d)-K
    %     z_2 = z_notarget(zz,:)/max(z_notarget(zz,:)); 
    %     z = squeeze(s(zz,:,:)).'+d(zz,:)/max(abs(z_notarget(zz,:)));
        z_2 = z_notarget(zz,:); 
        z = z_target(jj,zz,:);
        z = z(1,:);


        % Estimate the clutter mean and covariance
        mu_est = sum(d(zz+1:zz+K,:),'all')/(K*N);                                      % Estimated clutter mean
        % mu_est = sum(d_train,[2,3])/(K*N);                                      % Estimated clutter mean
        cov_est = zeros(K,N,N);
        % for ll = 1:length(d_train)
        %     z_train = squeeze(d_train(ll,:,:));
        for kk = 1:K
        %         cov_est(ll,kk,:,:) = (d_train(kk,:)'-mu_est)*(d_train(kk,:)-mu_est);
            cov_est(kk,:,:) = (d(zz+kk,:)'-mu_est)*(d(zz+kk,:)-mu_est);
        %     cov_est(kk,:,:) = d_train(kk,:)'*d_train(kk,:);
        end
        %     ll
        % end
        M_est = squeeze(sum(cov_est,1))/K;                                      % Estimated clutter covariance
%         M_est = (d(zz+1:zz+K,:)'-mu_est)*(d(zz+1:zz+K,:)-mu_est)/K;
        S = K*M_est;                                                            % Parameter S = KM
    %     S = 2*K*M;
        clear cov_est;


        [detection_glrt,eta_glrt,eta_0_glrt] = detector('glrt',z,p,S,P_fa,K,N);
        tally_glrt(jj) = tally_glrt(jj) + detection_glrt;
        [detection_glrt,eta_glrt_FA,eta_0_glrt_FA] = detector('glrt',z_2,p,S,P_fa,K,N);
        FA_glrt(jj) = FA_glrt(jj) + detection_glrt;

        [detection_amf,eta_amf,eta_0_amf] = detector('amf',z,p,S,P_fa,K,N);
        tally_amf(jj) = tally_amf(jj) + detection_amf;
        [detection_amf,eta_amf_FA,eta_0_amf_FA] = detector('amf',z_2,p,S,P_fa,K,N);
        FA_amf(jj) = FA_amf(jj) + detection_amf;

        [detection_ace,eta_ace,eta_0_ace] = detector('ace',z,p,S,P_fa,K,N);
        tally_ace(jj) = tally_ace(jj) + detection_ace;
        [detection_ace,eta_ace_FA,eta_0_ace_FA] = detector('ace',z_2,p,S,P_fa,K,N);
        FA_ace(jj) = FA_ace(jj) + detection_ace;
        
        jj
        zz
    end
end
zz
% switch alg
%     case 'glrt'
%         eta = (abs(conj(z)*inv(S)*p.')^2)/((1+...
%             (conj(z)*inv(S)*z.'))*(conj(p)*inv(S)*p.'));        % Test statistic eta for the GLRT detector
%         l_0 = 1/(P_fa^(1/(K+1-N)));                             % Set threshold l_0 from desired PFA, sample support K, and CPI pulse number N
%         eta_0 = (l_0-1)/l_0;                                    % Convert threshold to eta_0 for comparison with the test statistic
%         
%         eta = (abs(conj(z)/S*p.')^2)/((1+...
%             (conj(z)/S*z.'))*(conj(p)/S*p.'));        % Test statistic eta for the GLRT detector
%         l_0 = 1/(P_fa^(1/(K+1-N)));                             % Set threshold l_0 from desired PFA, sample support K, and CPI pulse number N
%         eta_0 = (l_0-1)/l_0;                                    % Convert threshold to eta_0 for comparison with the test statistic
%     case 'amf'
%         eta = (abs(conj(z)*inv(S)*p.')^2)/(conj(p)*inv(S)*p.'); % Test statistic eta for the AMF detector
%         eta_0 = ((K+1)/(K-N+1))*(((P_fa)^(-1/(K-N+2)))-1);      % Closed form threshold approximation for AMF detector
%     case 'ace'
%         eta = (abs(conj(z)*inv(S)*p.')^2)/...
%             ((conj(p)*inv(S)*p.')*(conj(z)*inv(S)*z.'));        % Test statistic eta for the ACE detector
%         eta_0 = ((1-P_fa^(1/(K-N+1)))/(1-((K-N+1)/(K+1))*...
%             (P_fa)^(1/(K-N+1))));                               % Closed form threshold approximation for ACE detector 
%     case 'rao'
%         eta = (abs(conj(z)*inv(z.'*conj(z)+S)*p.')^2)/...
%             (conj(p)*inv(z.'*conj(z)+S)*p.');                   % Test statistic eta for the Rao detector
%         eta_0 = 1-P_fa^(1/K);                                   % Compute threshold for Rao detector from desired PFA
% end
% 
% 
% % cov = eye(K)+rand(K).*(ones(K)-eye(K));