function [detection,eta,eta_0] = detector(alg,z,p,S,P_fa,K,N)

%% Calculate the test statistic and set the threshold

                        
switch alg
    case 'glrt'
        eta = (abs(conj(z)/S*p.')^2)/((1+...
            (conj(z)/S*z.'))*(conj(p)/S*p.'));                  % Test statistic eta for the GLRT detector
%         l_0 = 1/(P_fa^(1/(K+1-N)));                             % Set threshold l_0 from desired PFA, sample support K, and CPI pulse number N
%         eta_0 = (l_0-1)/l_0;                                    % Convert threshold to eta_0 for comparison with the test statistic
        
%         eta_0 = 0.3553;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, target_tau = 1.5
%         eta_0 = 0.3253;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, target_tau = 1.3
        
%         eta_0 = 0.0727;                                         % GLRT detector threshold for: P_fa = 1e-2, N = 15, K = 5*N
        
%         eta_0 = 0.0446;                                         % GLRT detector threshold for: P_fa = 1e-2, N = 25, K = 5*N
        
%         eta_0 = 0.197;                                          % GLRT detector threshold for: P_fa = 1e-2, N = 5, K = 5*N

%GLRT thresholds for the non-Gaussian cases
%         eta_0 = 0.1607;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 25, K = 3*N, type = 'K', a = 0.5, b = 4

        eta_0 = 0.5755;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 0.5, b = 1,4
%         eta_0 = 0.4273;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 1.5, b = 1
%         eta_0 = 0.3423;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 4.5, b = 1

%         eta_0 = 0.1361;%0.1071;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 15, K = 5*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.2601;%0.1071;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 15, K = 5*N, type = 'K', a = 0.5, b = 1
        
%         eta_0 = 0.0847;%0.0727;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 25, K = 5*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.1096;%0.0727;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 25, K = 4*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.1526;%0.0727;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 25, K = 3*N, type = 'K', a = 4.5, b = 1

%         eta_0 = 0.1682;%0.0727;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 25, K = 5*N, type = 'K', a = 0.5, b = 1
%         eta_0 = 0.2101;%0.0727;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 25, K = 4*N, type = 'K', a = 0.5, b = 1
%         eta_0 = 0.2861;%0.0727;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 25, K = 3*N, type = 'K', a = 0.5, b = 1




        
        
%         eta_0 = 0.4413;%0.2803;                                         % BAD - GLRT detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 0.5, b = 1,4
%         eta_0 = 0.3653;%0.2803;                                         % BAD - GLRT detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 1.5, b = 1
%         eta_0 = 0.3203;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 4.5, b = 1

%         eta_0 = 0.2998;%0.3203;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 4.5, b = 1, rho = 0.5
%         eta_0 = 0.2998;%0.3203;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 4.5, b = 1, rho = 0.1
%         eta_0 = 0.4013;%0.4413;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 0.5, b = 1, rho = 0.5
%         eta_0 = 0.4063;%0.4013;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 0.5, b = 1, rho = 0.1
        
%         eta_0 = 0.1161;%0.1071;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 15, K = 5*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.1471;%0.1071;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 15, K = 5*N, type = 'K', a = 0.5, b = 1

%         eta_0 = 0.0697;%0.0727;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 25, K = 5*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.0902;%0.1012;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 25, K = 4*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.1302;%0.1532;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 25, K = 3*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.0852;%0.0727;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 25, K = 5*N, type = 'K', a = 0.5, b = 1
%         eta_0 = 0.1112;%0.0727;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 25, K = 4*N, type = 'K', a = 0.5, b = 1
%         eta_0 = 0.1532;%0.1472;                                         % GLRT detector threshold for: P_fa = 1e-3, N = 25, K = 3*N, type = 'K', a = 0.5, b = 1


    case 'amf'
        eta = (abs(conj(z)/S*p.')^2)/(conj(p)/S*p.');           % Test statistic eta for the AMF detector
%         eta_0 = ((K+1)/(K-N+1))*(((P_fa)^(-1/(K-N+2)))-1);      % Closed form threshold approximation for AMF detector
        
%         eta_0 = 0.7167;                                         % AMF detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, target_tau = 1.5
%         eta_0 = 0.6167;                                         % AMF detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, target_tau = 1.3
        
%         eta_0 = 0.0971;                                         % AMF detector threshold for: P_fa = 1e-2, N = 15, K = 5*N
%         eta_0 = 0.1468;                                         % AMF detector threshold for: P_fa = 1e-3, N = 15, K = 5*N
%         eta_0 = 0.1995;                                         % AMF detector threshold for: P_fa = 1e-4, N = 15, K = 5*N
        
%         eta_0 = 0.0576;                                         % AMF detector threshold for: P_fa = 1e-2, N = 25, K = 5*N
%         eta_0 = 0.0883;                                         % AMF detector threshold for: P_fa = 1e-3, N = 25, K = 5*N
%         eta_0 = 0.1277;                                         % AMF detector threshold for: P_fa = 1e-3, N = 25, K = 4*N
%         eta_0 = 0.2133;%0.2220;                                         % AMF detector threshold for: P_fa = 1e-3, N = 25, K = 3*N
%         eta_0 = 0.1182;                                         % AMF detector threshold for: P_fa = 1e-4, N = 25, K = 5*N

%         eta_0 = 0.294;                                          % AMF detector threshold for: P_fa = 1e-2, N = 5, K = 5*N
%         eta_0 = 0.472;                                          % AMF detector threshold for: P_fa = 1e-3, N = 5, K = 5*N
%         eta_0 = 0.675;                                          % AMF detector threshold for: P_fa = 1e-4, N = 5, K = 5*N

%AMF thresholds for the non-Gaussian cases
%         eta_0 = 0.4303;                                         % AMF detector threshold for: P_fa = 1e-3, N = 25, K = 3*N, type = 'K', a = 0.5, b = 4

        eta_0 = 3.222;                                          % AMF detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 0.5, b = 1,4
%         eta_0 = 1.202;%0.472;                                   % AMF detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 1.5, b = 1
%         eta_0 = 0.737;%0.472;                                   % AMF detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 4.5, b = 1
 
%         eta_0 = 0.2258;%0.1768;                                         % AMF detector threshold for: P_fa = 1e-3, N = 15, K = 5*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.9758;%0.1468;                                         % AMF detector threshold for: P_fa = 1e-3, N = 15, K = 5*N, type = 'K', a = 0.5, b = 1

%         eta_0 = 0.1393;%0.0983;                                         % AMF detector threshold for: P_fa = 1e-3, N = 25, K = 5*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.1958;%0.0883;                                         % AMF detector threshold for: P_fa = 1e-3, N = 25, K = 4*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.3558;%0.0883;                                         % AMF detector threshold for: P_fa = 1e-3, N = 25, K = 3*N, type = 'K', a = 4.5, b = 1

%         eta_0 = 0.5683;%0.0883;                                         % AMF detector threshold for: P_fa = 1e-3, N = 25, K = 5*N, type = 'K', a = 0.5, b = 1
%         eta_0 = 0.9758;%0.0883;                                         % AMF detector threshold for: P_fa = 1e-3, N = 25, K = 4*N, type = 'K', a = 0.5, b = 1
%         eta_0 = 2.0058;%0.0883;                                         % AMF detector threshold for: P_fa = 1e-3, N = 25, K = 3*N, type = 'K', a = 0.5, b = 1


        
        
        
%         eta_0 = 1.822;%0.472;                                          % BAD - AMF detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 0.5, b = 1,4
%         eta_0 = 1.072;%0.472;                                          % BAD - AMF detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 1.5, b = 1
%         eta_0 = 0.687;%0.472;                                          % BAD - AMF detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 4.5, b = 1

%         eta_0 = 0.562;%0.687;%0.472;                                          % AMF detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 4.5, b = 1, rho = 0.5
%         eta_0 = 0.562;%0.687;%0.472;                                          % AMF detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 4.5, b = 1, rho = 0.1
%         eta_0 = 1.422;%1.822;                                          % AMF detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 0.5, b = 1, rho = 0.5
%         eta_0 = 1.392;%1.422;                                          % AMF detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 0.5, b = 1, rho = 0.1
        
%         eta_0 = 0.1758;%0.1468;                                         % AMF detector threshold for: P_fa = 1e-3, N = 15, K = 5*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.3358;%0.1468;                                         % AMF detector threshold for: P_fa = 1e-3, N = 15, K = 5*N, type = 'K', a = 0.5, b = 1

%         eta_0 = 0.0993;%0.0883;                                         % AMF detector threshold for: P_fa = 1e-3, N = 25, K = 5*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.1433;%0.1453;                                         % AMF detector threshold for: P_fa = 1e-3, N = 25, K = 4*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.2443;%0.1433;                                         % AMF detector threshold for: P_fa = 1e-3, N = 25, K = 3*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.1683;%0.0883;                                         % AMF detector threshold for: P_fa = 1e-3, N = 25, K = 5*N, type = 'K', a = 0.5, b = 1
%         eta_0 = 0.2283;%0.0883;                                         % AMF detector threshold for: P_fa = 1e-3, N = 25, K = 4*N, type = 'K', a = 0.5, b = 1
%         eta_0 = 0.4183;%0.2483;                                         % AMF detector threshold for: P_fa = 1e-3, N = 25, K = 3*N, type = 'K', a = 0.5, b = 1

    case 'ace'
        eta = (abs(conj(z)/S*p.')^2)/...
            ((conj(p)/S*p.')*(conj(z)/S*z.'));                  % Test statistic eta for the ACE detector
%         eta_0 = ((1-P_fa^(1/(K-N+1)))/(1-((K-N+1)/(K+1))*...
%             (P_fa)^(1/(K-N+1))));                               % Closed form threshold approximation for ACE detector 
        
%         eta_0 = 0.3273;                                         % ACE detector threshold for: P_fa = 1e-2, N = 15, K = 5*N
%         eta_0 = 0.4419;                                         % ACE detector threshold for: P_fa = 1e-3, N = 15, K = 5*N
%         eta_0 = 0.5438;                                         % ACE detector threshold for: P_fa = 1e-4, N = 15, K = 5*N
        
%         eta_0 = 0.21;                                           % ACE detector threshold for: P_fa = 1e-2, N = 25, K = 5*N
%         eta_0 = 0.296;                                          % ACE detector threshold for: P_fa = 1e-3, N = 25, K = 5*N
%         eta_0 = 0.31;                                           % ACE detector threshold for: P_fa = 1e-3, N = 25, K = 4*N
%         eta_0 = 0.3400;%0.3300;                                         % ACE detector threshold for: P_fa = 1e-3, N = 25, K = 3*N
%         eta_0 = 0.367;                                          % ACE detector threshold for: P_fa = 1e-4, N = 25, K = 5*N

%         eta_0 = 0.732;                                          % ACE detector threshold for: P_fa = 1e-2, N = 5, K = 5*N
%         eta_0 = 0.854;                                          % ACE detector threshold for: P_fa = 1e-3, N = 5, K = 5*N
%         eta_0 = 0.92;                                           % ACE detector threshold for: P_fa = 1e-4, N = 5, K = 5*N

%ACE thresholds for the non-Gaussian cases
%         eta_0 = 0.3400;                                         % ACE detector threshold for: P_fa = 1e-3, N = 25, K = 3*N, type = 'K', a = 0.5, b = 4
         
%           eta_0 = 0.854;                                        % ACE detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 4.5, b = 1
%             eta_0 = 0.869;                                      % ACE detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 1.5, b = 1
            eta_0 = 0.894;                                      % ACE detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 0.5, b = 1,4
   
%         eta_0 = 0.4519;                                         % ACE detector threshold for: P_fa = 1e-3, N = 15, K = 5*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.5200;                                         % ACE detector threshold for: P_fa = 1e-3, N = 15, K = 5*N, type = 'K', a = 0.5, b = 1

%         eta_0 = 0.306;                                         % ACE detector threshold for: P_fa = 1e-3, N = 25, K = 5*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.3145;                                         % ACE detector threshold for: P_fa = 1e-3, N = 25, K = 4*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.3515;                                         % ACE detector threshold for: P_fa = 1e-3, N = 25, K = 3*N, type = 'K', a = 4.5, b = 1

%         eta_0 = 0.354;                                         % ACE detector threshold for: P_fa = 1e-3, N = 25, K = 5*N, type = 'K', a = 0.5, b = 1
%         eta_0 = 0.3820;                                         % ACE detector threshold for: P_fa = 1e-3, N = 25, K = 4*N, type = 'K', a = 0.5, b = 1
%         eta_0 = 0.4295;                                         % ACE detector threshold for: P_fa = 1e-3, N = 25, K = 3*N, type = 'K', a = 0.5, b = 1





%         eta_0 = 0.819;%0.854;                                          % BAD - ACE detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 0.5, b = 1,4
%         eta_0 = 0.844;%0.854;                                          % BAD - ACE detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 1.5, b = 1
%         eta_0 = 0.854;                                          % BAD - ACE detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 4.5, b = 1
           
%         eta_0 = 0.844;                                          % ACE detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 4.5, b = 1, rho = 0.5
%         eta_0 = 0.844;                                          % ACE detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 4.5, b = 1, rho = 0.1
%         eta_0 = 0.814;%0.844;                                          % ACE detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 0.5, b = 1, rho = 0.5
%         eta_0 = 0.814;                                          % ACE detector threshold for: P_fa = 1e-3, N = 5, K = 5*N, type = 'K', a = 0.5, b = 1, rho = 0.1
        
%         eta_0 = 0.4419;                                         % ACE detector threshold for: P_fa = 1e-3, N = 15, K = 5*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.4219;                                         % ACE detector threshold for: P_fa = 1e-3, N = 15, K = 5*N, type = 'K', a = 0.5, b = 1
        
%         eta_0 = 0.296;                                         % ACE detector threshold for: P_fa = 1e-3, N = 25, K = 5*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.312;                                         % ACE detector threshold for: P_fa = 1e-3, N = 25, K = 4*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.337;%0.332                                         % ACE detector threshold for: P_fa = 1e-3, N = 25, K = 3*N, type = 'K', a = 4.5, b = 1
%         eta_0 = 0.286;                                         % ACE detector threshold for: P_fa = 1e-3, N = 25, K = 5*N, type = 'K', a = 0.5, b = 1
%         eta_0 = 0.299;                                         % ACE detector threshold for: P_fa = 1e-3, N = 25, K = 4*N, type = 'K', a = 0.5, b = 1
%         eta_0 = 0.332;%0.329                                         % ACE detector threshold for: P_fa = 1e-3, N = 25, K = 3*N, type = 'K', a = 0.5, b = 1
        
    case 'rao'
        eta = (abs(conj(z)/(z.'*conj(z)+S)*p.')^2)/...
            (conj(p)/(z.'*conj(z)+S)*p.');                      % Test statistic eta for the Rao detector
        eta_0 = 1-P_fa^(1/K);                                   % Compute threshold for Rao detector from desired PFA
end

if eta > eta_0
    detection = 1;
else
    detection = 0;
end
    
    