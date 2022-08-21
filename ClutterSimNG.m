function [d] = ClutterSimNG(type,J,K,N,rho,a,b)

% Calculate the interference return signal. Modeled as d = sqrt(tau)*x.
% Define x, the interference speckle. This is modeled as circular complex
% Gaussian.
mu_x = zeros(1,N);                              % Expected values of Re(x) and Im(x).
% gamma_x = [1 1];                                % Variance of x.
M = zeros(N,N);
% mu = zeros(N);
for ii = 1:N
%     mu(ii) = 0;
    for jj = 1:N
        M(ii,jj) = rho^(abs(ii-jj));            %Covariance of x
    end
end
x_mv = mvnrnd(mu_x,M,2*(K+J));                  % Generate Re(x) and Im(x) as Gaussian RVs.
x_Kplus1 = x_mv(1:2:end,:)+1j*x_mv(2:2:end,:);  % Generate x values for CUT and training set.
x = x_Kplus1(1:J,:);                              % Define x values for CUT
% x_train = x_Kplus1(J+1:end,:);                    % Define x values for training set

% Model clutter signal from speckle and texture.
% Define tau, the interference texture. If this is set to 1 d is Gaussian.
% To make d K-distributd, set tau to be Gamma distributed.
switch type
    case 'gaussian'        
        tau = 1;            % Constant texture value. 
    case 'K'
%         tau = gamrnd(a,b,[J,N]);
        tau = repmat(gamrnd(a,b,[J,1]),[1,N]);
    case 'pareto1'
        tau = repmat(1./gamrnd(a,b,[J,1]),[1,N]);
end
d = sqrt(tau).*x;    % Model test sample clutter (d = sqrt(tau)*x).
% d = (tau.^0.5).*x;    % Model test sample clutter (d = sqrt(tau)*x).
% d = x;    % Model test sample clutter (d = sqrt(tau)*x).
% d_train = x_train;  % Model training sample clutter (d = x).