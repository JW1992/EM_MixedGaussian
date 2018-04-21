close all;
clear all;
clc;
set(0, 'DefaultAxesFontSize', 20);
set(0, 'DefaultTextFontSize', 20);

%%%%%%%%%%%%%%%%%%%%
% Inputs to the data sampler
SampleNum = 1E4;
GaussianModelNum = 2;
Mu = [-8, 5];
Sigma = [5, 4];
% Determine the model used for random samples by comparing with thr
ModelThr = [0, 0.5];
%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%
% EM iteration
maxIteration = 1E3;
% Squared sum threshold for convergence
errorThr = 1E-4;
likeliHoodHistory = zeros(1, maxIteration);
%%%%%%%%%%%%%%%

X = zeros(SampleNum, 1);
rndNum = rand(SampleNum, 1);

%%%%%%%%%%%%%%%%%%%%
% Generating the random samples based on the given model
for i = 1:SampleNum
    model = sum(rndNum(i)>ModelThr);
    model = max(model, 1);
    X(i) = normrnd(Mu(model), Sigma(model));
end
%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%
% Check the distribution of X
figure;
hist(X, 30);
xlabel('X');
ylabel('frequency');
hold on;
%%%%%%%%%%%

%%%%%%%%
% Initial guess
% Mu_EM = ones(1, GaussianModelNum);
Mu_EM = [1, 2];
Sigma_EM = ones(1, GaussianModelNum);
% soft guess of the models each sample might belong to.
% In the form of weights.
modelEM = ones(SampleNum, GaussianModelNum)/GaussianModelNum;
% This relative frequency is the latent variable being guessed.
relModelFreq_EM = ones(GaussianModelNum, 1)/GaussianModelNum;

bHardGuess = 1;

% Draw the initially guessed models
XDraw = -20:0.25:20;
YDraws = zeros(161, GaussianModelNum);
for i = 1:GaussianModelNum
    YDraws(:,i) = relModelFreq_EM(i)*SampleNum ...
        * GaussianDistribution(XDraw', Mu_EM(i), Sigma_EM(i));
    plot(XDraw, YDraws(:,i), 'g', 'LineWidth', 1.2);
end
hold off;
%%%%%%%%

for iteration = 1:maxIteration
    likeliHoodHistory(iteration) = LogLikeliHood(X, Mu_EM, Sigma_EM, ...
        relModelFreq_EM, modelEM);
    
    %%%%%%%%%%%%%%%%%%%
    % E-step
    for j = 1:GaussianModelNum
        modelEM(:, j) = GaussianDistribution(X, Mu_EM(j), Sigma_EM(j)) ...
            .* relModelFreq_EM(j);
    end
    
    if bHardGuess==0
        for j = 1:SampleNum
            modelEM(j, :) = modelEM(j, :)/sum(modelEM(j, :));
        end
    else
        for j = 1:SampleNum
            [M, I] = max(modelEM(j, :));
            modelEM(j, :) = zeros(1, GaussianModelNum);
            modelEM(j, I) = 1;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%
    % M-step
    n_k = sum(modelEM);
    relModelFreq_EM = n_k/SampleNum;
    relModelFreq_EM = relModelFreq_EM/sum(relModelFreq_EM);
    for j = 1:GaussianModelNum
        Mu_EM(j) = sum(modelEM(:, j).*X)/n_k(j);
        Sigma_EM(j) = sqrt(sum(modelEM(:, j).*(X-Mu_EM(j)).^2)/n_k(j));
    end
    
    
    if iteration>1 && (likeliHoodHistory(iteration)-likeliHoodHistory(iteration-1))^2 < errorThr
        break;
    end
end

figure(1); hold on;
for i = 1:GaussianModelNum
    YDraws(:,i) = relModelFreq_EM(i)*SampleNum ...
        * GaussianDistribution(XDraw', Mu_EM(i), Sigma_EM(i));
    plot(XDraw, YDraws(:,i), 'r', 'LineWidth', 1.2);
end
hold off;

figure; plot(likeliHoodHistory(3:iteration));
xlabel('Iteration');
ylabel('Log likelihood');
set(gca, 'YScale', 'log');

disp(Mu_EM);
disp(Sigma_EM);
disp(relModelFreq_EM);

