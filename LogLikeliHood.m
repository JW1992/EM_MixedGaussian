function logLH = ...
    LogLikeliHood(X, Mu_EM, Sigma_EM, relModelFreq, modelEM)

%%%%%%%%%%%%%%%
% loglikelihood based on the hard guess.
m = size(X,1);
modelNum = size(Mu_EM,2);
logLH = 0;

for i = 1:m
    temp = 0;
    for j = 1:modelNum
        temp = temp + GaussianDistribution(X(i), Mu_EM(j), Sigma_EM(j)) ...
            * modelEM(i, j) * relModelFreq(j);
    end
    logLH = logLH + log(temp);
end


