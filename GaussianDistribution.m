function GaussPDF = GaussianDistribution(X, Mu, Sigma)

% X is vectorized: m-by-1

GaussPDF = exp(-(X-Mu).^2/Sigma^2)/sqrt(2*pi)/Sigma;