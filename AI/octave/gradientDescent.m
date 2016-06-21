function [theta]  = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

  m = length(y); % number of training examples

  for iter = 1:num_iters
    J = ((X * theta - y)' * X) .* (alpha/m);
    theta = theta - J;
    % uncomment for debugging, cost should go down with each iteration
    % computeCostMulti(X, y, theta)
  end
end
