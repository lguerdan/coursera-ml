function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
theta = fliplr(theta);

%calculate regularized cost
cost = (1 / (2*m)) * sum(((X * theta) - y) .^ 2);
regularization = (lambda / (2 * m )) * sum(theta(2:end) .^ 2);
J = cost + regularization;

%calculate gradient
grad = ((1 / m) * ((X * theta)- y)' * X);

grad(2:end) = grad(2:end) + ((lambda / m) * theta(2:end)');
grad = grad(:);

end
