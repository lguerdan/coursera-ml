function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

cost = 0.0;

for i = 1:m
    A_1 = [1 X(i,:)];
    output = y(i);

    % compute hypothesis
    Z_2 = A_1 * transpose(Theta1);
    A_2 = [1, sigmoid(Z_2)];
    Z_3 = A_2 * transpose(Theta2);
    A_3 = sigmoid(Z_3);

    % format labled output parameter
    K = length(Z_3);
    YV = zeros(K,1);
    YV(output) = 1;
    
    %propogate error back through network (back prop)
    L_3 = transpose(A_3) - YV;
    L_2 = transpose(Theta2(:, 2:end)) * L_3 .* transpose(sigmoidGradient(Z_2));
    
    %accumulate error among training samples
    Theta2_grad = Theta2_grad + (L_3 * A_2);
    Theta1_grad = Theta1_grad + (L_2 * A_1);
    
    %copute vectorized cost
    cost = cost + ((log(A_3) * -YV) - (log(1 - A_3) * (1 - YV)));
end

cost = (cost / m);

%compute cost regularization as summed squares of each weight matrix
regularization  =(lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .*  Theta1(:, 2:end))) + sum(sum(Theta2(:, 2:end) .* Theta2(:, 2:end))));
J = cost + regularization;

% Compute gradient regularization 
Theta2_grad = (Theta2_grad ./ m);
Theta2_reg = (lambda / m) * [zeros(size(Theta2,1),1) Theta2(:, 2:end)];
Theta2_grad = Theta2_grad + Theta2_reg;

Theta1_grad = (Theta1_grad ./ m);
Theta1_reg = (lambda / m) * [zeros(size(Theta1,1),1) Theta1(:, 2:end)];
Theta1_grad = Theta1_grad + Theta1_reg;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
