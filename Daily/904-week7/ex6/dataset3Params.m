function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 250;
sigma = 333;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_set = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
sigma_set = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]'; 
J_min = Inf;
J = zeros(length(C_set), length(sigma_set));

for i = 1:length(C_set)
    for j = 1:length(sigma_set)
        model = svmTrain(X, y, C_set(i), @(x1, x2) gaussianKernel(x1, x2, sigma_set(j))); 
        predict = svmPredict(model, Xval);
        error = mean(double(predict ~=yval));
        J(i, j) = error;
        if(error < J_min)
            C = C_set(i);
            sigma = sigma_set(j);
            J_min = error;
        end
    end
end


% =========================================================================

end
