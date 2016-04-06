function [J, gradTheta] = cost_gradTheta(paramTheta, paramX, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(paramX(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(paramTheta(1:num_users*num_features), num_users, num_features);
%
J = 0;
%X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% =============================================================
% Cost function: evaluation
temp = X * Theta' - Y;
temp = temp .* temp;
temp = temp .* R;
J = 0.5 * sum(sum(temp));
% -----------------------------------------------------------
% no regularization term on the first feature (intercept)
J = J ...
    + (lambda/2) * sum(sum(Theta(:,2:end) .* Theta(:,2:end))) ...
    + (lambda/2) * sum(sum(X(:,2:end) .* X(:,2:end)));
%
%
%
% =============================================================
% X: gradient
% for i=1:1:num_movies
%     idx = find(R(i,:) == 1);
%     Theta_temp = Theta(idx,:);
%     Y_temp = Y(i,idx);
%     X_grad(i,:) = (X(i,:)*Theta_temp' - Y_temp)*Theta_temp;
%     
%     % -------------------------------------------------------
%     if( i == 1 ) % for the intercept term
%         X_grad(i,:) = X_grad(i,:);
%     else
%         X_grad(i,:) = X_grad(i,:) + lambda * X(i,:);
%     end
% end
% =============================================================
%
%
%
% =============================================================
% Theta : gradient
for j=1:1:num_users
    idx = find(R(:,j) == 1);
    X_temp = X(idx,:);
    Y_temp = Y(idx,j);
    Theta_grad(j,:) = (Theta(j,:)*X_temp' - Y_temp')*X_temp;

    % -------------------------------------------------------
    if( j == 1 ) % for the intercept term
        Theta_grad(j,:) = Theta_grad(j,:);
    else
        Theta_grad(j,:) = Theta_grad(j,:) + lambda * Theta(j,:);
    end
end
% =============================================================
%
gradTheta = Theta_grad(:);
%
end
