function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
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

% -----------------------------------------------------------
% 2.2.1 collaborative filtering cost function
% (a) un-vectorized
% tempJ = 0;
% for i=1:1:num_movies
%     for j=1:1:num_users
%         temp = Theta(j,:) * X(i,:)' - Y(i,j);
%         % by multiplying R(i,j), we only add those movies rated
%         tempJ = tempJ + R(i,j) * (temp*temp);
%     end
% end
% J = 0.5 * tempJ;
% (b) vectorized
temp = X * Theta' - Y;
temp = temp .* temp;
temp = temp .* R;
J = 0.5 * sum(sum(temp));
% -----------------------------------------------------------
% 2.2.3 regularized cost function
J = J ...
    + (lambda/2) * sum(sum(Theta .* Theta)) ...
    + (lambda/2) * sum(sum(X .* X));

% -----------------------------------------------------------
% 2.2.2 collaborative filtering gradient
% X_grad = zeros(size(X)); <==> num_movie * num_features
% Theta_grad = zeros(size(Theta)); <==> num_uses * num_features
%
% (a.1) un-vectorized: X_grad
% for i=1:1:num_movies
%     for k=1:1:num_features
%         temp = 0;
%         for j=1:1:num_users
%             temp = temp ...
%                    + R(i,j) * (X(i,:) * Theta(j,:)' - Y(i,j)) * Theta(j,k);
%         end
%         X_grad(i,k) = temp;
%     end
% end
%(a.2) vectorized: X_grad
for i=1:1:num_movies
    idx = find(R(i,:) == 1);
    Theta_temp = Theta(idx,:);
    Y_temp = Y(i,idx);
    X_grad(i,:) = (X(i,:)*Theta_temp' - Y_temp)*Theta_temp;
    % -------------------------------------------------------
    % 2.2.4 regularized gradient
    X_grad(i,:) = X_grad(i,:) + lambda * X(i,:);
end



% -----------------------------------------------------------
% 2.2.2 collaborative filtering gradient
% (b.1) un-vectorized: Theta_grad
% for j=1:1:num_users
%     for k=1:1:num_features
%         temp = 0;
%         for i=1:1:num_movies
%             temp = temp ...
%                    + R(i,j) * (X(i,:) * Theta(j,:)' - Y(i,j)) * X(i,k);
%         end
%         Theta_grad(j,k) = temp;
%     end
% end
% (b.2) vectorized: Theta_grad
for j=1:1:num_users
    idx = find(R(:,j) == 1);
    X_temp = X(idx,:);
    Y_temp = Y(idx,j);
    Theta_grad(j,:) = (Theta(j,:)*X_temp' - Y_temp')*X_temp;
    % -------------------------------------------------------
    % 2.2.4 regularized gradient
    Theta_grad(j,:) = Theta_grad(j,:) + lambda * Theta(j,:);
end



% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
