%% Collaborative Filtering for the Movie Recommender system 
% Let's go

%% =============== Part 0: INIT ================
clear; clc; close all;

GRAD = 0;
if( GRAD == 1 )
    fprintf('Gradient-method is enabled [ENTER]\n'); pause;
end

%% =========== Part 1: Load data set ============
%  loading the movie ratings dataset to understand the
%  structure of the data.
%  
fprintf('Loading movie ratings dataset.\n');

%  Load data
load ('movies.mat');
%  Y is a 1682 x 943 matrix, containing ratings (1-5) of 1682 movies on 
%  943 users
%
%  R is a 1682 x 943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i


%% ========== Part 2: Entering ratings for a new user ===========
%  Before we will train the collaborative filtering model, we will first
%  add ratings that correspond to a new user that we just observed. This
%  part of the code will also allow you to put in your own ratings for the
%  movies in our dataset!
%
movieList = read_movie_titles();

%  Initialize my ratings
my_ratings = zeros(1682, 1);

% Check the file movie_idx.txt for id of each movie in our dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings(1) = 4;

% Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings(98) = 2;

% We have selected a few movies we liked / did not like and the ratings we
% gave are as follows:
my_ratings(7) = 3;
my_ratings(12)= 5;
my_ratings(54) = 4;
my_ratings(64)= 5;
my_ratings(66)= 3;
my_ratings(69) = 5;
my_ratings(183) = 4;
my_ratings(226) = 5;
my_ratings(355)= 5;

fprintf('New user ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% ============== Part 7: Learning Movie Ratings ================
%  Now, you will train the collaborative filtering model on a movie rating 
%  dataset of 1682 movies and 943 users
%

fprintf('\nTraining collaborative filtering...\n');

%  Load data
load('movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = meanNormalization(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 50;  % more num of features might increate the accuracy

% Set Initial Parameters (Theta, X) to small random values
X = 0.5 .* randn(num_movies, num_features);
Theta = 0.5 .* randn(num_users, num_features);
    initX = X;
    initTheta = Theta;
    
initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
MAX_ITER_FMINCG = 100;
options = optimset('GradObj', 'on', 'MaxIter', MAX_ITER_FMINCG);

% Set Regularization
lambda = 5; % weight to the regularization term, less weight might increase the accuracy
theta = fmincg (@(t)(cost_grad(t, Ynorm, R, num_users, num_movies, ...
                                  num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into X and Theta
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

fprintf('Recommender system learning completed.\n');

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;            
 

%% ============== 8. Make some recommendations ================
%  After training the model, you can now make recommendations by computing
%  the predictions matrix.
%

p = X * Theta'; % calculate the predictions
my_predictions = p(:,1) + Ymean; % take my predictions

[r, ix] = sort(my_predictions, 'descend');
    adjust = 5.0/max(my_predictions);
fprintf('\nTop recommendations for you:\n');
for i=1:20
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', ...
            my_predictions(j) * adjust, ...
            movieList{j});
end

fprintf('\n\nOriginal ratings provided vs. estimated:\n');
mse = 0; cnt = 0;
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('%.2f vs %.2f ... [idx: %04d] %s \n', my_ratings(i), ...
                 my_predictions(i), i, movieList{i} );
        mse = mse + (my_ratings(i) - my_predictions(i))^2;
        cnt = cnt + 1;
    end
end
mse = mse / cnt;
fprintf('Mean suquared error: %f\n', mse);

%% Gradient-based algorithm
if( GRAD == 1 )
gX = initX;
gTheta = initTheta;

MAX_ITER_GRAD = 100;
a = 0.000047; % alpha

for i=1:1:MAX_ITER_GRAD
    gInit = [gX; gTheta];
    [JJ ggrad] = cost_grad(gInit, Ynorm, R, num_users, num_movies, ...
                           num_features, lambda);
    fprintf('[%04d] JJ: %.0f\n', i, JJ);                   
    gradX = reshape(ggrad(1:num_movies*num_features), ...
                    num_movies, num_features);
    gX = gX ...
         + ((-1*a/i) .* gradX);
    gradTheta = reshape(ggrad(num_movies*num_features+1:end), ...
                        num_users, num_features);
    gTheta = gTheta ...
         + ((-1*a/i) .* gradTheta);
end

%  After training the model, you can now make recommendations by computing
%  the predictions matrix.
%

gp = gX * gTheta'; % calculate the predictions
gmy_predictions = gp(:,1) + Ymean; % take my predictions

[r, ix] = sort(gmy_predictions, 'descend');
    adjust = 5.0/max(gmy_predictions);
fprintf('\nTop recommendations for you:\n');
for i=1:20
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', ...
            gmy_predictions(j) * adjust, ...
            movieList{j});
end

fprintf('\n\nOriginal ratings provided vs. estimated:\n');
mse = 0; cnt = 0;
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('%.2f vs %.2f ... [idx: %04d] %s \n', my_ratings(i), ...
                 gmy_predictions(i), i, movieList{i} );
        mse = mse + (my_ratings(i) - gmy_predictions(i))^2;
    end
end
mse = mse / cnt;
fprintf('Mean suquared error: %f\n', mse);

end
% END