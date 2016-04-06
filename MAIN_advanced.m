%% Collaborative Filtering for the Movie Recommender system 
% Let's go

%% Part 0: INIT & Configurations
clear; clc; close all;

FMINGCon = 1;
MAX_ITER_FMINCG = 100;

GRAD = 0; % gradient-based descent method
MAX_ITER_GRAD = 1;

% Do not do this! Generates and out-of-memory error!
ALS_CVX = 0; % alaternating least squares with CVX
MAX_ITER_ALS = 10;

if( FMINGCon == 1 )
    fprintf('fmincg-method is enabled [ENTER]\n'); pause;
elseif( GRAD == 1 )
    fprintf('Gradient-method is enabled [ENTER]\n'); pause;
elseif (ALS_CVX == 1)
    fprintf('ALS (CVX)-method is enabled [ENTER]\n'); 
    fprintf('Do not enable this [ENTER]\n');
    pause;
end

% Num of features of our interest
% more num of features might increate the accuracy
% if num_featuers is too big, cvx will generate out of MEM issue.
num_features = 50;  

%% Part 1: Load data set
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

%% Part 2.1: Entering ratings for a new user
%  add ratings for a new user 
%
movieList = read_movie_titles();

%  Initialize my ratings
my_ratings = zeros(1682, 1);

%
% I will give high ratings to action/adventure/animations
% so as to see if the recommendations come with similar movies.
%
% Check the file movie_idx.txt for id of each movie in our dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings(1) = 4;
% Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings(98) = 1;
% We have selected a few movies we liked / did not like and the ratings we
% gave are as follows:
my_ratings(7)   = 2; % Twelve Monkeys (1995)
my_ratings(12)  = 2; % Usual Suspects, The (1995)
my_ratings(54)  = 4; % Outbreak (1995)
my_ratings(64)  = 2; % Shawshank Redemption, The (1994)
my_ratings(66)  = 1; % While You Were Sleeping (1995)
my_ratings(69)  = 3; % Forrest Gump (1994)
my_ratings(127) = 5; % Godfather, The (1972)
my_ratings(183) = 4; % Alien (1979)
my_ratings(226) = 5; % Die Hard 2 (1990)
my_ratings(355) = 5; % Sphere (1998)
my_ratings(897) = 4; % Time Tracers (1995)
my_ratings(1304)= 5; % New York Cop (1996)
my_ratings(1314)= 4; % Surviving the Game (1994)
my_ratings(1646)= 4; % Men With Guns (1997)

fprintf('New user ratings:\n');
for i = 1:length(my_ratings)
    if( my_ratings(i) > 0 )
        fprintf('Rated %d : %s\n', my_ratings(i), movieList{i});
    end
end

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% Part 2.2 Setup

%  Y: 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
%  943 users
%  R: 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  Add our own ratings to the data matrix
Y = [my_ratings Y]; R = [(my_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = meanNormalization(Y, R);

%  Useful Values
num_users = size(Y, 2); num_movies = size(Y, 1);

% Set Initial Parameters (Theta, X) to small random values
X = 0.5 .* randn(num_movies, num_features);
Theta = 0.5 .* randn(num_users, num_features);
    initX = X; initTheta = Theta;

% Set Regularization: lambda
% weight to the regularization term, and
% less weight might increase the accuracy    
lambda = 5; 


%% Part 2.3 Prep for error check

% I will hide one movie rating from each user so that I can use them
% after learning to see how accurate the prediction is.

hidden_info = zeros(num_users, 2);
    % row : user id
    % column 1 : movie index
    % colum 2 : rating
    % eg) hidden_info(1, 1) = movie id of user 1
    %     hidden_info(1, 2) = rating of user 1 on the movie
    MOVIE_ID = 1; RATING = 2;
for user=1:1:num_users
    % get the indices of rated movies
    idx = find(R(:,user) ~= 0);
    % randomly choose one of them
    how_many = size(idx,1);
    pick = randi([1 how_many]);
    movie_id = idx(pick);
    rating_given = Y(movie_id,user);
    % store the movie id and rating
    hidden_info(user, MOVIE_ID) = movie_id;
    hidden_info(user, RATING) = rating_given;
    % hide the corresponding info from Y and R
    Y( pick, user ) = 0;
    R( pick, user ) = 0;
end


%% Part 3.1: Learning Movie Ratings (fmincg)
%  Now, you will train the collaborative filtering model on a movie rating 
%  dataset of 1682 movies and 943 users
%

fprintf('\nTraining collaborative filtering...\n');
    
if( FMINGCon == 1 )    
initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', MAX_ITER_FMINCG);

theta = fmincg (@(t)(cost_grad(t, Ynorm, R, num_users, num_movies, ...
                                  num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into X and Theta
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

fprintf('[FMINCG] Recommender system learning completed.\n');
%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;            
 
% -----------------------------------------------------------
% 3.2 Make some recommendations (fmincg)
% -----------------------------------------------------------
%  After training the model, you can now make recommendations by computing
%  the predictions matrix.
%

prediction_fmincg = X * Theta'; % calculate the predictions
my_predictions = prediction_fmincg(:,1) + Ymean; % take my predictions

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

% -----------------------------------------------------------
% 3.3 Check the accuracy of estimation (fmincg)
% -----------------------------------------------------------
fprintf('\n\nMSE of rating estimation of the movies hidden:\n');
mse = 0;
for user=1:1:num_users
    % get the movie index
    m_id = hidden_info(user, MOVIE_ID);
    % get the estimated rating
    est = prediction_fmincg(m_id, user) + Ymean(m_id);
    % get the rating given
    rat = hidden_info(user, RATING);
    % print it out
    fprintf('USER[%04d]-MOVIE[%04d] rated: %d vs estimated: %.2f\n', ...
            user, m_id, rat, est);
    % Mean square error
    mse = mse + (est - rat)^2;
end
mse = mse / num_users;
fprintf('Mean suquared error: %f\n', mse);


end % if FMINGCon

%% 4.1 Gradient-based algorithm
if( GRAD == 1 )
gX = initX;
gTheta = initTheta;

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
fprintf('[GRADIENT] Recommender system learning completed.\n');

% -----------------------------------------------------------
% 4.2 Recommendations (gradient)
% -----------------------------------------------------------
%  After training the model, you can now make recommendations by computing
%  the predictions matrix.
%

prediction_grad = gX * gTheta'; % calculate the predictions
gmy_predictions = prediction_grad(:,1) + Ymean; % take my predictions

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


% -----------------------------------------------------------
% 4.3 Check the accuracy of estimation (grad)
% -----------------------------------------------------------
fprintf('\n\nMSE of rating estimation of the movies hidden:\n');
mse = 0;
for user=1:1:num_users
    % get the movie index
    m_id = hidden_info(user, MOVIE_ID);
    % get the estimated rating
    est = prediction_grad(m_id, user) + Ymean(m_id);
    % get the rating given
    rat = hidden_info(user, RATING);
    % print it out
    fprintf('USER[%04d]-MOVIE[%04d] rated: %d vs estimated: %.2f\n', ...
            user, m_id, rat, est);
    % Mean square error
    mse = mse + (est - rat)^2;
end
mse = mse / num_users;
fprintf('Mean suquared error: %f\n', mse);




end % if GRAD == 1



%% 5.1 Alternating Least Squares (ALS)
% if( ALS_CVX == 1 )    
% aX = initX;
% aTheta = initTheta;
% for i=1:1:MAX_ITER_ALS    
%     % 1. fix Theta
%     cvx_begin
%         variable tX(num_movies, num_features);
%         minimize ( ( norm(tX * aTheta' - Ynorm, 2) ) );
%     cvx_end
%     aX = tX;
%     % print out the cost
%     param = [aX; aTheta];
%     [aJ agrad] = cost_grad(param, Ynorm, R, num_users, num_movies, ...
%                            num_features, lambda);
%     fprintf('[%04d (1/2)] Cost: %.0f\n', i, aJ);  
%     
%     % 2. fix X    
%     
%     aTheta = tTheta;
%     % print out the cost
%     param = [aX; aTheta];
%     [aJ agrad] = cost_grad(param, Ynorm, R, num_users, num_movies, ...
%                            num_features, lambda);
%     fprintf('[%04d (2/2)] Cost: %.0f\n', i, aJ);      
% end % for i
% end % if ALS
% fprintf('[ALS] Recommender system learning completed.\n');

% -----------------------------------------------------------
% 5.2 Recommendations (ALS)
% -----------------------------------------------------------
% not implemented yet


%% Concluding remarks...
% good job
% END