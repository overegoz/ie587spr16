function [mse, hlu] = optimizer_Hsu(R, Y, testR, testY, algorithm,...
    lambda, num_features, max_iter_outer, max_iter_inner, fold_num,...
    movieList, list_len, hlu_d, hlu_alpha)
%% Collaborative Filtering for the Movie Recommender system 
% Let's go

%% Part 0: INIT & Configurations
%clear; clc; close all;


% NOTICE
% Enable (setting to 1) only one of the followings:
% 1) FMINGCon
% 2) GRAD
% 3) ALS_CVX <== do not enable this
% 4) ALS_FMINCG
if strcmp(algorithm, 'FMINGCon') == 1
    %max_iter_outer = MAX_ITER_FMINCG;
    %max_iter_inner = [0];
    FMINGCon = 1; 
    GRAD = 0;
    ALS_CVX = 0; % DO NOT SET THIS TO 1
    ALS_FMINCG = 0; % alaternating least squares with fmincg
    ALS = ALS_CVX + ALS_FMINCG;
    MAX_ITER_FMINCG = max_iter_outer;
    MAX_ITER_GRAD = 0;
    MAX_ITER_ALS_OUTER = 0; 
    MAX_ITER_ALS_INNER = 0;
elseif strcmp(algorithm, 'GRAD') == 1
    %max_iter_outer = MAX_ITER_GRAD;
    %max_iter_inner = [0];
    FMINGCon = 0; 
    GRAD = 1;
    ALS_CVX = 0; % DO NOT SET THIS TO 1
    ALS_FMINCG = 0; % alaternating least squares with fmincg
    ALS = ALS_CVX + ALS_FMINCG;
    MAX_ITER_GRAD = max_iter_outer;
    MAX_ITER_FMINCG = 0;
    MAX_ITER_ALS_OUTER = 0; 
    MAX_ITER_ALS_INNER = 0;
elseif strcmp(algorithm, 'ALS_FMINCG') == 1
    %max_iter_outer = MAX_ITER_ALS_OUTER;
    %max_iter_inner = MAX_ITER_ALS_INNER;
    FMINGCon = 0; 
    GRAD = 0;
    % Do not do this! Generates and out-of-memory error!
    ALS_CVX = 0; % DO NOT SET THIS TO 1
    ALS_FMINCG = 1; % alaternating least squares with fmincg
    ALS = ALS_CVX + ALS_FMINCG;
    MAX_ITER_ALS_OUTER = max_iter_outer; 
    MAX_ITER_ALS_INNER = max_iter_inner;
    MAX_ITER_GRAD = 0;
    MAX_ITER_FMINCG = 0;
else
    fprintf('There is nothing to do! [ENTER]\n'); 
    pause;
end






%if( FMINGCon > 0 )
%    fprintf('fmincg-method is enabled [ENTER]\n'); pause;
%elseif( GRAD > 0 )
%    fprintf('Gradient-method is enabled [ENTER]\n'); pause;
%elseif (ALS > 0)
%    fprintf('ALS (CVX)-method is enabled\n'); 
%    fprintf('Do not enable ALS_CVX [ENTER]\n'); pause;
%else
%    fprintf('There is nothing do to! [ENTER]\n'); pause;
%end

% Num of features of our interest
% more num of features might increate the accuracy
% if num_featuers is too big, cvx will generate out of MEM issue.
%num_features = 50;  

%% Part 1: Load data set
%  loading the movie ratings dataset to understand the
%  structure of the data.
%  
%fprintf('Loading movie ratings dataset.\n');

%  Load data
%load ('movies.mat');
%  Y is a 1682 x 943 matrix, containing ratings (1-5) of 1682 movies on 
%  943 users
%
%  R is a 1682 x 943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%% Part 2.1: Entering ratings for a new user
%  add ratings for a new user 
%
%movieList = read_mov_title_Hsu(movie_path, delimit, movie_num);

%  Initialize my ratings
%my_ratings = zeros(movie_num, 1);

%
% I will give high ratings to action/adventure/animations
% so as to see if the recommendations come with similar movies.
%
% Check the file movie_idx.txt for id of each movie in our dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
%my_ratings(1) = 4;
% Or suppose did not enjoy Silence of the Lambs (1991), you can set
%my_ratings(98) = 1;
% We have selected a few movies we liked / did not like and the ratings we
% gave are as follows:
%my_ratings(7)   = 2; % Twelve Monkeys (1995)
%my_ratings(12)  = 2; % Usual Suspects, The (1995)
%my_ratings(54)  = 4; % Outbreak (1995)
%my_ratings(64)  = 2; % Shawshank Redemption, The (1994)
%my_ratings(66)  = 1; % While You Were Sleeping (1995)
%my_ratings(69)  = 3; % Forrest Gump (1994)
%my_ratings(127) = 5; % Godfather, The (1972)
%my_ratings(183) = 4; % Alien (1979)
%my_ratings(226) = 5; % Die Hard 2 (1990)
%my_ratings(355) = 5; % Sphere (1998)
%my_ratings(897) = 4; % Time Tracers (1995)
%my_ratings(1304)= 5; % New York Cop (1996)
%my_ratings(1314)= 4; % Surviving the Game (1994)
%my_ratings(1646)= 4; % Men With Guns (1997)

%fprintf('New user ratings:\n');
%for i = 1:length(my_ratings)
%    if( my_ratings(i) > 0 )
%        fprintf('Rated %d : %s\n', my_ratings(i), movieList{i});
%    end
%end

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% Part 2.2 Setup

%  Y: 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
%  943 users
%  R: 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  Add our own ratings to the data matrix
%Y = [my_ratings Y]; R = [(my_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = meanNormalization(Y, R);

%  Useful Values
num_users = size(Y, 2); num_movies = size(Y, 1);

% Set Initial Parameters (Theta, X) to small random values
X = 0.5 .* randn(num_movies, num_features);
Theta = 0.5 .* randn(num_users, num_features);
initX = X; 
initTheta = Theta;

% Set Regularization: lambda
% weight to the regularization term, and
% less weight might increase the accuracy    
%lambda = 5; 


%% Part 2.3 Prep for error check

% I will hide one movie rating from each user so that I can use them
% after learning to see how accurate the prediction is.

%hidden_info = zeros(num_users, 2);
    % row : user id
    % column 1 : movie index
    % colum 2 : rating
    % eg) hidden_info(1, 1) = movie id of user 1
    %     hidden_info(1, 2) = rating of user 1 on the movie
%    MOVIE_ID = 1; RATING = 2;
%for user=1:1:num_users
    % get the indices of rated movies
%    idx = find(R(:,user) ~= 0);
    % randomly choose one of them
%    how_many = size(idx,1);
%    pick = randi([1 how_many]);
%    movie_id = idx(pick);
%    rating_given = Y(movie_id,user);
    % store the movie id and rating
%    hidden_info(user, MOVIE_ID) = movie_id;
%    hidden_info(user, RATING) = rating_given;
    % hide the corresponding info from Y and R
%    Y( pick, user ) = 0;
%    R( pick, user ) = 0;
%end


%% Part 3.1: Learning Movie Ratings (fmincg)
%  Now, you will train the collaborative filtering model on a movie rating 
%  dataset of 1682 movies and 943 users
%

fprintf('\nTraining collaborative filtering...(lambda: %f, num_features: %d, max_iter_outer: %d, max_iter_inner: %d)\n',...
    lambda, num_features, max_iter_outer, max_iter_inner);
    
if( FMINGCon == 1 )    
    initial_parameters = [X(:); Theta(:)];

    % Set options for fmincg
    options = optimset('GradObj', 'on', 'MaxIter', MAX_ITER_FMINCG);

    theta = fmincg (@(t)(cost_grad(t, Ynorm, R, num_users, num_movies, ...
                                      num_features, lambda)), ...
                    initial_parameters, options);

    % Unfold the returned theta back into X and Theta
    X = reshape(theta(1:num_movies*num_features), num_movies,...
        num_features);
    Theta = reshape(theta(num_movies*num_features+1:end), ...
                    num_users, num_features);

    fprintf('[FMINCG] Recommender system learning completed for fold %d.\n'...
        , fold_num);
    %fprintf('\nProgram paused. Press enter to continue.\n');
    %pause;            

    % -----------------------------------------------------------
    % 3.2 Make some recommendations (fmincg)
    % -----------------------------------------------------------
    %  After training the model, you can now make recommendations by 
    %  computingthe predictions matrix.
    %

    prediction_fmincg = X * Theta'; % calculate the predictions
    %my_predictions = prediction_fmincg(:,1) + Ymean; % take my predictions
    predictions = prediction_fmincg + repmat(Ymean, 1,...
        size(prediction_fmincg, 2));    
    %for k = 1:size(predictions, 1)
    %    for l = 1:size(predictions, 2)
    %        if isnan(predictions(k, l)) == 1
   %             fprintf('%d %d %f %f %f\n', k, l, prediction_fmincg(k, l), predictions(k, l), Ymean(k)); pause;
    %        end
    %    end
    %end
   %fprintf('%f', Ymean); pause;
    %hlu = 0;
    t_half_life = 0;
    t_max_hlu = 0;
    for k = 1:num_users
        max_list = zeros(num_movies, 1);
        max_list = Y(:, k);
        for l = 1:size(testY, 1)
            if testY(l, k) > 0
                max_list(testR(l, k)) = testY(l, k);
                if Y(testR(l, k), k) > 0
                    fprintf('duplicate sample\n');
                    pause;
                end
            end
        end
        [max_r, max_ix] = sort(max_list, 'descend');
        [r, ix] = sort(predictions(:, k), 'descend');
        adjust = 5.0/max(predictions(:, k));
        %fprintf('\nTop recommendations for %d user:\n', k);
        max_hlu = 0;
        half_life = 0;
        for l = 1:list_len
            j = ix(l);
            %fprintf('User %d:\n', k);
            %fprintf('Predicting rating %f for movie %s (rated: %f)\n', ...
            %        predictions(j, k) * adjust, movieList{j}, max_list(j, 1));
            %fprintf('User rating %f for movie %s\n', ...
            %        max_r(l), movieList{max_ix(l)});
            half_life = half_life + (max([max_list(j, 1)-hlu_d, 0])/...
                (2^((l-1)/(hlu_alpha-1))));
            max_hlu = max_hlu + (max([max_r(l)-hlu_d, 0])/(2^((l-1)/...
                (hlu_alpha-1))));
            %if max_hlu == 0
                %fprintf('%d %d %f %f\n', k, l, max_r(l), (2^((l-1)/(hlu_alpha-1))));
            %end
        end
        %if max_hlu == 0
        %    max_hlu = 1;
        %end
        %hlu = hlu + (half_life / max_hlu);
        t_half_life = t_half_life + half_life;
        t_max_hlu = t_max_hlu + max_hlu;
        %fprintf('%f %f %f\n', half_life, max_hlu, hlu); %pause;
    end
    %hlu = hlu / num_users;
    hlu = t_half_life / t_max_hlu;
    fprintf('Half-life utility: %f\n', hlu);
    if isnan(hlu) == 1
        pause;
    end
    %fprintf('\n\nOriginal ratings provided vs. estimated:\n');
    training_mse = 0; 
    cnt = 0;
    for j = 1:size(R, 2)
        for i = 1:size(R, 1)
            if Y(i, j) > 0 
                %fprintf('%.2f vs %.2f ... [user: %d, idx: %04d] %s \n',...
                %    Y(i, j), predictions(i, j), j, i, movieList{i});
                training_mse = training_mse + (Y(i, j) -...
                    predictions(i, j))^2;
                cnt = cnt + 1;
            end
        end
    end
    training_mse = training_mse / cnt;
    fprintf('Training mean suquared error: %f on fold %d\n',...
        training_mse, fold_num);

    % -----------------------------------------------------------
    % Part 3.3 Check the accuracy of estimation (fmincg)
    % -----------------------------------------------------------
    fprintf('\n\nMSE of rating estimation on the test set (fold %d):\n',...
        fold_num);
    mse = 0;
    cnt = 0;
    for user=1:1:num_users
        for movie = 1:size(testR, 1)
            if (testR(movie, user) > 0)
                cnt = cnt + 1;
                % get the movie index
                m_id = testR(movie, user);
                % get the estimated rating
                est = predictions(m_id, user) + Ymean(m_id);
                % get the rating given
                rat = testY(movie, user);
                % print it out
                %fprintf('USER[%04d]-MOVIE[%04d] rated: %f vs estimated: %.2f\n',...
                %    user, m_id, rat, est);
                % Mean square error
                mse = mse + (est - rat)^2;
            end
        end
    end
    mse = sqrt(mse / cnt);
    fprintf('Testing root mean suquared error: %f on fold %d\n', mse,...
        fold_num);


%end % if FMINGCon

%% 4.1 Gradient-based algorithm
elseif( GRAD == 1 )
    gX = initX;
    gTheta = initTheta;

    a = 0.000047; % alpha

    for i=1:1:MAX_ITER_GRAD
        gInit = [gX; gTheta];
        [gradCost gradients] = cost_grad(gInit, Ynorm, R, num_users,...
            num_movies, num_features, lambda);
        fprintf('[%04d] Cost: %.0f\n', i, gradCost);                   

        gradX = reshape(gradients(1:num_movies*num_features), ...
                        num_movies, num_features);
        gX = gX + ((-1 * a/i) .* gradX);

        gradTheta = reshape(gradients(num_movies*num_features+1:end), ...
                            num_users, num_features);
        gTheta = gTheta + ((-1 * a/i) .* gradTheta); 
    end
    fprintf('[GRADIENT] Recommender system learning completed on fold %d.\n',...
        fold_num);

    % -----------------------------------------------------------
    % 4.2 Recommendations (gradient)
    % -----------------------------------------------------------
    %  After training the model, you can now make recommendations by 
    %  computing the predictions matrix.
    %

    prediction_grad = gX * gTheta'; % calculate the predictions
    predictions = prediction_grad + repmat(Ymean, 1,...
        size(prediction_grad, 2)); % take my predictions
    t_max_hlu = 0;
    t_half_life = 0;
    %hlu = 0;
    for k = 1:num_users
        max_list = zeros(num_movies, 1);
        max_list = Y(:, k);
        for i = 1:size(testY, 1)
            if testY(i, k) > 0
                max_list(testR(i, k)) = testY(i,k);
            end
        end
        [max_r, max_ix] = sort(max_list, 'descend');
        [r, ix] = sort(predictions(:, k), 'descend');
        adjust = 5.0/max(predictions(:, k));
        %fprintf('\nTop recommendations for %d user:\n', k);
        max_hlu = 0;
        half_life = 0;
        for i = 1:list_len
            j = ix(i);
            %fprintf('User %d:\n', k);
            %fprintf('Predicting rating %.1f for movie %s\n', ...
            %        predictions(j, k) * adjust, movieList{j});
            max_hlu = max_hlu + (max([max_r(i)-hlu_d, 0])/2^((i-1)/...
                (hlu_alpha-1)));
            half_life = half_life + (max([max_list(j, 1)-hlu_d, 0])/...
                2^((i-1)/(hlu_alpha-1)));
        end
        %hlu = hlu + (half_life / max_hlu);
        t_half_life = t_half_life + half_life;
        t_max_hlu = t_max_hlu + max_hlu;
    end
    %hlu = hlu / num_users;
    hlu = t_half_life / t_max_hlu;
    fprintf('Half-life utility: %f\n', hlu);
    if isnan(hlu) == 1
        pause;
    end
    fprintf('\n\nOriginal ratings provided vs. estimated:\n');
    training_mse = 0; 
    cnt = 0;
    for j = 1:size(R, 2)
        for i = 1:size(R, 1)
            if Y(i, j) > 0 
                %fprintf('%.2f vs %.2f ... [user: %d, idx: %04d] %s \n',...
                %    Y(i, j), predictions(i, j), j, i, movieList{i});
                training_mse = training_mse + (Y(i, j) -...
                    predictions(i, j))^2;
                cnt = cnt + 1;
            end
        end
    end
    training_mse = training_mse / cnt;
    fprintf('Training mean suquared error: %f on fold %d\n',...
        training_mse, fold_num);


    % -----------------------------------------------------------
    % 4.3 Check the accuracy of estimation (grad)
    % -----------------------------------------------------------
    fprintf('\n\nMSE of rating estimation of the test set (fold %d):\n',...
        fold_num);
    mse = 0;
    cnt = 0;
    for user=1:1:num_users
        for movie = 1:size(testR, 1)
            if testR(movie, user) > 0
                cnt = cnt + 1;
                % get the movie index
                m_id = testR(movie, user);
                % get the estimated rating
                est = predictions(m_id, user) + Ymean(m_id);
                % get the rating given
                rat = testY(movie, user);
                % print it out
                %fprintf('USER[%04d]-MOVIE[%04d] rated: %f vs estimated: %.2f\n',...
                %    user, m_id, rat, est);
                % Mean square error
                mse = mse + (est - rat)^2;
            end
        end
    end
    mse = sqrt(mse / cnt);
    fprintf('Testing root mean suquared error: %f on fold %d\n', mse,...
        fold_num);


%end % if GRAD == 1


%% 5.1 Alternating Least Squares (ALS)

% in ALS, we do not count the regularization terms
% check the lecture slide
% in fact, this works better than the one with the regularization term
%lambda_als = 0;

elseif( ALS > 0 )    
    alsX = initX;
    alsTheta = initTheta;
    if( ALS_CVX == 1 )
        % nothing to do
        fprintf('You made a wrong choice [ENTER]\n'); pause;
    elseif( ALS_FMINCG == 1 )
        option1 = optimset('GradObj', 'on', 'MaxIter', MAX_ITER_ALS_INNER);
        for iter=1:1:MAX_ITER_ALS_OUTER
            %
            % fix Theta: X is the only decision variable here
            X_init = alsX;
            alsX = fmincg (@(t)(cost_gradX(t, alsTheta(:), ...
                                         Ynorm, R, ...
                                         num_users, num_movies, ...
                                         num_features, lambda)), ...
                              X_init(:), option1);
            alsX = reshape(alsX(1:num_movies*num_features), ...
                           num_movies, num_features);
            %
            % fix X: Theta is the only decision variable here
            Theta_init = alsTheta;
            alsTheta = fmincg (@(t)(cost_gradTheta(t, alsX(:), ...
                                             Ynorm, R, ...
                                             num_users, num_movies, ...
                                             num_features, lambda)), ...
                                  Theta_init(:), option1);
            alsTheta = reshape(alsTheta(1:num_users*num_features), ...
                               num_users, num_features);
        end
    end

    fprintf('[ALS] Recommender system learning completed on fold %d.\n',...
        fold_num);

    % -----------------------------------------------------------
    % 5.2 Recommendations (ALS)
    % -----------------------------------------------------------
    %  After training the model, you can now make recommendations by 
    %  computing the predictions matrix.
    %
    prediction_alsfmincg = alsX * alsTheta'; % calculate the predictions
    predictions = prediction_alsfmincg + repmat(Ymean, 1,...
        size(prediction_alsfmincg, 2)); 
    % take my predictions
    t_max_hlu = 0;
    t_half_life = 0;
    %hlu = 0;
    for k = 1:num_users
        max_list = zeros(num_movies, 1);
        max_list = Y(:, k);
        for i = 1:size(testY, 1)
            if testY(i, k) > 0
                max_list(testR(i, k)) = testY(i, k);
            end
        end
        [max_r, max_ix] = sort(max_list, 'descend');
        [r, ix] = sort(predictions(:, k), 'descend');
        adjust = 5.0/max(predictions(:, k));
        %fprintf('\nTop recommendations for %d user:\n', k);
        max_hlu = 0;
        half_life = 0;
        for i = 1:list_len
            j = ix(i);
            %fprintf('User %d:\n', k);
            %fprintf('Predicting rating %.1f for movie %s\n', ...
            %        predictions(j, k) * adjust, movieList{j});
            max_hlu = max_hlu + (max([max_r(i)-hlu_d, 0])/2^((i-1)/...
                (hlu_alpha-1)));
            half_life = half_life + (max([max_list(j, 1)-hlu_d, 0])/...
                2^((i-1)/(hlu_alpha-1)));
        end
        t_half_life = t_half_life + half_life; 
        t_max_hlu = t_max_hlu + max_hlu;
        %hlu = hlu + (half_life / max_hlu);
    end
    %hlu = hlu / num_users;
    hlu = t_half_life / t_max_hlu;
    fprintf('Half-life utility: %f\n', hlu);
    if isnan(hlu) == 1
        pause;
    end
    %fprintf('\n\nOriginal ratings provided vs. estimated:\n');
    training_mse = 0; 
    cnt = 0;
    for j = 1:size(R, 2)
        for i = 1:size(R, 1)
            if Y(i, j) > 0 
                %fprintf('%.2f vs %.2f ... [user: %d, idx: %04d] %s \n',...
                %    Y(i, j), predictions(i, j), j, i, movieList{i});
                training_mse = training_mse + (Y(i, j) -...
                    predictions(i, j))^2;
                cnt = cnt + 1;
            end
        end
    end
    training_mse = training_mse / cnt;
    fprintf('Training mean suquared error: %f on fold %d\n',...
        training_mse, fold_num);

    % -----------------------------------------------------------
    % Part 5.3 Check the accuracy of estimation (fmincg)
    % -----------------------------------------------------------
    fprintf('\n\nMSE of rating estimation of the test set (fold %d):\n',...
        fold_num);
    mse = 0;
    cnt = 0;
    for user=1:1:num_users
        for movie = 1:size(testR, 1)
            if testR(movie, user) > 0
                cnt = cnt + 1;
                % get the movie index
                m_id = testR(movie, user);
                % get the estimated rating
                est = predictions(m_id, user) + Ymean(m_id);
                % get the rating given
                rat = testY(movie, user);
                % print it out
                %fprintf('USER[%04d]-MOVIE[%04d] rated: %f vs estimated: %.2f\n',...
                %    user, m_id, rat, est);
                % Mean square error
                mse = mse + (est - rat)^2;
            end
        end
    end
    mse = sqrt(mse / cnt);
    fprintf('Testing root mean suquared error: %f on fold %d\n',...
        mse, fold_num);
else
    fprintf('Parameter setting abnormal, no algorithm selected.');
    pause;
end % if ALS

%% Concluding remarks...
% good job
% END