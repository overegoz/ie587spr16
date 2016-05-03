%% Build Y and R matrices
% Build the following two matrices
% R : (i,j) is 1 if movie-i is rated by user-j; 0 otherwise
% Y : (i,j) is [1,5] if movie-i is rated by user-j; 0 otherwise
clear; clc;
enable_new_rating = 1;
n_movies = [1682, 10329, 3952, 10681, 27278, 34208];    %size(n_movies);
n_users = [943, 668, 6040, 71567, 138493, 247753];
dataset_path = ['..\MovieLens Dataset\100K\',...
    '..\MovieLens Dataset\Latest\Small\ml-latest-small\',... 
    '..\MovieLens Dataset\1M\ml-1m\',...
    '..\MovieLens Dataset\10M\ml-10M100K\',...
    '..\MovieLens Dataset\20M\ml-20m\',...
    '..\MovieLens Dataset\Latest\Full\ml-latest\'];
dataset_name = ['u.dat', 'ratings.csv', 'ratings.dat', 'ratings.dat',...
    'ratings.csv', 'ratings.csv'];
movie_name = ['u.item', 'movies.csv', 'movies.dat', 'movies.dat',...
    'movies.csv', 'movies.csv'];
dataset_size = ['100K', 'Small', '1M', '10M', '20M', 'Full'];
mov_delimit = ['|', ',', '::', '::', ',', ','];
if size(n_movies, 2) ~= size(n_users, 2) ~= size(dataset_path, 2)...
        ~= size(mov_delimit, 2) ~= size(dataset_size, 2)
    fprintf('The numbers of data sets are different. Wish to go on?');
    pause;
end
alg = ['FMINGCon', 'GRAD', 'ALS_FMINCG'];
lambda = [0, 5];
num_features = [50];
MAX_ITER_FMINCG = [20];
MAX_ITER_GRAD = [50];
MAX_ITER_ALS_OUTER = [10]; 
MAX_ITER_ALS_INNER = [2];
movieList = read_mov_title_Hsu(movie_path, delimit, movie_num);
fid = fopen('results.csv', 'w');
fprintf(fid, 'dataset, algorithm, max_iter_outer, max_iter_inner,' + ...
    'lambda, num_features, MSE, HLU\n');
dataset_num = 1;    % number of data sets to run
fold_num = 5;   %cross-validation fold number (1/fold_num = percentage of
                %ratings of a user for testing)
seed_num = 1234;
% for each data set, models are learned and evaluated within each
% iterations
for j = 1:dataset_num
    d = load(datasset_path(j) + dataset_name(j));
    iter_max = size(d,1); 
    tY = zeros(n_movies(j), n_users(j));
    tR = zeros(n_movies(j), n_users(j));
    rating_num = zeros(1, n_users(j));
    %rating_start = zeros(1, n_users(j));
    for i=1:1:iter_max
        user = d(i,1);
        movie = d(i,2); 
        rate = d(i,3);
        time = d(i,4); % we don't care this data
        tY(movie,user) = rate;
        tR(movie,user) = 1;
        rating_num(1, user) = rating_num(1, user) + 1;
    end
    if enable_new_rating == 1
        [trainR, trainY] = new_ratings(tR, tY, movieList);
    else
        fprintf('First original user ratings:\n');
        for i = 1:size(R, 1)
            if( tR(i, 1) > 0 )
                fprintf('Rated %d : %s\n', tY(1, i), movieList{i});
            end
        end
    
        fprintf('\nProgram paused. Press enter to continue.\n');
        pause;
        trainR = tR;
        trainY = tY;
    end
    for m = 1:size(alg, 2)
        for n = 1:size(lambda, 2)
            for o = 1:size(num_features, 2)
                if strcmp(alg(m), 'FMINGCon') == 1
                    max_iter_outer = MAX_ITER_FMINCG;
                    max_iter_inner = [0];
                elseif strcmp(alg(m), 'GRAD') == 1
                    max_iter_outer = MAX_ITER_GRAD;
                    max_iter_inner = [0];
                elseif strcmp(alg(m), 'ALS_FMINCG') == 1
                    max_iter_outer = MAX_ITER_ALS_OUTER;
                    max_iter_inner = MAX_ITER_ALS_INNER;
                else
                    fprintf('There is nothing to do! [ENTER]\n'); 
                    pause;
                end
                for p = 1:size(max_iter_outer, 2)
                    for q = 1:size(max_iter_inner, 2)
                        measurement = zeros(1, 2);
                        for i = 1:fold_num
                            testR = zeros(max(rating_num(1, :)),...
                                size(trainR, 2));
                            testY = zeros(max(rating_num(1, :)),...
                                size(trainY, 2));
                            for k = 1:size(R, 2)
                                rng(seed_num + k);
                                indices = crossvalind('Kfold',...
                                    rating_num(1, k), fold_num);
                                test = (indices == i);
                                cnt = 0;
                                for l = 1:size(trainR, 1)
                                    if trainR(l, k) == 1
                                        cnt = cnt + 1;
                                        if test(cnt) == 1
                                            testY(cnt, k) =...
                                                trainY(l, k);
                                            testR(cnt, k) = l;
                                            trainR(l, k) = 0;
                                            trainY(l, k) = 0;

                                        end
                                    end

                                end
                                if cnt ~= rating_num(1, k)
                                    fprintf('The numbers of rating'...
                                        + 'records of user %d do' +...
                                        'not match (%d and %d)!',... 
                                        k, rating_num(1, k), cnt);
                                    pause;
                                end
                            end
                            [MSE, HLU] = optimizer(trainR, trainY,...
                                testR, testY, alg(m), lambda(n),...
                                num_feature(o), max_iter_outer(p),...
                                max_iter_inner(q), i, movieList);
                            obj_val = [MSE, HLU];

                            for k = 1:size(measurement, 2)
                                measurement(1, k) =...
                                    measurement(1, k) + obj_val(k);
                            end
                        end
                        measurement(1,:) = measurement(1,:) / fold_num;
                        fprintf(fid, '%s, %s, %d, %d, %f, %d, %f,'...
                            + '%f\n', dataset_size(j), alg(m),...
                            max_iter_outer(p), max_iter_inner(q),...
                            lambda(n), num_features(o),...
                            measurement(1), measurement(2));
                    end
                end
                
            end
        end
    end
end
fclose(fid);