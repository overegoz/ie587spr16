%% Build Y and R matrices
% Build the following two matrices
% R : (i,j) is 1 if movie-i is rated by user-j; 0 otherwise
% Y : (i,j) is [1,5] if movie-i is rated by user-j; 0 otherwise
clear; clc; close all;
enable_new_rating = 0;
current_dir = pwd;
n_movies = [1682, 10329, 3952, 10681, 27278, 34208];    %size(n_movies);
n_users = [943, 668, 6040, 71567, 138493, 247753];
dataset_path = {'\MovieLens Dataset\100K\',...
    '\MovieLens Dataset\Latest\Small\ml-latest-small\',... 
    '\MovieLens Dataset\1M\ml-1m\',...
    '\MovieLens Dataset\10M\ml-10M100K\',...
    '\MovieLens Dataset\20M\ml-20m\',...
    '\MovieLens Dataset\Latest\Full\ml-latest\'};
%dataset_path = {'/MovieLens Dataset/100K/',...
%    '/MovieLens Dataset/Latest/Small/ml-latest-small/',... 
%    '/MovieLens Dataset/1M/ml-1m/',...
%    '/MovieLens Dataset/10M/ml-10M100K/',...
%    '/MovieLens Dataset/20M/ml-20m/',...
%    '/MovieLens Dataset/Latest/Full/ml-latest/'};
dataset_name = {'u.data', 'ratings.csv', 'ratings.dat', 'ratings.dat',...
    'ratings.csv', 'ratings.csv'};
movie_name = {'u.item', 'movies.csv', 'movies.dat', 'movies.dat',...
    'movies.csv', 'movies.csv'};
dataset_size = {'100K', 'Small', '1M', '10M', '20M', 'Full'};
rating_delimit = {'\t', ',', '::', '::', ',', ','};
mov_delimit = {'|', ',', '::', '::', ',', ','};
if size(n_movies, 2) ~= size(n_users, 2) || size(n_users, 2) ~=...
        size(dataset_path, 2) || size(dataset_path, 2) ~=...
        size(mov_delimit, 2) || size(mov_delimit, 2) ~=...
        size(dataset_size, 2)
    fprintf('The numbers of data sets are different. Wish to go on?');
    pause;
end
%alg = {'FMINGCon', 'GRAD', 'ALS_FMINCG'};
alg = {'FMINGCon', 'ALS_FMINCG'};
lambda = [0, 5];
num_features = [50];
MAX_ITER_FMINCG = [20];
MAX_ITER_GRAD = [50];
MAX_ITER_ALS_OUTER = [10]; 
MAX_ITER_ALS_INNER = [2];
%label = 'dataset, algorithm, max_iter_outer, max_iter_inner, lambda, num_features, MSE, HLU';
fid = fopen('results.csv', 'a');
%fprintf(fid, 'dataset, algorithm, max_iter_outer, max_iter_inner, lambda, num_features, MSE, HLU\n');
dataset_num = 1;    % number of data sets to run
fold_num = 5;   %cross-validation fold number (1/fold_num = percentage of
                %ratings of a user for testing)
seed_num = 1234;
rList_leng = 10;
hlu_d = 3;
hlu_alpha = 5;
% for each data set, models are learned and evaluated within each
% iterations
for j = 1:dataset_num
    %d = fopen(strcat(dataset_path(j), dataset_name(j)));
    [trainY, trainR, rating_num] = read_rating_data_Hsu(strcat(...
        current_dir, dataset_path{j}, dataset_name{j}),...
        rating_delimit{j}, n_movies(j), n_users(j));
    %iter_max = size(d,1); 
    %tY = zeros(n_movies(j), n_users(j));
    %tR = zeros(n_movies(j), n_users(j));
    %rating_num = zeros(1, n_users(j));
    movieList = read_mov_title_Hsu(strcat(current_dir, dataset_path{j},...
        movie_name{j}), mov_delimit{j}, n_movies(j));
    %rating_start = zeros(1, n_users(j));
    %for i=1:1:iter_max
    %    user = d(i,1);
    %    movie = d(i,2); 
    %    rate = d(i,3);
    %    time = d(i,4); % we don't care this data
    %    tY(movie,user) = rate;
    %    tR(movie,user) = 1;
    %    rating_num(1, user) = rating_num(1, user) + 1;
    %end
    if enable_new_rating == 1
        [trainR, trainY, rating_num] = new_ratings_Hsu(trainR, trainY,...
            movieList, rating_num);
    %else
    %    fprintf('First original user ratings:\n');
    %    for i = 1:size(trainR, 1)
    %        if( trainR(i, 1) > 0 )
    %            fprintf('Rated %d : %s\n', trainY(i, 1), movieList{i});
    %        end
    %    end
   % 
   %     %fprintf('\nProgram paused. Press enter to continue.\n');
   %     %pause;
   %     %trainR = tR;
   %     %trainY = tY;
    end
    %clear tR;
    %clear tY;
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
                            for k = 1:size(trainR, 2)
                                rng(seed_num + k);
                                indices = crossvalind('Kfold',...
                                    rating_num(1, k), fold_num);
                                %fprintf('%d\n', length(indices));
                                test = (indices == i);
%                                 for l = 1:length(test)
%                                     if test(l) == 1
%                                         fprintf('%d ',l);
%                                     end
%                                 end
                                %fprintf('\n');
                                cnt = 0;
                                for l = 1:size(trainR, 1)
                  
                                    if trainR(l, k) == 1
                                        cnt = cnt + 1;
                                        %fprintf('%d\n', l);
                                        if test(cnt) == 1
                                            testY(cnt, k) =...
                                                trainY(l, k);
                                            testR(cnt, k) = l;
                                            trainR(l, k) = 0;
                                            trainY(l, k) = 0;

                                        end
                                    end
                                    if trainR(l, k) == 1 && ismember(l,...
                                            testR(:, k)) == 1
                                        fprintf('duplicate samples %d %d %d\n',...
                                            l, trainY(l, k), find(testR(:, k), l));
                                        pause;
                                    end
                                end
                                if cnt ~= rating_num(1, k)
                                    fprintf('The numbers of rating records of user %d do not match (%d and %d)!',... 
                                        k, rating_num(1, k), cnt);
                                    pause;
                                end
                            end
                            [MSE, HLU] = optimizer_Hsu(trainR, trainY,...
                                testR, testY, alg(m), lambda(n),...
                                num_features(o), max_iter_outer(p),...
                                max_iter_inner(q), i, movieList,...
                                rList_leng, hlu_d, hlu_alpha);
                            obj_val = [MSE, HLU];
                            for k = 1:size(testY, 2)
                                for l = 1:size(testY, 1)
                                    if testY(l, k) > 0
                                        trainR(testR(l, k), k) = 1;
                                        trainY(testR(l, k), k) =...
                                            testY(l, k);
                                    end
                                end
                            end
                            %if isequal(trainR(:,2:end), tR) == 0 ||...
                            %        isequal(trainY(:,2:end), tY) == 0
                            %    fprintf('matrix is not recovered.');
                            %    pause;
                            %end
                            for k = 1:size(measurement, 2)
                                measurement(1, k) =...
                                    measurement(1, k) + obj_val(k);
                            end
                        end
                        measurement(1,:) = measurement(1,:) / fold_num;
                        fprintf(fid, '%s, %s, %d, %d, %f, %d, %f, %f\n',...
                            dataset_size{j}, alg{m}, max_iter_outer(p),...
                            max_iter_inner(q), lambda(n),...
                            num_features(o), measurement(1),...
                            measurement(2));
                    end
                end
                
            end
        end
    end
    fprintf('data set %s done.\n', dataset_size{j});
end
fclose(fid);