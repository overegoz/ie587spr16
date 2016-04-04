%% Build Y and R matrices
% Build the following two matrices
% R : (i,j) is 1 if movie-i is rated by user-j; 0 otherwise
% Y : (i,j) is [1,5] if movie-i is rated by user-j; 0 otherwise
clear; clc;

d = load('raw_data.txt');
n_movies = 1682;
n_users = 943;
iter_max = size(d,1); % it should be 100,000
tY = zeros(n_movies, n_users);
tR = zeros(n_movies, n_users);
for i=1:1:iter_max
    user = d(i,1);
    movie = d(i,2);
    rate = d(i,3);
    time = d(i,4); % we don't care this data
    tY(movie,user) = rate;
    tR(movie,user) = 1;
end

R = tR;
Y = tY;

save movies.mat R Y;