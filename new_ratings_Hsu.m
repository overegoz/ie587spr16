function [R, Y, rating_num] = new_ratings_Hsu(R, Y, movieList, rating_num)
%% Part 2.1: Entering ratings for a new user
%  add ratings for a new user 
%
%movieList = read_mov_title_Hsu(movie_path, delimit, movie_num);

%  Initialize my ratings
my_ratings = zeros(size(R, 1), 1);

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
rating_cnt = 0;
for i = 1:length(my_ratings)
    if( my_ratings(i) > 0 )
        fprintf('Rated %d : %s\n', my_ratings(i), movieList{i});
        rating_cnt = rating_cnt + 1;
    end
end

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;
rating_num = [rating_cnt rating_num];
Y = [my_ratings Y]; 
R = [(my_ratings ~= 0) R];