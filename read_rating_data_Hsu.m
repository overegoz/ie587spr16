function [tY, tR, rating_num] = read_rating_data_Hsu(rating_data_path,...
    delimiter, n_movies, n_users)
    fileID = fopen(rating_data_path);
    %disp(rating_data_path);
    C = textscan(fileID, '%d %d %f %d', 'Delimiter', delimiter,...
        'MultipleDelimsAsOne', 1);
    %A = dlmread(rating_data_path, delimiter);
    %disp(C{1});
    iter_max = length(C{1}); %disp(iter_max) 
    tY = zeros(n_movies, n_users);
    tR = zeros(n_movies, n_users);
    rating_num = zeros(1, n_users);
%     movieList = read_mov_title_Hsu(dataset_path{j} + movie_name{j},...
%     mov_delimit{j}, n_movies(j));
%     %rating_start = zeros(1, n_users(j));
    for i=1:1:iter_max
        user = C{1}(i);
        movie = C{2}(i); 
        rate = C{3}(i);
        time = C{4}(i); % we don't care this data
        tY(movie, user) = rate;
        tR(movie, user) = 1;
        rating_num(1, user) = rating_num(1, user) + 1;
    end
    fclose(fileID);