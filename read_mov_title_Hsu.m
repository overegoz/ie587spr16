function movieList = read_mov_title_Hsu(file_path, delimiter, movie_num)
%GETMOVIELIST reads the fixed movie list in movie.txt and returns a
%cell array of the words
%   movieList = GETMOVIELIST() reads the fixed movie list in movie.txt 
%   and returns a cell array of the words in movieList.


%% Read the fixed movieulary list
fid = fopen(file_path);

% Store all movies in cell array movie{}
n = movie_num;  % Total number of movies 

movieList = cell(n, 1);
for i = 1:n
    % Read line
    line = fgets(fid);
    % Word Index (can ignore since it will be = i)
    [idx, movieName] = strtok(line, delimiter);
    % Actual Word
    movieList{i} = strtrim(movieName);
end
fclose(fid);

end
