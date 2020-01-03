#MovieLens Project
################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#######################
# DATA PREPARATION AND EXPLORATION
#######################
# edx data is in tidy format
edx %>% as_tibble()

nrow(edx) 
ncol(edx)

# Check for missing values
any(is.na(edx))

# Number of unique movies and users 
edx %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

# Modify edx by adding a column for year
edx_mod <- edx %>% mutate(year = as.numeric(str_sub(title, -5, -2)))
edx_mod %>% as_tibble()

# additional libraries 
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")


# distribution of movie ratings 
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Distribution of Movie Ratings") 

# distribution of user ratings
edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Distribution of User Ratings") 

# distribution of ratings per year
rating_years <- edx_mod %>%
  group_by(year) %>%
  summarize(Count = n()) %>%
  arrange(desc(Count))

rating_years %>%
  ggplot(aes(year, Count)) +
  geom_line() + 
  ggtitle("Ratings per Year") 

#Rating and Release Year
edx_mod %>% group_by(year) %>%
  summarize(mu_rt = mean(rating)) %>%
  ggplot(aes(year, mu_rt)) +
  geom_point() + 
  geom_smooth() + 
  ggtitle("Avg Rating by Release Year") 


#######################
#PREDICTION MODELS
#######################

# Create a funtion for RMSE (root mean squre errors) since it will be used multiple times
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


### MODEL 1 - NAIVE APPROACH
#Just the average rating for all movies
mu <- mean(edx$rating)
mu

# Calculate the RMSE 
naive_rmse <- RMSE(validation$rating, mu)
naive_rmse

#add results to a table
rmse_results <- data.frame(method = "Naive Approach", RMSE = naive_rmse)
rmse_results %>% knitr::kable()


### MODEL 2 - MOVIE EFFECT
# estimate movie bias b_i 
movie_avgs <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# plot movie bias
movie_avgs %>% qplot(b_i, geom = "histogram", bins = 10, data = ., color = I("black"))

# calculate the predictions 
predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

# calculate rmse after modelling movie effect
model_2_rmse <- RMSE(predicted_ratings, validation$rating)

# add result to table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie effect model",  
                                     RMSE = model_2_rmse))
rmse_results %>% knitr::kable()


### MODEL 3 - MOVIE + USER EFFECT
# estimate user bias 'b_u' for all users
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#plot user bias
user_avgs%>% qplot(b_u, geom="histogram", bins=30, data=., color=I("black"))

# calculate predictions considering user effects in previous model
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# calculate rmse after modelling user specific effect in previous model
model_3_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User effects model",  
                                     RMSE = model_3_rmse))
rmse_results %>% knitr::kable()


### MODEL 4 - REGULARIZING THE MOVIE + USER EFFECT
#labdas is a tuning parameter, we can use cross-validation to choose the penalty term
lambdas <- seq(0,10,0.25)

model_4_rmse <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating))
})

qplot(lambdas, model_4_rmse) 
#lambda that minimizes the rmse
lambda <- lambdas[which.min(model_4_rmse)]
lambda
#calculate and add record to table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User effect model",  
                                     RMSE = min(model_4_rmse)))
rmse_results %>% knitr::kable()

