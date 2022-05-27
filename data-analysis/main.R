library(ggplot2)
library(reshape2)
library(jsonlite)
library(dplyr)
library(forcats)
library(RColorBrewer)
library(tools)
library(tictoc)
library(magick)


extract_from_file <- function(path_to_file) {
    data <- fromJSON(path_to_file)
    
    observations <- sum(data$images$iterations)
    algorithm_col <- character(observations)
    iteration_col <- numeric(observations)
    filename_col <- character(observations)
    date_col <- numeric(observations)
    mean_col <- numeric(observations)
    ssim_col <- numeric(observations)
    std_col <- numeric(observations)
    rate_col <- numeric(observations)
    
    current_index <- 1
    runs_number <- nrow(data$images)
    date <- as.numeric(strptime(data$date, "%Y-%m-%dT%H:%M:%S", tz="UTC"))
    for(run_index in 1:runs_number) {
        iterations <- data$images$iterations[[run_index]]
        filename <- data$images$filename[[run_index]]
        means <- data$images$mean[[run_index]]
        ssims <- data$images$ssim[[run_index]]
        stds <- data$images$std[[run_index]]
        rates <- data$images$ratio[[run_index]]
        for(i in 1:iterations) {
            algorithm_col[current_index] <- data$algorithm
            iteration_col[current_index] <- i
            filename_col[current_index] <- filename
            date_col[current_index] <- date
            mean_col[current_index] <- means[i]
            ssim_col[current_index] <- ssims[i]
            std_col[current_index] <- stds[i]
            rate_col[current_index] <- rates[i]
            
            current_index <- current_index + 1
        }
    }
    
    return(data.frame(algorithm=factor(algorithm_col), filename=factor(filename_col), 
                      date=date_col, iteration=iteration_col, mean_difference=mean_col, 
                      SSIM=ssim_col, STD=std_col, rate=rate_col))
}

read_all_in_directory <- function(directory) {
  files <- list.files(directory, pattern="*.json", full.names=TRUE)
  data <- lapply(files, extract_from_file)
  names(data) <- lapply(lapply(files, basename), file_path_sans_ext)
  return(bind_rows(data, .id = "algorithm"))
}

get_last_iteration <- function(df) {
    return(df %>%
               group_by(algorithm, filename) %>%
               slice_max(iteration, n = 1) %>%
               ungroup())
}

plot_property <- function(df, property) {
    increasing_in <- sum(unlist(tapply(df[[property]], factor(df$algorithm), diff)))
    position <- 1
    
    if(increasing_in < 0) {
        position <- 0
    }
    
    gg <- df %>%
        ggplot(aes_string("iteration", property)) +
        geom_line(aes(colour = algorithm), size = 1.2, alpha = 0.6) + 
        scale_color_brewer(palette = "Set1") + 
        xlab("Iteration") + 
        ylab(toTitleCase(gsub("[^A-Za-z0-9]", " ", property))) +
        theme(
            legend.position = c(0.1, abs(position - 0.1)),
            legend.justification = c(0.1, abs(position - 0.1)),
            legend.margin = margin(6, 6, 6, 6),
            legend.title = element_blank(),
            legend.background = element_rect(fill=alpha("white", 0.6))
            ) + 
        scale_x_continuous(breaks = scales::pretty_breaks(n = 12)) +
        scale_y_continuous(breaks = scales::pretty_breaks(n = 12))
    
    return(gg)
}

generate_all_plots <- function(df) {
    filenames <- levels(df$filename)
    properties <- colnames(df)[5:8]
    
    for(file in filenames) {
        file_df <- df %>% filter(filename == file)
        for(property in properties) {
            gg <- plot_property(file_df, property)
            ggsave(gg, file=paste0("plots/", property, "_", file_path_sans_ext(file), ".png"),
                   width=12, height=8, dpi = 300)
        }
    }
}

merge_algorithms <- function(last_iteration_df, org_name, improved_name) {
  org_df <- last_iteration_df %>% 
    filter(algorithm == org_name) %>%
    select(-c(algorithm, date))
  
  improved_df <- last_iteration_df %>%
    filter(algorithm == improved_name) %>%
    select(-c(algorithm, date))
  
  colnames(improved_df)[2:6] <- paste0(colnames(improved_df)[2:6],"_improved")
  
  return(inner_join(org_df, improved_df, by = "filename"))
}

get_algorithm_ratio <- function(last_iteration_df, org_name, improved_name) {
  merged_df <- merge_algorithms(last_iteration_df, org_name, improved_name)
  ratios_df <- merged_df %>%
    mutate(SSIM_ratio=SSIM_improved/SSIM, STD_ratio=STD_improved/STD, 
           rate_ratio=rate_improved/rate, iteration_ratio=iteration_improved/iteration, 
           mean_ratio=mean_difference_improved/mean_difference) %>%
    select(c(filename, SSIM_ratio, STD_ratio, rate_ratio, iteration_ratio, mean_ratio))
  
  return(ratios_df)
}

summerize_last_iterations <- function(last_iteration_df) {
  last_iteration_df %>%
    group_by(algorithm) %>%
    summarise(mean_difference = mean(mean_difference), SSIM = mean(SSIM), STD = mean(STD), rate = mean(rate)) %>%
    print()
}

get_pixels_of_image <- function(file_path) {
  image_file <- image_read(file_path)
  return(as.vector(image_file[[1]], mode = "integer"))
}


df <- read_all_in_directory("data/custom")

bp_df <- df %>% filter(grepl("bp", algorithm))

# tic()
# generate_all_plots(bp_df)
# toc()


##################### COMPARING ALL ALGORITHMS #####################
# last_iteration <- df %>%
#           group_by(algorithm, filename) %>%
#           filter(mean_difference < 2) %>%
#           arrange(desc(rate)) %>%
#           slice_head(n = 1) %>%
#           ungroup()
last_iteration <- get_last_iteration(df)

summerize_last_iterations(last_iteration)

last_iteration %>%
    mutate(algorithm = fct_reorder(algorithm, SSIM)) %>%
    ggplot(aes(x=algorithm, y=SSIM)) +
    geom_boxplot()


# print(tapply(last_iteration$rate, last_iteration$algorithm, summary))
# print(tapply(last_iteration$SSIM, last_iteration$algorithm, summary))
# print(tapply(last_iteration$STD, last_iteration$algorithm, summary))
# print(tapply(last_iteration$rate, last_iteration$algorithm, quantile, probs=1:10/10))

### Try to get max iteration of bp_unidirection and then only get that iteration for both algorithms
# max_iteration <- df %>%
#     group_by(algorithm, filename) %>%
#     filter(algorithm == "bp_unidirection" | algorithm == "bp_unidirection_improved") %>%
#     filter(iterations == )
#     ungroup()
ratios_df <- get_algorithm_ratio(last_iteration, "bp_unidirection", "bp_unidirection_improved")

org_pixels <- get_pixels_of_image("images/creek_greyscale/original.png")
bp_uni_pixels <- get_pixels_of_image("images/creek_greyscale/bp_unidirection.png")
bp_uni_improved_pixels <- get_pixels_of_image("images/creek_greyscale/bp_unidirection_improved.png")
bp_vb_scaling_pixels <- get_pixels_of_image("images/creek_greyscale/bp_vb_scaling.png")

plotting_df <- data.frame(bp_uni=bp_uni_pixels, bp_uni_improved=bp_uni_improved_pixels)
plotting_df <- melt(plotting_df)
colnames(plotting_df) <- c("image", "pixel_value")

ggplot(plotting_df, aes(x = pixel_value, fill = factor(image), color = factor(image), alpha = factor(image))) + 
  geom_histogram(alpha=0.3, bins=50, position="identity") +
  scale_fill_brewer(palette = "Dark2") +
  scale_color_brewer(palette = "Dark2")
  # scale_alpha_discrete(range = c(0.5, 0.3))

freq <- as.vector(table(bp_uni_improved_pixels))
