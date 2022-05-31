library(ggplot2)
library(dplyr)

df <- read.csv("compression_dataset.csv")
df$compressed_size <- df$compressed_size * 100 / df$data_size
df$percentage_of_ones <- df$percentage_of_ones
test_data_size = 1500

# df <- df %>% 
#     mutate(compressed_size = rowMeans(cbind(lag(compressed_size), compressed_size, lead(compressed_size)), na.rm = T))
# df <- df %>% 
#     mutate(compressed_size = rowMeans(cbind(lag(compressed_size), compressed_size, lead(compressed_size)), na.rm = T))
# df <- df %>% 
#     mutate(compressed_size = rowMeans(cbind(lag(compressed_size), compressed_size, lead(compressed_size)), na.rm = T))
# df <- df %>% 
#     mutate(compressed_size = rowMeans(cbind(lag(compressed_size), compressed_size, lead(compressed_size)), na.rm = T))
# df <- df %>% 
#     mutate(compressed_size = pmax(compressed_size, lead(compressed_size), lead(compressed_size, n = 2), na.rm = T))
# 
# df <- df %>% 
#     mutate(compressed_size = pmax(compressed_size, lead(compressed_size), lead(compressed_size, n = 2), na.rm = T))
# 
# df <- df %>% 
#     mutate(compressed_size = pmax(compressed_size, lead(compressed_size), lead(compressed_size, n = 2), na.rm = T))
# 
# df <- df %>% 
#     mutate(compressed_size = pmax(compressed_size, lead(compressed_size), lead(compressed_size, n = 2), na.rm = T))
# 
# df <- df %>% 
#     mutate(compressed_size = pmax(compressed_size, lead(compressed_size), lead(compressed_size, n = 2), na.rm = T))
# 
# df <- df %>% 
#     mutate(compressed_size = pmax(compressed_size, lead(compressed_size), lead(compressed_size, n = 2), na.rm = T))
# 
# df <- df %>% 
#     mutate(compressed_size = pmax(compressed_size, lead(compressed_size), lead(compressed_size, n = 2), na.rm = T))


df1 <- df %>%
    filter(data_size == 100000)

df2 <- df %>%
    filter(percentage_of_ones == 0.03)

x <- seq(0, 0.5, length.out = 10000)
y <- 2.4274572 * x^0.3849364 * test_data_size^-0.0395816 + 180.07035/test_data_size
#y_global <- (2.4274572 * x^0.3849364 - 0.1690710) * test_data_size^-0.0395816
y_global <- 2.318468 * x^0.405595 * test_data_size^-0.038066 + 226.678704/test_data_size

gg1 <- ggplot(df1, aes(x = percentage_of_ones, y = compressed_size)) +
    geom_point(alpha = 0.3) +
    # geom_line(data = data.frame(x, y), aes(x = x, y = y), color='red') +
    # geom_line(data = data.frame(x, y_global), aes(x = x, y = y_global), color='blue') +
    geom_smooth(method = "nls", method.args = list(start=c(a = 2, b = 0.2, c = 0)),
               formula = y~a*x^b + c, se = FALSE, size = 0.8)

gg2 <- ggplot(df2, aes(x = data_size, y = compressed_size)) +
    geom_point(alpha = 0.3) +
    # geom_line() +
    geom_smooth(method = "nls", method.args = list(start=c(a = 0.4, b = 0)),
                formula = y~a/x + b, se = FALSE, size = 1) + 
    geom_hline(yintercept = 100) +
    scale_x_continuous(breaks = round(seq(0, max(df2$data_size), by = 20000),1)) +
    scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) +
    labs(x ="Data size", y = "Compression rate (% over data size)") +
    theme_bw()

print(gg2)
ggsave(gg2, file="compression_model.png", width=12, height=8, dpi = 300)

df <- df %>% filter(data_size >= 180) %>% filter(data_size <= 1500) %>% filter(percentage_of_ones <= 0.16)
model <- nls(formula = compressed_size ~ a * percentage_of_ones^b * data_size^c + d/data_size, data = df, start=list(a = 8, b = .3, c = -.2, d = 100))
print(summary(model))

#print(1.628923*df$percentage_of_ones^0.348981 - 0.210933)
