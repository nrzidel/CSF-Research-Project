# Load patient data and log transformed sample data into memory
library(readxl)
library(ggplot2)
library(tidyverse)
library(MASS)

setwd("C:/Users/nicho/.vscode/Projects/Research/CSF-Research-Project")

data <- read_xlsx(path = "Data/FORD-0101-21ML+ DATA TABLES_CSF (METADATA UPDATE).XLSX", sheet = "Log Transformed Data")
patient.data <- read_xlsx(path = "Data/FORD-0101-21ML+ DATA TABLES_CSF (METADATA UPDATE).XLSX", sheet = "Sample Meta Data")



# this function is used for filtering and cleaning the data. It is also useful for 
# resetting the data if need be.
# future plans to use a parameter to select BL or V06 from PPMI_CLINICAL_EVENT
getdata <- function(input.data) {
  # Filter for PPMI samples only
  PPMI <- input.data$COHORT == "PPMI"
  
  # Filtering for data from samples with class (Etiher PD or Control)
  # Training data is class-ful, so it can be visualized in models
  PPMI.training <- (input.data$PPMI_COHORT == "PD" | input.data$PPMI_COHORT == "Control") & input.data$PPMI_CLINICAL_EVENT == "BL"
  PPMI.training.data <<- filter(data, PPMI.training)
  PPMI.training.data$PPMI_COHORT <<- as.factor(input.data$PPMI_COHORT[PPMI.training]) #Bind the class value to the training data and convert to a factor

  #Data Cleaning (Remove columns where 100% of the data is the same value)
  PPMI.training.data <<- PPMI.training.data %>%
    select(where(Negate(is.numeric))) %>% # Add non-numeric columns to front
    bind_cols(
      PPMI.training.data %>%
        select(where(is.numeric)) %>%
        select(where(~ max(table(.)) / length(.) < 1))          
    )
}




#Print out two box and whisker plots for each attribute. One for PD and the other for Control
whiskers <- function(input.data) {
  for(i in 3:length(input.data)) {
    data <- data.frame(
      Class = input.data$PPMI_COHORT,
      Value = input.data[[i]]
    )
    
    # Create the boxplot
    print(ggplot(data, aes(x = Class, y = Value, fill = Class)) +
            geom_boxplot() +
            theme_minimal() +
            labs(title = colnames(input.data[i]), x = "Class", y = "Value") +
            theme(legend.position = "none"))  # Hides legend if not needed
  }
}

print(ggplot(PPMI.training.data, mapping = aes(x=`35`, y=(`35` ^ 2), color = PPMI_COHORT)) +
  geom_count() +
  scale_size_area())

#Print out two box and whisker plots for each attribute. One for PD and the other for Control
polyKernal <- function(input.data, degree) {
  for(i in 3:50) {
    data <- data.frame(
      Class = input.data$PPMI_COHORT,
      Value = input.data[[i]]
    )
    
    # Create the boxplot
    print(ggplot(data, aes(x = Value, y = (Value ^ degree), color = Class, alpha = .5)) +
            geom_count() +
            scale_size_continuous() +
            labs(title = colnames(input.data[i]), x = "Value", y = degree))
  }
}


# Function calculates the PCA results from input data; 
# outputs the Scree, PCA scatter, and Cumulative Variance explained plots

do_pca <- function(input) {
  #PCA 
  pca_results <<- prcomp(input[3:(length(input))], retx = TRUE, center = TRUE)
  
  pca_df <- data.frame(
    Class = input[, 2],    # Class labels
    PC1 = pca_results$x[, 1],    # First Principal Component
    PC2 = pca_results$x[, 2]     # Second Principal Component
  )
  
  
  # Scree plot --------------------------------------------
  pca_var <- pca_results$sdev^2
  pca_var_per <- round(pca_var/sum(pca_var)*100, 1)
  barplot(pca_var_per, main = "Scree plot", xlab = "Principle Component", ylab = "Percent Variation")
  
  
  # Create the PCA scatter plot ---------------------------
  print(ggplot(pca_df, aes(x = PC1, y = PC2, color = PPMI_COHORT)) +
    geom_point(size = 3, alpha = 0.8) + 
    labs(title = "PCA Plot", 
         x = paste("PC1 ", pca_var_per[1], "%"), 
         y = paste("PC2 ", pca_var_per[2], "%")))
  
  
  # Get the proportion of variance explained by each principal component
  var_explained <- pca_results$sdev^2 / sum(pca_results$sdev^2)
  
  # Compute cumulative variance explained
  cumulative_var <- cumsum(var_explained)
  
  # Plot cumulative variance explained --------------------
  plot(cumulative_var, type = "b", pch = 19, col = "blue",
       xlab = "Number of Principal Components",
       ylab = "Cumulative Variance Explained",
       main = "Cumulative Variance Explained by PCA")
  
  # Add a reference line at 95% variance explained
  abline(h = 0.95, col = "red", lty = 2)
  
  # Find the number of PCs that explain at least 95% variance
  num_PCs <<- which(cumulative_var >= 0.95)[1]  # First PC reaching 95% threshold
  cat("Number of PCs needed to explain 95% variance:", num_PCs, "\n")
  
}

#Initial PCA of PPMI.training.data
do_pca(PPMI.training.data)







# Extract absolute loadings for PC1 through num_PCs
# You can think of each PC like a cocktail, which has a recipe of differing amounts
# of each attribute. Loadings are the recipe for each PC, which tells you how much
# of each attribute to put into each PC.
loadings <- abs(pca_results$rotation[, 1:num_PCs])

# Compute the mean loading, ie how much each attribute shows up in our loading recipe
# averaged accross all the selected PCs
mean_loadings <- rowMeans(loadings)

# View the attributes sorted by loading strength
sorted_loadings <- sort(mean_loadings, decreasing = TRUE)
head(sorted_loadings, 20)  # Top contributing attributes
tail(sorted_loadings, 20)  # Least contributing attributes

# Determine a dynamic threshold based on percentiles (e.g., top 90% of contributing attributes)
threshold <- quantile(mean_loadings, 0.90)  # Keep top 90% of contributing attributes

# Identify attributes to keep
important_attrs <- names(mean_loadings[mean_loadings >= threshold])

# Subset dataset to keep only important attributes
filtered_data <- PPMI.training.data[, c("PARENT_SAMPLE_NAME", "PPMI_COHORT", important_attrs)]

dim(filtered_data)

do_pca(filtered_data)




#random forest modeling

library(cowplot)
library(randomForest)



doRF <- function(input.data) {
  # Encountered a bug when the colnames were purly numeric. Added this line to fix it
  colnames(input.data) <- make.names(colnames(input.data))
  
  model <- randomForest(PPMI_COHORT ~ ., data = input.data[2:length(input.data)], proximity=TRUE)
  
  oob.error.data <- data.frame(
    Trees=rep(1:nrow(model$err.rate), times=3),
    Type=rep(c("OOB", "Control", "PD"), each=nrow(model$err.rate)),
    Error=(c(model$err.rate[,"OOB"],
             model$err.rate[, "Control"],
             model$err.rate[, "PD"]))
  )
  
  print(ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
    geom_line(aes(color=Type)))
  
  oob.values <- vector(length=length(input.data))
  for(i in 1:(length(input.data))) {
    print(i)
    temp.model <- randomForest(PPMI_COHORT ~ ., data=input.data[2:length(input.data)], mtry=i)
    oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
  }
  print(oob.values)
  print(min(oob.values))
  
}







