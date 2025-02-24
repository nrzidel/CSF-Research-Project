# Load patient data and log transformed sample data into memory
library(readxl)
library(ggplot2)
library(tidyverse)

data <- read_xlsx(path = "DATA/FORD-0101-21ML+ DATA TABLES_CSF (METADATA UPDATE).XLSX", sheet = "Log Transformed Data")
patient.data <- read_xlsx(path = "DATA/FORD-0101-21ML+ DATA TABLES_CSF (METADATA UPDATE).XLSX", sheet = "Sample Meta Data")



# Filter for PPMI samples only
PPMI <- patient.data$COHORT == "PPMI"

# Filtering for data from samples with class (Etiher PD or Control)
# Training data is class-ful, so it can be visualized in models
PPMI.training <- patient.data$PPMI_COHORT == "PD" | patient.data$PPMI_COHORT == "Control"
PPMI.training.data <- filter(data, PPMI.training)
PPMI.training.data$PPMI_COHORT <- patient.data$PPMI_COHORT[PPMI.training] #Bind the class value to the training data



#Data Cleaning (Remove columns missing 75% or more of data)
PPMI.training.data <- PPMI.training.data %>%
  select(where(Negate(is.numeric))) %>% # Add non-numeric columns to front
  bind_cols(
    PPMI.training.data %>%
      select(where(is.numeric)) %>%
      select(where(~ mean(.) < .75))          
    )


#PCA 

pca_results <- prcomp(PPMI.training.data[3:507], retx = TRUE, center = TRUE)

pca_df <- data.frame(
  Class = PPMI.training.data[, 2],    # Class labels
  PC1 = pca_results$x[, 1],    # First Principal Component
  PC2 = pca_results$x[, 2]     # Second Principal Component
)

# Scree plot

pca_var <- pca_results$sdev^2
pca_var_per <- round(pca_var/sum(pca_var)*100, 1)
barplot(pca_var_per, main = "Scree plot", xlab = "Principle Component", ylab = "Percent Variation")

# Create the PCA scatter plot
ggplot(pca_df, aes(x = PC1, y = PC2, color = PPMI_COHORT)) +
  geom_point(size = 3, alpha = 0.8) + 
  labs(title = "PCA Plot", 
       x = paste("PC1 ", pca_var_per[1], "%"), 
       y = paste("PC2 ", pca_var_per[2], "%"))


