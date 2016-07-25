# This script utilizes the TCGA2STAT package for convenient access to the TCGA 
# portal for dataset initiation.

#package import
library(TCGA2STAT)
library(feather)

# Download datasets into a list called 'PRAD'
PRAD <- getTCGA(disease = "PRAD", 
                data.type = "RNASeq2",
                p = getOption("mc.cores", 2L), 
                clinical = T)

#wrangle into data_frame
library(dplyr)
gc <- SampleSplit((PRAD[[1]]))[[1]] %>% t() %>% data.frame(stringsAsFactors = FALSE) #Split gene count data into tumor [[1]] versus no tumor [[2]], transpose column and rows and create data frame
gc <- cbind("gc_index" = row.names(gc), gc) #as row names get dropped in feather, create a temporary column with TCGA ID codes
gc$gc_index <- sapply(gc$gc_index, substr, 0, 12) #shorten extended ID to the standard 12 characters observed in clininical data set


clinical <- data.frame(PRAD[[2]], stringsAsFactors = FALSE)
clinical <- cbind("clinical_index" = row.names(clinical), clinical) #move row names to first column

#write files to be picked up in python
write_feather(gc, "feather_files/Gene_counts.feather") 
write_feather(clinical, "feather_files/Clinical_data.feather")
