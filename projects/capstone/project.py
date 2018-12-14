# General imports
import pandas as pd

################################
##### Reading the datasets #####
################################

# Reading the CSV that contains inspection data

# Reading the RDS that contains census data
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()

readRDS = robjects.r['readRDS']
df = readRDS('my_file.rds')
df = pandas2ri.ri2py(df)
# do something with the dataframe

############################
##### Data Exploration #####
############################

# Getting distribution of rescues per inspection

# Exploring the dimensionality of census data

############################
##### Label definition #####
############################

############################
##### Joining datasets #####
############################

####################################
##### Dimensionality reduction #####
####################################

######################################
##### Model training and testing #####
######################################

##########################
##### Model analysis #####
##########################

#########################################
##### Labeling other municipalities #####
#########################################

######################################
##### Joining all municipalities #####
######################################

##########################
##### Showing on map #####
##########################