

############################
### CONFIGURE PARAMETERS ###
############################

### Set Model parameters in this first section, and then run the rest of the script

#model objective column name
objective = 'price_up_down'

# Path to Input and Output Directories
input_directory = "C:/Users/Andy/Desktop/chipotle_model/inputs"
output_directory = "C:/Users/Andy/Desktop/chipotle_model/outputs"

#Names of training and testing csvs
train_file_name = "combine_vwagy_train"
test_file_name = "combine_vwagy_test"

# A csv that contians the names for each column
# This needs to be created by hand to map the column names 
#column_name_file = "census_income_columns"

#Manually Set Categorical Features that appear as numbers
categorical_features = []
ignore_columns = ["index","day","year","date","RF",
                  'delta_price','price_lag_30','price_dif_30',
                  'price_lag_60','price_dif_60','price_lag_90',
                  'price_dif_90','price_lag_15','price_dif_15',
                  'price_lag_7','price_dif_7']

# Specify infomration gain cutoff for Weight of Evidence Binning
information_gain_cutoff = -0.35

# Feature Selection - Select the top x features based on a criterion
max_features = 4
# Optionals are ks, total_iv, spearman_r, pearson_r, split_info 
criterion = 'total_iv'
use_chi_squared_test = False
use_vif_test = True

# Below this point, no manual changes are needed

##########################
#### LOAD ENVIRONMENT ####
##########################

import pandas as pd
import numpy as np
from sklearn import *
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import openpyxl
import scipy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from catboost import CatBoostClassifier
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats as stats

pd.set_option('display.max_columns', 500)
plt.rc('font', size=8)          # controls default text sizes
plt.rc('axes', titlesize=8)     # fontsize of the axes title
plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)

####################
#### LOAD DATA  ####
####################

# Read in training data
input_file_path = input_directory + "/" + train_file_name + ".csv"
#columns_file_path = input_directory + "/" + column_name_file + ".csv"
census_raw = pd.read_csv(input_file_path)

#Read in and designate column names based on column-mapping file
#census_columns = pd.read_csv(columns_file_path)
census_raw.columns = census_raw.columns.str.replace(" ","_")

# Read and rename columns for test file
test_file_path = input_directory + "/" + test_file_name + ".csv"
census_raw_test = pd.read_csv(test_file_path)
census_raw_test.columns = census_raw_test.columns.str.replace(" ","_")

#### Make Objective Var ####
def make_objective_column(input_df,objective=objective):
      input_df[objective] = 0
      input_df[objective][input_df["price_lag_30"] - input_df["Prices"] > 0] = 1
      return input_df

def drop_columns(input_df,ignore_columns=ignore_columns):
      input_df = input_df.drop(columns=ignore_columns)
      return input_df


census_raw_test = make_objective_column(census_raw_test)
census_raw_test = drop_columns(census_raw_test)

census_raw = make_objective_column(census_raw)
census_raw = drop_columns(census_raw)

#####################
### DATA CLEANING ###
#####################

# Copy the raw training data and replace the objective function with classes of 1 and 0
# We will have to look at the existing values of the objective variable so that we can 
# be sure we are replacing all values with either a 1 or 0
census_clean = census_raw.copy()
census_clean[objective].unique()
#census_clean[objective].replace("-50000",0,inplace=True)
#census_clean[objective].replace(" 50000+.",1,inplace=True)
census_clean[objective].unique()

census_clean_test = census_raw_test.copy()
census_clean_test[objective].unique()
#census_clean_test[objective].replace("-50000",0,inplace=True)
#census_clean_test[objective].replace(" 50000+.",1,inplace=True)
#census_clean_test[objective].replace("- 50000",0,inplace=True)
#census_clean_test[objective].replace(" - 50000.",0,inplace=True)

# Fill in Numeric NA's with -9999 for the test set. This will differentiate missing values from naturally occuring values.
# We will do this later for the training data, since we first want to see how many
# observations are missing for each feature.
census_clean_test = census_clean_test.fillna(0)

# Label categorical variables that need to be set as categorical manually
census_clean[categorical_features] = census_clean[categorical_features].astype(str)
census_clean_test[categorical_features] = census_clean_test[categorical_features].astype(str)


######################################
######### FEATURE DISCOVERY ##########
######################################

 
#### Calcualte Statistics for Each Feature   ###
# The get_column_stats function takes in a dataset, a feauture, and an objective variable
# and produces a summary of key statistics about this column, including:
# whether the column is numeric, the number of missing observations,
# infromation_value, spearman correlation, pearson correlation,
# breakdowns of the objective variable by feature percetnile - this includes
# the percent of high_income vs. low_income individuals within a decile
# and the distributions of high_income and low_income individuals across deciles.
# For categorical variables, this breakdown is done by category instead of percentile. 
 
def get_column_stats(df,x,y,bins=20): 
    df['feature'] = x
    #Check if feature is numeric
    if np.issubdtype(df[x].dtype, np.number):
       df['numeric'] = 1
       #For numeric variables, count missing as Nan
       df['missing_count'] = df[x].isna().sum()
       df[x] = df[x].fillna(0)
       #, if so split into 20 groups based on percentile
       df['binned_feature'] = pd.qcut(df[x],bins,duplicates='drop')
       pearson_r = abs(df[x].corr(df[y],method = 'pearson')).min()
       spearman_r = abs(df[x].corr(df[y],method = 'spearman')).min()
    else:
       df['numeric'] = 0
       #IF feature is not numeric, count strings that indicate a missing value
       df['missing_count'] = df[x][df[x].isin(['?','Not in universe',' Not in universe','Not in universe ',"NA","NULL"])].count()
       df['binned_feature'] = df[x]   
       pearson_r = 0
       spearman_r = 0
    #Count the number and percent of high_income and low_income individuals in each group
    df =  df.groupby(['feature','binned_feature','missing_count','numeric'])[y].agg({'total_count': 'count',
             'high_income':'sum'}).reset_index()
    df['low_income'] = df['total_count'] - df['high_income']
    df['low_income_percent'] = df['low_income']/df['total_count']
    df['high_income_percent'] = df['high_income']/df['total_count']
    df['total_dist'] = df['total_count']/df['total_count'].sum()
    # Get the distribution of high and low income individuals across groups
    df['high_dist'] = df['high_income']/df['high_income'].sum()
    df['low_dist'] = df['low_income']/df['low_income'].sum()
    # Calculate information value of the feature
    df["woe"] = np.log(df['high_dist']/df['low_dist']).replace(np.inf,0).replace(-np.inf,0)
    df['binned_iv'] = (df['high_dist'] - df['low_dist'])*df["woe"]
    df['total_iv'] = df["binned_iv"].sum()
    df['split_info'] = -np.log(df["total_dist"])*df["total_dist"]
    if df["numeric"].max() == 1:
       df = df.sort_values(by = 'binned_feature', ascending = True)
    else:
       df = df.sort_values(by = 'high_income_percent', ascending = True)
    # Calculate Kolmogorov_Smirnov Statistic - A measure of how differently 
    # the high income and low income groups are disribted
    df['high_dist_cuml'] = df['high_dist'].cumsum()
    df['low_dist_cuml'] = df['low_dist'].cumsum()
    df["ks"] = (df['high_dist'] - df['low_dist']).max()
    df["spearman_r"] = spearman_r
    df["pearson_r"] = pearson_r
    return df
 
# Create the lit of features that we want to get stats for    
feature_df = pd.DataFrame()
feature_list = list(census_clean)
feature_list.remove(objective)

# For each feature, get the feature stats and save them all to a dataframe
for var in feature_list:
   var_stats = get_column_stats(census_clean[[objective,var]].copy(), var,objective,20)
   var_iv = var_stats['total_iv'].max()
   var_ks = var_stats['ks'].max()
   print("Information Value:")
   print(var +": " + str(var_iv))
   print("KS:")
   print(var +": " + str(var_ks))
   feature_df = feature_df.append(var_stats)
feature_df = feature_df.sort_values(by='feature')

# Fill in NA's with -9999. This will differentiate these values from a naturally occuring value like 0.
# We do not need the NA's now that we counted the number of missing obs in each feature
census_clean = census_clean.fillna(0)


############################################
###  PLOT AND SAVE FEATURE DISTRIBUTIONS ###
############################################

# Next we look at distributions of our features visually with matplotlib
# Using openpyxl, we can save our tables and graphs to an excel workbook to view later 
writer = pd.ExcelWriter(output_directory + '/' + 'variable_performance5.xlsx',
                        engine="openpyxl")

# Get the feature statistics that we want to save to excel
feature_summary = feature_df[['feature','missing_count','total_iv','numeric','ks','spearman_r','pearson_r']].groupby('feature').max()
feature_summary.to_excel(writer, sheet_name='Feature Summary',
             startrow=0, startcol=0, header=True, index=True)

# Save stats to excel - each column has its own sheet - and close the workbook
for var in feature_list:
   if(feature_df['numeric'][feature_df['feature'] == var].max()==1): 
        sort_column = "binned_feature" 
   else: 
        sort_column = "high_income_percent"
   feature_dist_stats = feature_df[feature_df['feature'] == var][['binned_feature','total_count','high_income','low_income','high_income_percent','low_income_percent','high_dist','low_dist']]
   feature_dist_stats.sort_values(by=sort_column).to_excel(writer, sheet_name=var[0:30],
             startrow=0, startcol=0, header=True, index=False)
writer.save()  
writer.close()

# Reopen workbook to save graphs
wb = openpyxl.load_workbook(output_directory + '/' + 'variable_performance5.xlsx')
for var in feature_list:
   if(feature_df['numeric'][feature_df['feature'] == var].max()==1): 
        sort_column = "binned_feature" 
   else: 
        sort_column = "high_income_percent"
   ws = wb.worksheets[feature_list.index(var)+1]
   # Create a workbook showing the marginal distributions of each feature
   bar_plot = feature_df[feature_df['feature'] == var].sort_values(by=sort_column).plot(x = 'binned_feature',
          y=['high_income_percent'],kind = 'bar',figsize=(8,18),
          title=(var+" marginal distribution"))
   plt.savefig(output_directory +"/bar_fig_" + var + ".png")
   plt.close()
   bar_img = openpyxl.drawing.image.Image(output_directory +"/bar_fig_"+var+".png")
   bar_img.anchor = 'I1'
   ws.add_image(bar_img)
   # If the feature is numeric we can also get a cumulative distribution of the feature
   #if feature_df['numeric'][feature_df['feature'] == var].max() == 1:
   dist_df = feature_df[feature_df['feature'] == var].sort_values(by=sort_column) 
   dist_df['high_dist_cuml'] = dist_df['high_dist'].cumsum()
   dist_df['low_dist_cuml'] = dist_df['low_dist'].cumsum() 
   dist_plot = dist_df.plot(x = 'binned_feature',
          y=['high_dist_cuml','low_dist_cuml'],kind = 'line',figsize=(8,18),
          title=(var+" cumulative distribution"))
   plt.xticks(rotation=90)
   plt.savefig(output_directory +"/dist_fig_" + var + ".png", title=(var+" cumulative distribution"))
   plt.close()
   dist_img = openpyxl.drawing.image.Image(output_directory +"/dist_fig_"+var+".png")
   dist_img.anchor = 'Q1'
   ws.add_image(dist_img)
 
# Save the workbook and close it
wb.save(output_directory + '/' + 'variable_performance6.xlsx')
wb.close()

##################################
### WEIGHT OF EVIDENCE BINNING ###
##################################

# Weight of Evidence binning is a method to combine classes of categorical variables
# that have similar lielihoods of the classes of the objective variable.
# If combining two classes does not reduce information value for this feature by
# a significant amount, then we will combine the bins. The threshold for change in 
# information value is specified in the first section of this script.

# The main benefits Weight of Evidence binning include dimensionality-reduction
# descreased over-fitting, less correlation among of predictors, 
# treatment of outliers, and usually a more interpretable model.

# The education feature is a great example. The binning algorithm identified that
# the following groups look similar and could be combined without signfiantly 
# decreasing the information value of the variable. It is easy to see how these classes
# for the education variable are similar and it makes intuitive sense that they
# are being grouped together, it is simply everyone who did not finish High School:
# Children, Less than 1st grade, 5th or 6th grade, 1st 2nd 3rd or 4th grade,
# 7th and 8th grade,9th grade,10th grade,  11th grade, 12th grade no diploma

# Calculate information gain ratio given a df describing the distribution of the objective 
# by feature cateogry.
def information_gain_ratio(df):
    split_info = (-1)*sum( (df['total_count']/df['total_count'].sum())*np.log(df['total_count']/df['total_count'].sum()) )
    info_1 = (-1)*sum( (df['high_income']/df['high_income'].sum())*np.log(df['high_income']/df['high_income'].sum()).replace(-np.inf,0).replace(np.inf,0) )
    df = df[['high_income','woe_binned_feature']].groupby('woe_binned_feature').sum().reset_index()
    info_2 = (-1)*sum( (df['high_income']/df['high_income'].sum())*np.log(df['high_income']/df['high_income'].sum()).replace(-np.inf,0).replace(np.inf,0) )
    return (info_2 - info_1)/split_info


# Each feature will have a set of binned_features, which are the original groups
# for the feature, and each feature will have a set of woe_binned_features, which
# are the final bins after WoE binning is compelte.

woe_bin_df = pd.DataFrame()
feature_df["woe_binned_feature"] = feature_df["binned_feature"]
for var in feature_list:
    # Grab only the rows relevant to this variable
    initial_bins_df = feature_df[feature_df['feature'] == var]
    #The order matters for WoE binning
    #If the feature is numeric, sort the bins by the value of the feature, starting with
    # the lowest percetniles. If categorical, sort the bins by distriution of the
    # objective function.
    numeric = initial_bins_df['numeric'].max()
    if(numeric==1): 
        sort_column = "woe_binned_feature" 
    else: 
        sort_column = "high_income_percent"
    initial_bins_df = initial_bins_df.sort_values(by=sort_column)
    #Iterate through each bin associated with the feature
    for i in range(1,initial_bins_df['woe_binned_feature'].count() ):
      # Create a dataset that merges two bins and calcualte information gain ratio
      combined_bins_df = initial_bins_df.copy()
      prev_bin = list(combined_bins_df['woe_binned_feature'])[i-1]  
      this_bin = list(combined_bins_df['woe_binned_feature'])[i]
      if numeric==1:
         new_bin = pd.Interval(prev_bin.left,this_bin.right)
      else:
          new_bin = prev_bin + "," + this_bin
      combined_bins_df["woe_binned_feature"][combined_bins_df["woe_binned_feature"]==prev_bin] = new_bin
      combined_bins_df["woe_binned_feature"][combined_bins_df["woe_binned_feature"]==this_bin] = new_bin
      # If the change does not reduce information vaalue greatly, keep the change
      if information_gain_ratio(combined_bins_df) > information_gain_cutoff:
          initial_bins_df = combined_bins_df.copy()
          print("Combining bins " + str(prev_bin) + " and " + str(this_bin))
    woe_bin_df = woe_bin_df.append(initial_bins_df)

feature_df = woe_bin_df
feature_df["binned_feature_name"] = "binned_" + feature_df["feature"] 
# The woe_binned_feature column now contains the optimal binning for each feature

#########################
### MAPPING THE BINS ####
#########################

# The apply map function will help us map feature values to bins
bin_map = ''      
def apply_map(x):
    for key in bin_map:
        if x in key:
            return key
    return bin_map[-1]

#    return "missing"

# Create a dataframe that has the objective function and add in the binned features
modeling_df = pd.DataFrame()
modeling_df[objective] = census_clean[objective]
for var in feature_list:
    # For each feature, get the optimally binned features as a list
    bin_map = list(feature_df["woe_binned_feature"][feature_df['feature']==var])
    # Map the observed values to the bins that they fall into, and place them in the modeling df
    modeling_df["binned_"+var] = census_clean[var].map(apply_map)
    
binned_model_features  = list(modeling_df)[1:]
# Change the variable types to strings for all features
# Check that none are missing and fill in 'missing' if they are. This does not happen for our data.
modeling_df[binned_model_features] = modeling_df[binned_model_features].astype(str).fillna("missing")


# Just as we did with the training data, map the feature values to bins for test data
modeling_df_test = pd.DataFrame()
modeling_df_test[objective] = census_clean_test[objective]
for var in feature_list:
    bin_map = list(feature_df["woe_binned_feature"][feature_df['feature']==var])
    modeling_df_test["binned_"+var] = census_clean_test[var].map(apply_map)    
modeling_df_test[binned_model_features] = modeling_df_test[binned_model_features].astype(str).fillna("missing")



##############################################
### CORRELATION & MULTICOLLINEARITY CHECKS ###
##############################################

# To satisfy OLS asusmptions, we need to check that out model features
# are independent and are not collinear. 
# Since our variables are cateogrical, we cannot use pearson correlation to 
# test independence. A wwo-way Chi-Squared test is a good substitute when 
# dealing with categoricals.
# There is no ideal multicollinearity test for categoricals. Seperating the categories
# into many dummy vairables and checking the variance-inflation-factor of the entire
# feature set is a good approach, but will make it harder for vairables with many categories
# to pass the VIF check. Similarly, variables that are mostly made up of a single class
# will have a tougher time passing the VIF check. 

# Computes p-value of chi_squared test for two features
def chi_squared_test(a,b):
    tab = pd.crosstab(pd.Series(a),pd.Series(b))
    return scipy.stats.chi2_contingency(tab)[1]

# For every pair of features, get the p-value of a chi squared test to test independence
# A p-value above 0.05 will indicate the variables are not independent and cannot both be used
# This process can take a few minutes
chi_sq_df = pd.DataFrame()  
i = 0
percent_complete = 0
for x in binned_model_features:
    for y in binned_model_features:
        p_value = chi_squared_test(modeling_df[x].values,modeling_df[y].values)
        chi_sq_df.loc[x,y] = p_value
        #Tracking for what % of pairwise chi squared tests are complete
        i = i +1
        percent_complete_new = round(100*i/(len(binned_model_features)*len(binned_model_features))) 
        if percent_complete_new>(percent_complete+0.1):
           percent_complete = percent_complete_new
           print("Running Chi Squared Tests - " + str(percent_complete) + "% Complete")

# Rank the features by our specified feature_selection criterion and test them out in order
# If the feature passes the Chi Squred and VIF checks, include it in the model
# Stop the search when we run out of features or have hit our specified max_features limit

ranked_features = list("binned_" + feature_df.sort_values(by=criterion, ascending = False)['feature'].unique() )
model_feature_list = [ranked_features[0]]

for var in ranked_features[1:]:
    print("")
    print(var)
    # Run chi squared test for this feature, and all features that have already been chosen
    
    if use_chi_squared_test==True:
        max_chi_p = chi_sq_df[var][chi_sq_df.index.isin(model_feature_list)].max()
    else:
        max_chi_p = 0.00
    if max_chi_p < 0.05:
        over_vif_cutoff= False
        if use_vif_test==True:
            # Break our existing features into dummy vairables, each with one cateogry removed
            unique_bin_count = len(feature_df['woe_binned_feature'][feature_df['binned_feature_name']==var].unique() )
            dummy_df = pd.get_dummies(modeling_df[model_feature_list+[var] ],drop_first=True)
            dummy_values = dummy_df.values
            # Run the VIF for every new column that we are adding
            for x in range( len(list(dummy_df))-unique_bin_count-1, len(list(dummy_df))-1):
                # Check VIF does not increase past 10.0 when adding any of the dummy columns to the model set of features
                # Using statsmodels variance inflation factor function
                vif = variance_inflation_factor(dummy_values,x)
                # If any dummy fails the VIF test, move on to next variable
                if vif > 10:
                   over_vif_cutoff = True
                   break
             # If feature passes both tests, add it to our list of model features
        if(over_vif_cutoff==False):
             print("Added!")
             model_feature_list.append(var)
        else:
            print("Fails VIF Test")
    else:
        print("Fails Chi Sq test")
    # If we have hit our max feaure limit, stop the search
    if len(model_feature_list) == max_features:
        break

##########################################
### BUILD FINAL TRAINING DATASETS ######
##########################################    
    
# Create Model feature and model ojective datasets
# Get dummies for our categorical model features, each dropping one cateogry
modeling_X = modeling_df[model_feature_list]
modeling_y =modeling_df[objective]

# Repeat for testing data
test_X =modeling_df_test[model_feature_list]
test_y =modeling_df_test[objective]

# Calculate the sample weights that would be needed to get balanced classes 
# for the high-income and low-income groups
low_count = modeling_y[modeling_y == 0].count()
high_count = modeling_y[modeling_y == 1].count()
high_weight = low_count/high_count
modeling_w = modeling_y.replace(1,high_weight)
modeling_w = modeling_w.replace(0,1)

# Get weights we need for balanced classes on our testing data
low_count = test_y[test_y == 0].count()
high_count = test_y[test_y == 1].count()
high_weight = low_count/high_count
test_w = test_y.replace(1,high_weight)
test_w = test_w.replace(0,1)

# Uniform_w will be used if we want to give an equal weight to each observation
uniform_w = None



###########################################
### LOGISTIC CLASSIFICATION TRAINING ######
###########################################

# Create formula for logistic model
formula = objective + " ~ C(" + ") + C(".join(model_feature_list) + ")"
# Create Model and fit results
logistic_model = smf.glm(formula=formula, data=modeling_df, family=sm.families.Binomial(),
                         freq_weights=modeling_w)
logistic_model_results = logistic_model.fit()

# Store probabilties produced by model. 
# Use rounding to map probabilites to binary classes
logistic_pred_train = logistic_model_results.predict(modeling_df[model_feature_list])
logistic_model_df_train = pd.concat([modeling_y,pd.Series( logistic_pred_train )],axis=1)
logistic_model_df_train.columns = ['outcome','prob']
logistic_model_df_train['pred'] = logistic_model_df_train['prob'].round(0)

# Show Confusion Matrix of Predictions and Outcomes
pd.crosstab(logistic_model_df_train.pred,logistic_model_df_train.outcome)

# Use an F test to test that all the classes of a cateogical variables are jointly significant
# The F test will also gie us the Sum of Squares of each variable,
# which explains which variables are driving variance in the model.

# Run F test for each feature. Store results in a df
# Due to a limitation in statsmodels, we have to insert the SSR into the model manually
logistic_model_results.ssr = sum( np.square(logistic_pred_train - logistic_pred_train.mean() ) )
F_test = stats.anova.anova_single(logistic_model_results)

###########################################
### BOOSTING CLASSIFICATION TRAINING ######
###########################################

# Train and fit a boosting model, we will tune the parameters later
# The CatBoost model is a boosting method designed to handle categoricals
boost_model = CatBoostClassifier(iterations=20,
                          learning_rate=0.2,
                          depth=3)
boost_model = boost_model.fit(census_clean[feature_list],modeling_y,sample_weight=modeling_w,
                               cat_features= np.where(census_clean[feature_list].dtypes == np.object )[0]
                              )
boost_pred_train = boost_model.predict(census_clean[feature_list])

# Build Confusion Matrix
boost_model_df_train = pd.concat([modeling_y,pd.Series( boost_pred_train )],axis=1)
boost_model_df_train.columns = ['outcome','pred']
# Show Confusion Matrix of Predictions and Outcomes
pd.crosstab(boost_model_df_train.pred,boost_model_df_train.outcome)
# Get accuracy
metrics.accuracy_score(logistic_model_df_train.pred,logistic_model_df_train.outcome)



# HYPERPARAMTER TUNING

# Tune the Boosting tree that we just started with using sklearns GridSearch
# Results are cross-validated with 5 folds
boost_search = model_selection.GridSearchCV(boost_model, 
             {'depth':[2,3,4,5],
              'learning_rate':[0.01,0.1,0.2,0.25]}
             , cv=5).fit(census_clean[feature_list],modeling_y,sample_weight=modeling_w,
                               cat_features= np.where(census_clean[feature_list].dtypes == np.object )[0]
                              )
boost_search_pred_train = boost_search.predict(census_clean[feature_list])
#boost_search_pred_train_saved = boost_search_pred_train

# Build Confusion Matrix
boost_search_df_train = pd.concat([modeling_y,pd.Series( boost_search_pred_train )],axis=1)
boost_search_df_train.columns = ['outcome','pred']
# Show Confusion Matrix of Predictions and Outcomes
pd.crosstab(boost_search_df_train.pred,boost_search_df_train.outcome)
# Show accuracy
metrics.accuracy_score(boost_search_df_train.pred,boost_search_df_train.outcome)

# View feature importance for the best specification
f_imp = boost_search.best_estimator_.get_feature_importance(prettified=True)

# View the parameters that created the best model
boost_search.best_estimator_.get_params()

########################################
### DEPLOY MODELS ON TESTING DATA ######
########################################

# Use rounding to map probabilites to binary classes
logistic_pred_test = logistic_model_results.predict(modeling_df_test[model_feature_list])
logistic_model_df_test = pd.concat([test_y,pd.Series( logistic_pred_test )],axis=1)
logistic_model_df_test.columns = ['outcome','prob']
logistic_model_df_test['pred'] = logistic_model_df_test['prob'].round(0)

# Show Confusion Matrix of Predictions and Outcomes
pd.crosstab(logistic_model_df_test.pred,logistic_model_df_test.outcome)
pd.crosstab( (logistic_model_df_test.prob).round(1),logistic_model_df_test.outcome)

# Get accuracy
metrics.accuracy_score(logistic_model_df_test.pred,logistic_model_df_test.outcome)

# Apply Boosting model to test data
boost_search_pred_test = boost_model.predict(census_clean_test[feature_list])
# Build Confusion Matrix
boost_search_df_test = pd.concat([test_y,pd.Series( boost_search_pred_test )],axis=1)
boost_search_df_test.columns = ['outcome','pred']
# Show Confusion Matrix of Predictions and Outcomes
pd.crosstab(boost_search_df_test.pred,boost_search_df_test.outcome)
# Get accuracy
metrics.accuracy_score(boost_search_df_test.pred,boost_search_df_test.outcome)



################################
#### VIEW MODEL RESULTS ########
################################


# Save F-test and Feature Importance tables
F_test.to_csv(output_directory + "/" + "logistic_model_anova.csv", header=True)
f_imp.to_csv(output_directory + "/" + "boosting_model_fimp.csv", header=True)

# Save feature bins
feature_df.to_csv(output_directory + "/" + "binned_features_df.csv", header=True, index=False)

# Create model results file
logistic_summary = logistic_model_results.summary()


f = open(output_directory + "/Model_Results.csv","w")
f.write(logistic_model_results.summary().as_csv())
f.close()

f = open(output_directory + "/Model_Results.txt","w")
f.write(logistic_model_results.summary().as_text())
f.close()


# Logistic Training Results
logistic_train_acc = metrics.accuracy_score(logistic_model_df_train.pred,logistic_model_df_train.outcome)
logistic_train_recall = metrics.recall_score(logistic_model_df_train.pred,logistic_model_df_train.outcome)
logistic_train_prec = metrics.precision_score(logistic_model_df_train.pred,logistic_model_df_train.outcome)

# Logistic Test Set Results
logistic_test_acc = metrics.accuracy_score(logistic_model_df_test.pred,logistic_model_df_test.outcome)
logistic_test_recall = metrics.recall_score(logistic_model_df_test.pred,logistic_model_df_test.outcome)
logistic_test_prec = metrics.precision_score(logistic_model_df_test.pred,logistic_model_df_test.outcome)

# Boost Training Results
boosting_train_acc = metrics.accuracy_score(boost_model_df_train.pred,boost_model_df_train.outcome)
boosting_train_recall = metrics.recall_score(boost_model_df_train.pred,boost_model_df_train.outcome)
boosting_train_prec = metrics.precision_score(boost_model_df_train.pred,boost_model_df_train.outcome)


# Boost Grid Search Training Results
boosting_search_acc = metrics.accuracy_score(boost_search_df_train.pred,boost_model_df_train.outcome)
boosting_search_recall = metrics.recall_score(boost_search_df_train.pred,boost_model_df_train.outcome)
boosting_search_prec = metrics.precision_score(boost_search_df_train.pred,boost_model_df_train.outcome)

# Boost Grid Search Testing Results
boosting_search_acc = metrics.accuracy_score(boost_search_df_test.pred,boost_search_df_test.outcome)
boosting_search_recall = metrics.recall_score(boost_search_df_test.pred,boost_search_df_test.outcome)
boosting_search_prec = metrics.precision_score(boost_search_df_test.pred,boost_search_df_test.outcome)



######################
##### END SCRIPT #####
######################



