
#this script contains functions that calculate the numeric values shown on the quality metric plots


import scanpy as sc
import dropkick as dk
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
import seaborn as sn 
import anndata as ad

from sklearn.metrics import auc as auc
from scipy.optimize import curve_fit
from scipy import stats


#//TODO: change this area
#//chnage zc_function to data_processing 
import sys
sys.path.append("/home/lucy/")
import zc_function as zc
sys.path.append("/home/lucy/STAR_Protocol/")
import QCPipe
sys.path.append("/home/lucy/quality/")
import zc_quality_function as zc_qc




BIN_SIZE = 100 # this size should be set as a constant between samples to compare quality 
SCALE_SLOPE = True




def add_qc_ambient_metrics(sample_dat, dropout_threshold =2):
    
    #add dropout values 
    if( not('pct_dropout_by_counts'  in sample_dat.var )): 
        sample_dat.var["pct_dropout_by_counts"] = np.array(
            (1 - (sample_dat.X.astype(bool).sum(axis=0) / sample_dat.n_obs)) * 100
            ).squeeze()
    num_ambient = len( sample_dat.var["pct_dropout_by_counts"][ sample_dat.var["pct_dropout_by_counts"]<=dropout_threshold])
    
    #add ambient var column and pct_counts_ambient in obs
    if('pct_counts_ambient' not in sample_dat.obs.columns):
        lowest_dropout = sample_dat.var.pct_dropout_by_counts.nsmallest(n=num_ambient).min()
        highest_dropout = sample_dat.var.pct_dropout_by_counts.nsmallest(n=num_ambient).max()
        sample_dat.var["ambient"] = sample_dat.var.pct_dropout_by_counts <= highest_dropout
        
        sc.pp.calculate_qc_metrics(
            sample_dat, qc_vars=[ "ambient"], inplace=True, percent_top=None)
    sample_dat.uns["num_ambient"] = num_ambient
    
    return   
        
    


    
#frequency * slope curve area ratio
#can use plt.ioff() functio nto disable figure output 
#TODO: double check for potential scaling than more than 1 times (see plot function plot 2)
def bin_cumsum_slope( sample_dat, scale = True, mode = -1, scale_slope_vtr = SCALE_SLOPE):
    """This function takes-in an anndata object, computes the anndata obj's obs["total counts"]'s cumulative sum distribution curve, and return the bin heights and widths after binning the slope of the curve at each data point
    
    param sample_dat: the anndata obj
    param scale: set True to scale the x and y axis to values between 0 and 1
    param mode: this parameter determines the cut-off method used to exlude the high-frequency bins for very low slopes from the cumsum curve. 
        If it is an integer >0, the cut-off will be the mean+ 1*stdev of the last portion of the cumsum values, where the portion =(inlfection_fold -mode)/inflection_fold. 
        If it is 0.5, it will simply set the cut-off at the mean of the entire cumsum data points. 
        If it is 0, it will set the cut off at mean of all slopes + 1 std of all slopes
        If it is -1, it will set the cut-off at the median of all slopes + 1std of all slopes
        Default set to -1 (the median) 
    
    return: thresh, ret, 
        where 
        "mean_grad" is the average slope at the end portion of the cumsum curve, and the portion is (inflection_fold-1)/inflection_fold
        "ret" contains bin heights, bin edges and patch (a plotting container)"""
    
    sample_cumsum =  np.cumsum(sample_dat.obs['total_counts']) #cumulative sum 
    x1 = np.arange(0,sample_dat.n_obs) # x axis positions 
    
    if(scale):
        x1 = minmax_scale(x1)
        sample_cumsum = minmax_scale(sample_cumsum) 
    
    
    s1_grad = np.gradient(sample_cumsum, x1) #slope at each point for the cumulative sum curve
    
    if(scale_slope_vtr):
        #scale the gradient so the sum of this data vector is 1
        s1_grad = np.divide(s1_grad, s1_grad.sum())
    
    
    thresh = get_freq_slope_bin_cut(mode, sample_dat, s1_grad)
    
    _, ax = plt.subplots(figsize=(4, 4))
    ret = ax.hist(s1_grad, bins = BIN_SIZE) # n, bin, patch
    #where n are bin heights and bin are bin edges 
    plt.ioff()
    
    return thresh, ret


def get_freq_slope_bin_cut(mode, sample_dat, s1_grad):
    """This is a helper function for bin_cumsum_slope() function that helps to find the freq*slope curve cut-off to exlude the bins of very low slopes
    param mode: this parameter determines the cut-off method used to exlude the high-frequency bins for very low slopes from the cumsum curve. 
        If it is an integer >0, the cut-off will be the mean+ 1*stdev of the last portion of the cumsum values, where the portion =(inlfection_fold -mode)/inflection_fold.  
        If it is 0, it will set the cut off at mean of all slopes + 1 std of all slopes
        If it is -1, it will set the cut-off at the median of all slopes + 1std of all slopes
        Default set to -1 (the median) 
    param sample_dat: the anndata object whose data was used in the calculation
    param s1_grad: the slope of the cumsum curve in numpy array
    
    return: a single value below which the freq*slope curve's x values should be excluded """
    
    #TODO: better documentation and consider to remove the modes other than median 
    
    if(mode == 0):
        # try to find a approximate normal distribution of the major peak and cut 1 std after the mean
        s1_grad_mean = s1_grad.mean()
        s1_grad_std = s1_grad.std()
        thresh = s1_grad_mean + s1_grad_std
        #print(f"from calc get freq slope cut func mode 0: thrsh = {thresh}")
        return thresh
    
    if(mode == -1):
        # try to find a approximate normal distribution of the major peak and cut 1 std after the median
        s1_grad_mean = np.median(s1_grad)
        s1_grad_std = s1_grad.std()
        thresh = s1_grad_mean + s1_grad_std
        #print(f"from calc get freq slope cut func mode -1: thrsh = {thresh}")
        return thresh
    
    else:
        raise ValueError("mode number should 0 or -1")
        return 
    
    
        
            


        
#change all high_slope_area to high_slope_area       
def high_slope_area( sample_dat, thresh = None, scale = True, ret = None, mode = -1):  
    """This function calculate the ratio value shown on the slope* frequency plot (plot3)
    param sample_dat: the anndata obj from which the values are calculated
    param thresh: a threshold used to exlude the bins that have high frequency and very low slopes potentially from ambient genes 
    param scale: set True if want to scale the x and y axes value to 1, see the description of bin_cumsum_slope()
    param mode: this parameter determines the cut-off method used to exlude the high-frequency bins for very low slopes from the cumsum curve. 
        If it is an integer >0, the cut-off will be the mean+ 1*stdev of the last portion of the cumsum values, where the portion =(inlfection_fold -mode)/inflection_fold. 
        If it is 0, it will set the cut off at mean of all slopes + 1 std of all slopes
        If it is -1, it will set the cut-off at the median of all slopes + 1std of all slopes
        Default set to -1 (the median)  
    
    return: values required for the plots: res, mean_slope and high_slope_area, 
    where 
    "res" is the list of freq*slope values, 
    "mean_slope" is a list of slope values for the bin (AKA mid-points of each bin at x-axis )
    "high_slope_area" is the area ratio between the curve and the min square that can enclose it and a list of mean slopes for the bins """
    
    if(ret is None):
        thresh, ret = bin_cumsum_slope(sample_dat, scale, mode = mode) #get the binning result 
        
    
    res=ret[0]*ret[1][0:-1] # frequency value for each bar (AKA bin height)  * bin width 
    bin_ticks = ret[1]
    
    #get a list of slope values for the bin (AKA mid-points of each bin at x-axis )
    mean_slope= [0.5*( bin_ticks[i+1] + bin_ticks[i] ) for i in range(len(bin_ticks) -1)]
    
    #cut down mean slope where <= thresh
    mean_slope = np.array(mean_slope)
    
    mean_slope2 = mean_slope[mean_slope > thresh]
    res2 = res[len(res) - len(mean_slope2) : len(res)]
    
    
    high_slope_area = res2.sum()
    
    return res, mean_slope, high_slope_area, thresh
    
    

# max secant distance
# secant line std
# cumulative sum area ratio
def secant_metrics(sample_dat, scale = True): 
    """ This function calculates the values shown on the secant line plot (plot 4)
    
    param sample_dat: the anndata obj from which, values are calculated
    
    return: the max secant distance, the standard dev. of the secant lines and the area ratio between the min square that can enclose the curve"""
    
    cumsum = np.cumsum(sample_dat.obs['total_counts'])
    if(scale):
        minmax_scale(cumsum, copy= False)
    x_vals=np.arange(0,sample_dat.n_obs) #unit x values 
    
    #calculate secant line length
    secant_coef= cumsum[sample_dat.obs.shape[0]-1]/sample_dat.obs.shape[0]

    secant_line=secant_coef*x_vals
    secant_dist=abs(cumsum-secant_line)
    std_val = np.std(secant_dist)
    ratio = zc_qc.area_ratio_sample(sample_dat)
    max_dist = max(secant_dist)
    
    return [max_dist, std_val, ratio ] 




# number of ambient genes
def num_ambient(sample_dat, dropout_threshold = 2):
    """This function calculate the number of ambient genes of a sample_dat based on a dropout_threshold
    return the number of ambient gene and modified sample_dat obj with 'pct_dropout_by_counts' added in its var column"""
    
    if( not('pct_dropout_by_counts'  in sample_dat.var )): 
        sample_dat.var["pct_dropout_by_counts"] = np.array(
            (1 - (sample_dat.X.astype(bool).sum(axis=0) / sample_dat.n_obs)) * 100
            ).squeeze()
        
    num_ambient = len( sample_dat.var["pct_dropout_by_counts"][ sample_dat.var["pct_dropout_by_counts"]<=dropout_threshold])
    
    return num_ambient, sample_dat



# mean pct count ambient

def mean_pct_ambient(sample_dat, num_ambient):
    """This function calculate the average percentage of ambient genes for sample_dat
    num_ambient should be calculated before this function, and pct_dropout_by_counts should be in the var field of the anndata object
    
    return average pct ambient gene """
    
    num_ambient = sample_dat.uns["num_ambient"]
    
    if num_ambient == 0:
        return 0
    if('pct_dropout_by_counts' not in sample_dat.var.columns):
        print("run add_qc_ambient_metrics() before this function")
        return None
    
    if('pct_counts_ambient' not in sample_dat.obs.columns):
        lowest_dropout = sample_dat.var.pct_dropout_by_counts.nsmallest(n=num_ambient).min()
        highest_dropout = sample_dat.var.pct_dropout_by_counts.nsmallest(n=num_ambient).max()
        sample_dat.var["ambient"] = sample_dat.var.pct_dropout_by_counts <= highest_dropout
        sc.pp.calculate_qc_metrics(
            sample_dat, qc_vars=[ "ambient"], inplace=True, percent_top=None
        )
    
    mean_pct_ambient = sample_dat.obs["pct_counts_ambient"].mean()
    
    return mean_pct_ambient





#--------------------------------------------------------------------------------------------------------------------------------------------------------Inverse some metrics to make them positively associated with ambient level----------------------------------------------------------------------------------------------------------------------------------------------------

def inverse_max_secant(curr_val):
    """max sencat values range (0,1), so subtract the value from 1 will give an inverted value"""
    return 1-curr_val

def inverse_scaled_slope_sum(curr_val):
    """cells' scaled_slope_sum values range (0,1), so subtract the value from 1 will give an inverted value"""
    return 1-curr_val

def inverse_auc_pct(curr_val):
    """AUC percentage values range (0.5, 1), so we subtract curr_value from 1, which will give values range from (0, 0.5).
    To scale up to (0,1), just multiply the value with 2 """
    return (1-curr_val)*2

def inverse_secant_std(curr_val):
    """From simulated data, the secant line std values range from ~0.025 to 0.225
    In extreme cases, where n is very small, std approaches 0.5, and std cannot get smaller than 0.
    Subtracting the curr_val from 0.5 is chosen to invert the value """

    return 0.5-curr_val

