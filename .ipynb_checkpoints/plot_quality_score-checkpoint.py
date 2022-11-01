#TODO: delete unused packages 
#TODO: add the inverted flag (copy from the quality folder script )
#TODO: label which figures from the paper 


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



#TODO: change this area

import sys
#import data_processing as dataproc #change zc to dataproc
sys.path.append("./QCPipe_dir/")
import QCPipe
import quality_control_function as func_qc 
import calculation as calc 

BIN_SIZE = 100


# plot 1
def plot_cum_sum(sample_dat, scale = True, ax =None, return_data = False ):
    """
    This function plots the transcript count cumsum vs ranked barcode curve.
    @param sample_dat: the anndata object of the data
    @param scale: set True if to scale the x and y axes between 0 and 1 
    @param ax: the Axes object from pyplot (or seaborn) where this plot will be shown. If set None, a new axes object will be created. Axes will always be returned 
    @param return_data: if set True, the Axes and the cumsum curve (y-values) will be returned, else only the Axes will be returned
    
    @return: see return_data
    """
    
    if (ax is None):
        _, ax = plt.subplots(figsize=(4, 4))
    
    
    s1_cumsum =  np.cumsum(sample_dat.obs['total_counts']) #cumulative sum 
    x1 = np.arange(0,sample_dat.n_obs) # x axis positions 
    
    if(scale):
        x1 = minmax_scale(x1) #scale x-axis
        minmax_scale(s1_cumsum, copy = False) #inplace scaling s1_cumsum (y-axis)
        ax.set_xlabel('Scaled Ranked Barcode')
        ax.set_ylabel('Scaled Cumulative Count')
    else:
        ax.set_xlabel('Ranked Barcode')
        ax.set_ylabel('Cumulative Count')
    
    ax.plot(x1, s1_cumsum) # plot the cumulative curve 
    
    if(return_data):
        return  ax, s1_cumsum
    else:
        return ax
    
    
    
    
#plot 2
#TODO: make sure all place can set mode 
def plot_slope(sample_dat, cum_sum = None, ax = None, return_dat = False, scale = True, mode = -1):
    """
    A frequency distribution histogram of the slope of the cumulative curve plot. Number of bins set to 100. 
    @param sample_dat: the anndata object of the data
    @param cum_sum: cumulative curve that will be used (returned value from plot1). If None, calculate from data
    @param ax: the Axes object from pyplot (or seaborn) where this plot will be shown. If set None, a new axes object will be created. Axes will always be returned
    @param return_data: if set True, the [ax, ret, thresh] will be returned, else only the Axes will be returned. ax is the Axes object, ret is a list of frequency values for the histgram bins, and thresh is a threshold calculated for plot3. Else, only return the Axes
    @param scale: set True if to scale the CUMSUM CURVE's x and y axes between 0 and 1 
    @param mode: the mode to cut-off slopes for plot 3. See calculation.get_freq_slope_bin_cut() function for detailed description. Default set to -1: threshold = median of slopes + 1 stdev of the slopes
    
    @return see return_data
    
    """
    
    if (ax is None):
        _, ax = plt.subplots(figsize=(4, 4))
        
    x1 = np.arange(0,sample_dat.n_obs) # x axis positions 
    
    if(cum_sum is None):
        cum_sum = np.cumsum(sample_dat.obs['total_counts'])
        
    #this if block is newly added, if the result differs a lot, revert back to not having this block
    if(scale):
        x1 = minmax_scale(x1)
        cum_sum = minmax_scale(cum_sum)
    
    
    s1_grad = np.gradient(cum_sum, x1) #slope at each point for the cumulative sum curve 
    s1_grad_plot = s1_grad.copy()
    s1_grad = np.divide(s1_grad, s1_grad.sum()) #scale the slopes so the sum will be 1
    thresh = calc.get_freq_slope_bin_cut(mode, sample_dat, s1_grad) #get threshold for the slope*freq curve 
    
    
    ax2 = ax
    ret = ax2.hist(s1_grad, bins = BIN_SIZE)#storing the histgram bins' frequencies 
    ax2.set_xlabel('Slope')
    ax2.set_ylabel('Frequency')
    x_axis = np.linspace(min(s1_grad), max(s1_grad), 6)
    ax2.set_xticks(x_axis)
    
    x_axis_scaled = np.linspace(min(s1_grad_plot), max(s1_grad_plot), 6)
    ax2.set_xticklabels([f"{i:.2f}" for i in x_axis_scaled])
    
    if (return_dat):
        return ax, ret, thresh # the result needed for frequency*bin_width plot
    
    return ax


#plot 3 

def plot_freq_weighted_slope(sample_dat,  thresh = None ,ax = None, ret = None, return_dat = False, scale = True, mode = -1):
    """
    Scaled slope sum plot, where the y values are bins' freqency (height)* bin's x-axis midpoint(mean slope of cumsum curve) for the bins in plot2's histgram. The x-values are slopes of cumsum curve (same with plot 2). The pink color indicates an estimated real cell region whereas blue color indicates empty droplets. 
    
    @param sample_dat: the anndata object of the data
    @param thresh: the threshold above which real cell regions are defined. A returned value from the plot_slope() function, but will be calculated if None is passed in.
    @param ax: the Axes object from pyplot (or seaborn) where this plot will be shown. If set None, a new axes object will be created. Axes will always be returned
    @param ret: a list of frequency values for the histgram bins. A returned value from the plot_slopw() function but will be calculated if None is passed in. 
    @param return_dat: if set True, the [ax, high_slope_area] will be returned, else only the Axes will be returned. high_slope_area is the numerical value of the area colored in pink as real cell region.
    @param scale: set True to scale the x and y axis to values between 0 and 1 for the cumsum curve (only used if ret is None)
    @param mode: the mode to cut-off slopes for plot 3. See calculation.get_freq_slope_bin_cut() function for detailed description. Default set to -1: threshold = median of slopes + 1 stdev of the slopes 
    
    @return: see return_dat
    

    """
    
    if (ax is None):
        _, ax = plt.subplots(figsize=(4, 4))
    
    if(ret is None):
        #calculate required results
        thresh, ret = calc.bin_cumsum_slope( sample_dat, scale, mode = mode) 
    
    #getting numerical values from the calculation moodule 
    res, mean_slope, high_slope_area, thresh = calc.high_slope_area( sample_dat, thresh, scale = scale, ret = ret)
    
    #high slopes are those beyound threshold 
    high_slopes = mean_slope[mean_slope>thresh]
    #res are the mean_slope* frequency values (the y-values for this plot)
    #getting the y-values that sohuld be colored as high slope region
    high_res = res[len(res) - len(high_slopes) : len(res)]
    
    ax5 = ax
    ax5.plot(mean_slope, res) 
    ax5.fill_between(mean_slope, res, step="pre", alpha=0.4)
    #to high light the high slope area
    ax5.fill_between(high_slopes, high_res, step="pre", alpha=0.4, color = 'salmon')
    ax5.set_xlabel('Mean Slope of Bins')
    ax5.set_ylabel('Frequency * Slope')
    ax5.text(0.6*max(mean_slope), 0.4*max(res), f"high slope area:\n {high_slope_area: .2f}")
    x_axis = np.linspace(min(mean_slope), max(mean_slope), 6)
    ax5.set_xticks(x_axis)
    ax5.set_xticklabels([f"{i:.1e}" for i in x_axis])
    
    
    if(return_dat):
        return ax, high_slope_area
    return ax


#finished annotation of plot3, TODO: start with plot4
#plot 4
def plot_secant_line( sample_dat, cum_sum = None, ax = None,  return_dat = False):
    """This function calls calc_secant_lines() function to overlay secant lines, high-light max secant lines on the cumulative sum curve with corresponding numeric annotations 
    @param sample_dat: the anndata object of the data
    @param cum_sum: cumulative curve that will be used (returned value from plot1). If None, calculate from data
    @param ax: the Axes object from pyplot (or seaborn) where this plot will be shown. If set None, a new axes object will be created. Axes will always be returned
    @param return_dat: if set True, the [ax,max_secant, std_val, cum_curve_area_ratio] will be returned, else only the Axes will be returned. max_secant is the length of the maximal secant line among all secant lines, std_val is the standard deviation of secant lines, cum_curve_area_ratio is the AUC percentage defined in the paper 
    
    @return: see return_dat param description 
    """
    if(cum_sum is None):
        cum_sum = np.cumsum(sample_dat.obs['total_counts'])
        
    #max secant line distance, secant line st. dev and AUC percentage (AKA ratio (area under the cumsum curve: area of the minimal rectangle circumscribing the cumsum curve) ) 
    [max_secant, std_val, cum_curve_area_ratio] = calc_secant_lines(sample_dat, cumsum=cum_sum, ax = ax)

    
    if return_dat:
        return ax, max_secant, std_val, cum_curve_area_ratio
    return ax




def calc_secant_lines(samp, cumsum ,  ax = None, invert_score = False):
    """ This function is a helper function that is called by plot_secant_line() and calculates the secant lines for the cumulative count curve for ranked barcode given a sample. See plot_secant_line() for more detail. Some codes are learned from QCPipe.qc.find_inflection() method"""

    if( cumsum is None):
        cumsum =  np.cumsum(samp.obs['total_counts'])
    
    cumsum = np.array(cumsum) #enforce array type for cumsum 
    x_vals=np.arange(0,samp.n_obs) #unit x values 

    #Secant line computation learned from QCPipe 
    #calculate secant line length
    secant_coef=cumsum[samp.obs.shape[0]-1]/samp.obs.shape[0]

    secant_line=secant_coef*x_vals
    secant_dist=abs(cumsum-secant_line)
    std_val = np.std(secant_dist)
    ratio = zc_qc.area_ratio_sample(samp)
    max_dist = max(secant_dist)
    max_ind = np.argmax(secant_dist)

    #plot the secant lines
    if (ax is None):
        _, ax = plt.subplots(figsize=(4, 4))
    ax.plot(x_vals, secant_line)
    ax.plot(x_vals, cumsum)
    vline_range = range(0, samp.n_obs, 200)
    
    #plot all secant lines 
    ax.vlines(x_vals[vline_range], secant_line[vline_range],cumsum[vline_range], colors="lightgrey" )
    #high light the max secant line
    ax.vlines(x_vals[max_ind], secant_line[max_ind], cumsum[max_ind], colors = 'green')
    
    if(invert_score):
        #make the results positively correlated with ambient contamination 
        max_dist = calc.inverse_max_secant(max_dist)
        std_val = calc.inverse_secant_std(std_val)
        ratio = calc.inverse_auc_pct(ratio)
        ax.text(int(0.32*max(x_vals) ), 0.1*max(cumsum) , f"inverted max dist.: {max_dist:.2f}\n inverted stdev: {std_val:.2e}\n inverted area ratio: {ratio:.2f}", fontsize = 'medium')
    else:      
        ax.text(int(0.4*max(x_vals) ), 0.3*max(cumsum) , f"max distance: {max_dist:.2f}\n stdev: {std_val:.2e}\n area ratio: {ratio:.2f}", fontsize = 'medium')
        
    ax.set_xlabel('Ranked Barcode')
    ax.set_ylabel('Scaled Cumulative Count')



    #return the standard deviation for secant lines 
    return [max_dist, std_val, ratio ]


#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------

#ambient gene plots
def plot_dropout(sample_dat, drop_cut_boolean, dropout_thresh = 2, ax = None, return_dat = False):
    """This function makes the drop-out over ranked barcode plot"""
    if (ax is None):
        _, ax = plt.subplots(figsize=(4, 4))
    
        
    recalc_boolean = not ( ('pct_counts_ambient'  in sample_dat.obs) and ("arcsinh_total_counts"  in sample_dat.obs ) )
    
    #print(recalc_boolean)
    
    if(drop_cut_boolean):
        if( recalc_boolean ):
            sample_dat.var["pct_dropout_by_counts"] = np.array( 
                (1 - (sample_dat.X.astype(bool).sum(axis=0)/ sample_dat.n_obs)) * 100 
            ).squeeze()
            qc_plt, amb_genes, sample_dat = func_qc.dropkick_curves_dropout_cutoff(sample_dat,dropout_thresh , ax = ax) 
            #set the threshold 
        else:
            #have already calculated metrics based on cut-off 
            qc_plt, amb_genes = func_qc.revized_dropout_plot(sample_dat, show=False, ax=ax)
            
    else:  # no dropout threshold
        if(recalc_boolean):
            qc_plt, amb_genes, sample_dat = func_qc.dropkick_curves(sample_dat, ax = ax) 
        else: 
            #directly plotting 
            qc_plt, amb_genes = func_qc.revized_dropout_plot(sample_dat, show=False, ax=ax)
            
    ax.axhline( y = dropout_thresh, xmin=0, xmax = 1,color = 'salmon' )
    
    if return_dat:
        return ax, amb_genes, sample_dat
    return ax


#plot 2 
def plot_pct_ambient(sample_dat, dat_plot_dropout = False, ax = None , return_dat = False):
    if(dat_plot_dropout == False):
        print("run plot_dropout() on the sample before this function")
        return 
    
    sample_dat.obs['pct_counts_ambient'] = sample_dat.obs['pct_counts_ambient'].replace(np.nan, 0)
    
    
    if (ax is None):
        _, ax = plt.subplots(figsize=(4, 4))
    
    ax1 = ax
        
    if(sum(sample_dat.obs['pct_counts_ambient']) == 0 ): #no need to plot if there's no ambient gene
        mean_pct = 0
        ax1.axvline(x = mean_pct, ymin=0, ymax = 1,color = 'salmon' )
        ax1.set_xlim(-0.1, 1)
        ax1.set_ylim(0, 2)
        ax1.set_xlabel("pct_counts_ambient")
        ax1.set_ylabel("Count")
        ax1.text(0.6, 0.3, f"avg = {mean_pct: .2f}", transform = ax1.transAxes, )
        print("no ambient genes to be plotted")
    
    else:
        mean_pct  = np.mean(sample_dat.obs['pct_counts_ambient'] )
        sn.histplot(sample_dat.obs['pct_counts_ambient'] , kde=True, color = 'grey', ax=ax1)
        ax1.axvline(x = mean_pct, ymin=0, ymax = 1,color = 'salmon' )
        ax1.text(0.6, 0.3, f"avg = {mean_pct: .2f}", transform = ax1.transAxes, )
        sample_dat.obs["ambient_count"] = sample_dat[:,sample_dat.var.ambient].X.sum(axis=1)
    
    if return_dat:
        return ax1, mean_pct
    return ax1





#plot3
def plot_total_count(sample_dat, ax = None, fill = True):
    if (ax is None):
        _, ax = plt.subplots(figsize=(4, 4))
        
    if("log_total_counts" not in sample_dat.obs):
        sample_dat = func_qc.add_log10_total_counts(sample_dat)
        
    ax2 = ax
    sn.histplot(sample_dat.obs["log_total_counts"], kde = True, color = 'grey', ax = ax2, fill = fill)
    ax2.set_xlabel("Reads per Barcode")
    ax2.set_xlim(2,6 )
    ax2.set_xticks(list(range(2,6,1)))
    ax2.set_xticklabels([f"10^{i}" for i in range(2,6,1)], fontsize = 8)
    ax2.set_ylabel("Number of Barcode", fontsize = 10)
    
    return ax2



#updated version of formatted figures 
#8/19/22
def formatted_figures(dat, save_amb_ls = None, save_fig = None, show_dat_name = None, mode = -1):
    fig = plt.figure(figsize = [15,20])
    ax1 = fig.add_subplot(421)
    ax2 = fig.add_subplot(422)
    ax3 = fig.add_subplot(423)
    ax4 = fig.add_subplot(424)
    ax5 = fig.add_subplot(425)
    ax6 = fig.add_subplot(426)
    ax7 = fig.add_subplot(427)


    ax1, cum_sum = plot_cum_sum(dat, scale = True, ax =ax1, return_data = True )
    if(show_dat_name):
        ax1.set_title(show_dat_name)
    ax2, ret, mean_end_grad = plot_slope( dat, cum_sum = cum_sum, ax = ax2, return_dat = True )
    print(f"mean_end_grad: {mean_end_grad}")
    ax3, ratio = plot_freq_weighted_slope(dat, mean_end_grad,  ax = ax3, ret = ret, return_dat = True, mode = mode)
    ax4, max_secant, std_val, cum_curve_area_ratio = plot_secant_line( dat, cum_sum = cum_sum, ax = ax4,  return_dat = True)
    ax5, amb_genes, dat = plot_dropout(dat, True, 2, ax = ax5, return_dat = True)
    ret6 = plot_pct_ambient(dat, dat_plot_dropout = True, ax = ax6 , return_dat = True)
    if( ret6 == None ):
        mean_pct = 0
    else: 
        ax6, mean_pct = ret6
    ax7 =plot_total_count(dat, ax = ax7)
    
    if(save_amb_ls and len(amb_genes) >0):
        np.savetxt(save_amb_ls,amb_genes, delimiter=',', fmt = "%s")
    #the 1st ratio is freq*slope area ratio over the entire square ratio
    if(save_fig):
        plt.savefig(save_fig)
    return [ratio, max_secant, std_val, cum_curve_area_ratio, len(amb_genes), mean_pct, dat]



def formatted_figures_one_column(dat, save_amb_ls = None, 
                      save_fig = None, show_dat_name = None, 
                      slope_freq_mode = 1):
    
    fig = plt.figure(figsize = [6,34])
    ax1 = fig.add_subplot(711)
    ax2 = fig.add_subplot(712)
    ax3 = fig.add_subplot(713)
    ax4 = fig.add_subplot(714)
    ax5 = fig.add_subplot(715)
    ax6 = fig.add_subplot(716)
    ax7 = fig.add_subplot(717)

    
    ax1, cum_sum = plot_cum_sum(dat, scale = True, ax =ax1, return_data = True )
    if(show_dat_name):
        ax1.set_title(show_dat_name)
    ax2, ret, mean_end_grad = plot_slope( dat, cum_sum = cum_sum, ax = ax2, return_dat = True, mode = slope_freq_mode)
    print(f"mean_end_grad: {mean_end_grad}")
    ax3, ratio = plot_freq_weighted_slope(dat, mean_end_grad,  ax = ax3, ret = ret, return_dat = True)
    print(f"slope freq high slope ratio: {ratio}")
    ax4, max_secant, std_val, cum_curve_area_ratio = plot_secant_line( dat, cum_sum = cum_sum, ax = ax4,  return_dat = True)
    ax5, amb_genes, dat = plot_dropout(dat, True, 2, ax = ax5, return_dat = True)
    ret6 = plot_pct_ambient(dat, dat_plot_dropout = True, ax = ax6 , return_dat = True)
    if( ret6 == None ):
        mean_pct = 0
    else: 
        ax6, mean_pct = ret6
    ax7 = plot_total_count(dat, ax = ax7)
    
    plt.tight_layout()
    plt.rcParams["axes.grid"] =False
    
    
    if(save_amb_ls and len(amb_genes) >0):
        np.savetxt(save_amb_ls,amb_genes, delimiter=',', fmt = "%s")
    #the 1st ratio is freq*slope area ratio over the entire square ratio
    if(save_fig):
        plt.savefig(save_fig)
    return [ratio, max_secant, std_val, cum_curve_area_ratio, len(amb_genes), mean_pct, dat]