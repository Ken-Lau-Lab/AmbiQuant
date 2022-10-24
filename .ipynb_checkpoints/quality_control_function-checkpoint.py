
"""
Functions related to quality control 
"""

import scanpy as sc; sc.set_figure_params(color_map="viridis", frameon=False, figsize= [4,4])
import dropkick as dk
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn 
import anndata as ad
import glob


from sklearn.preprocessing import minmax_scale
from sklearn.metrics import auc as auc
from scipy.optimize import curve_fit
from scipy import stats


import data_processing as data_proc
sys.path.append("./QCPipe_dir/")
import QCPipe



#---------------------------------------------------------------------------------------------
#---------------------------Functions for Loading and Inspecting Data--------------------------
#----------------------------------------------------------------------------------------------


def read_dropest(dir_path, gene_file_identifier = "features", barcode_file_identifier = "barcodes", 
                 gene_file_delim = ',', barcode_file_delim = ',',
                 var_col = 0, obs_col = 0):
    """Read in dropset data from gene, barcode and count matrix files. The files should be all in the same directory specified by dir_path. This function is a modification of the read_dropset function from QCPipe.qc. 
   @param  dir_path: the pathway of the folder that contains the count matrix, gene and barcode files
   @param gene_file_identifier and barcode_file_identifier: strings contained in the file names of the gene and barcode file
   @param gene_file_delim and barcode_file_delim: delimiters used in the 2 files. Will pass to pandas.read_csv() function as the argument for delimiter parameter
   @param var_col and obs_col: when reading in data for adata's var and obs index, which column of the original file should be used. Default set to 0. Note that sometimes, the gene file has multiple columns such as ensembl ID, gene names and other info.Ensembl ID are in the first column in most cases
   
   @return: the anndata object for the dataset """
    
    data_matrix = glob.glob(dir_path+"/*.mtx")[0]
    print(dir_path+ f"/*{gene_file_identifier}.*")
    data_genes = glob.glob(dir_path+ f"/*{gene_file_identifier}.*")[0]
    data_barcodes = glob.glob(dir_path+ f"/*{barcode_file_identifier}.*")[0]
    adata = sc.read_mtx(data_matrix).T
    adata.var.index = pd.read_csv(data_genes,header=None, delimiter = gene_file_delim)[var_col].values
    adata.obs.index = pd.read_csv(data_barcodes,header=None, delimiter = barcode_file_delim)[obs_col].values
    adata.obs.index.name = 'Cells'
    adata.var.index.name = 'Genes'
    adata = QCPipe.qc.reorder_AnnData(adata,descending=True)
    adata.raw = adata
    
    return(adata)





def inspect_data( data_path, save_name, cut_off = None, 
                 gene_file_identifier = "features", barcode_file_identifier = "barcodes", 
                 gene_file_delim = ',', barcode_file_delim = ',',
                 var_col = 0, obs_col = 0):
    """this function reads in count matrices, gene and barcode files and combines them to an h5ad file and visualize the cumulative transcripts count curve and inflection points using QCPipe's qc.find_inflection() function.
    
    @param data_path: the directory that stores the raw data files
    @param savename: pathway and file name to store the h5ad file, set to None if no need to save the h5ad
    @param cut_off: set to an int to specifcy how many top ranked barcodes you want to keep, default set to None which will not cut-off the dataset 
    @param gene_file_identifier, barcode_file_identifier, gene_file_delim, barcode_file_delim: parameters to specify how to read in the barcode and gene_name file. See read_dropset for detailed description 
    
    @return: return the anndata object 
    """
    
     # combine counts, gene and barcode info to one object
    samp = read_dropest( data_path, gene_file_identifier = gene_file_identifier, 
                        barcode_file_identifier = barcode_file_identifier, 
                        gene_file_delim = gene_file_delim, barcode_file_delim = barcode_file_delim,
                        var_col = var_col, obs_col = obs_col )
    
    if(cut_off):

        if samp.n_obs > cut_off:
            samp = samp[0:cut_off]
    
    #make inflection point plot
    samp_inflection = QCPipe.qc.find_inflection(samp)
    
    #save the file in curr directory
    if(save_name):
        samp.write(save_name, compression = 'gzip')
    
    return samp





def find_inflection_numerics(adata_in):
    """ This function calculates the inflection points of an anndata obj based on its 'total_counts' field without output plots. Modified from QCPipe.qc's function. 
    @param adata_in: the anndata obj whose value will be calculated
    @return 1st, 2nd and 3rd inflection points indexed by ranked barcode 
    """
    
    data_cumsum = np.cumsum(adata_in.obs['total_counts'])
    x_vals=np.arange(0,adata_in.n_obs)
    secant_coef=data_cumsum[adata_in.obs.shape[0]-1]/adata_in.obs.shape[0]
    secant_line=secant_coef*x_vals
    secant_dist=abs(data_cumsum-secant_line)
    inflection_points = secant_dist.argsort()[::-1]
    inflection_percentiles = [0,15,30]
    inflection_percentiles_inds = np.percentile( x_vals[x_vals>inflection_points[0]],inflection_percentiles).astype(int)
    
    return inflection_percentiles_inds





#---------------------------------------------------------------------------------------------
#---------------------------Functions for cutting off the dataset--------------------------
#----------------------------------------------------------------------------------------------


def cut_off_from_dropset( data_path, 
                         gene_file_identifier = "features", barcode_file_identifier = "barcodes",
                         gene_file_delim = ',', barcode_file_delim = ',',
                         var_col = 0, obs_col = 0, 
                         inflection_fold = 4, est_real_cell= None, max_cell = 10000, 
                         mito_tag = "MT-",run_qc=True, print_info = True):
    
    """This function read-in count matrices, gene and barcode files and combines them to an h5ad. Cut-off the h5ad object based on a multiple of the first inflection point or an estimated real cell number passed in (use the max between the 2 quantities). Will keep cells no more than the max_cell despite of other parameter
    
    @param data_path: path to the folder containing dropset matricies and data
    @param gene_file_identifier, barcode_file_identifier, gene_file_delim, barcode_file_delim, var_col, and obs_col are parameters set to read in data from a dropset folder. See read_dropset for a more detailed description. 
    @param inflection_fold: how many times of the inflection point you want to use as the cut-off. Default to be 4
    @param est_real_cell: the user can manually enter an estimated real cell number that used to calculate number of cells to keep (warning message will be printed if estimated real cell larger than the inflection point)
    @param max_cell: maximal number of cells to keep despite of other parameters
    @param mito_tag: "MT-" for human or "mt-" for mouse, or other string specifies mitochondrial gene names 
    @param print_info: set to True if you want to know the number of cells in the original and the new dataset
    return the final anndata object after cut-off
    """
    
    samp = read_dropest( data_path,  gene_file_identifier = gene_file_identifier, 
                        barcode_file_identifier = barcode_file_identifier, 
                        gene_file_delim = gene_file_delim, barcode_file_delim = barcode_file_delim,
                        var_col = var_col, obs_col = obs_col) # combine counts, gene and barcode info to one object
    
    samp = cut_off_h5ad(samp, inflection_fold , est_real_cell , max_cell ,  mito_tag=mito_tag, run_qc=run_qc , print_info= print_info)
    
    return samp




    
    
    
    
#this function was originally written in 1_example_dataset notebook
#updated 6/10/22
def cut_off_h5ad(sample, inflection_fold = 4, est_real_cell = None, max_cell = 10000, 
                 qc_pipe_inflection = True, mito_tag = "MT-",run_qc=True, 
                 print_info = True):
    """
    This function cut-off a given h5ad annData object based on a multiple of the first inflection point or a multiple of an estimated real cell number passed in. 
    Will keep cells no more than the max_cell despite of other parameter
    
    @param sample: the annData object read from a h5ad file
    @param inflection_fold: how many times of the inflection point you want to use as the cut-off. Default to be 4
    @param est_real_cell: the user can manually enter an estimated real cell number that used to calculate number of cells to keep. Warning message will be printed if estimated real cell larger than the inflection point 
    @param qc_pipe_inflection: set True if want to use the QCPipe.qc's inflection point function, which will make plots and can run qc based on run_qc flag. Otherwise, only inflection points will be computed, but no other features. Set False for faster computation
    @param print_info: set to True if you want to know the number of cells in the original and the new dataset
    
    @return sample: the new annData object that has been cut
    """
    
    #pre-processing the raw h5ad
    
    sample.raw = sample
    sample.obs_names_make_unique()
    sample = QCPipe.qc.reorder_AnnData(sample)
    
    #get the inflection point
    if (qc_pipe_inflection):
        samp_inflection = QCPipe.qc.find_inflection(sample, mito_tag = mito_tag ,run_qc=run_qc) 
    else:
        samp_inflection = find_inflection_numerics(sample)
        
    #store the inflection point in the anndata obj
    sample.uns["inflection1"] = samp_inflection[0] 
    sample.uns["inflection_fold"] = inflection_fold
    
    cut_off_value = 0 #declare the cut off value 
    
    if(est_real_cell ):
        #if estimate real cell value entered, print warning if potentially over-estimated 
        if(est_real_cell > samp_inflection[0]):
            print(f"\nWarning: estimated real cell > first inflection point.\nRecommend to set est_real_cell lower than the inflection value of {samp_inflection[0]}\n")
        #set the cut-off at folds of est_real cell
        cut_off_value = inflection_fold * est_real_cell
    
    else:
        #if no est_real_cell entered
        cut_off_value = inflection_fold * samp_inflection[0] 
        
    #check if the cut-off value is larger than maximum cell number (default 10,000)
    #and set the cut off at the smaller value between the 2 
    cut_off_value = min(max_cell, cut_off_value)
    
    #make sure to cut-off only when the original sample has more cells than num_cell_to_keep
    sample2 = sample.copy()
    if sample.n_obs > cut_off_value:
        sample2 = sample2[0:cut_off_value]
    if(print_info):
        print(f"the original dataset has {sample.n_obs} cells. Cut to {sample2.n_obs} cells")
    
    return sample2






def cut_off_at_ranked_barcode_position( sample_dat, position):
    """Cut-off the ranked dataset by index/position of ranked barcodes. Barcodes are ranked descendingly on total gene counts.
    @param sample_dat: the anndata object of the dataset
    @param position: how many top-ranked barcode do you want to keep
    
    @return: the modified anndata object"""
    
    sample_dat.raw = sample_dat
    sample_dat = QCPipe.qc.reorder_AnnData(sample_dat)
    sample_dat = sample_dat[0:position]
    
    return sample_dat
    
#---------------------------------------------------------------------------------------------------------------
#------------------------------------ Data processing ---------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

#from 4_2_fit_gaussian_2 notebook
#updated 6/10/11

def add_log10_total_counts(sample):
    """This function add a new obs column, log10 transformed total counts for each droplet, to the anndata object passed in as sample
    @param sample: the anndata object 
    @return sample: the object with the new column, the original object should be modified in place as well"""
    
    sample.obs["log_total_counts"] = np.log10(sample.obs["total_counts"])
    return sample






#---------------------------------------------------------------------------------------------------------------
#------------------------------------ Metrics Calculation and Plots---------------------------------------
#------------------------------------------------------------------------------------------------------------


#TODO: check with method used 
def area_ratio_sample( sample_dat):
    """This function calculates the ratio between 
    areas under the cumulative curve and area of the minimal rectangle circumscribing the curve (its area = num of cell * total sum of read counts)
    
    @param sample_dat: the anndata object of the dataset
    @return: the area ratio value 
    """
    
    cumsum = np.cumsum(sample_dat.obs["total_counts"])
    x_val = np.arange(0, sample_dat.n_obs)
    s_auc = auc(x_val, cumsum) #auc for the sample curve
    
    tt_area = sample_dat.n_obs * max(cumsum) #the rectangle area
    area_ratio = s_auc / tt_area
    
    return area_ratio




#TODO: delete this part if not used else where
# def area_ratio_line( x, y):
#     """This function calculate the area under the curve y formed with the x-axis. It then calculates the ratio between the auc and the minimal rectangle formed with the max y and x-axis. Used to calculate the area for under the scaled slope sum curve
#     @param x and y: x and y coordinates of the curve whose area will be calculated 
#     @return: the area ratio between auc and the min-rectangle circumscribing it """
    
#     y_auc = auc(x, y)
#     rectangle = (max(x) - min(x)) * max(y)
#     area_ratio = y_auc / rectangle
    
#     return area_ratio



#TODO: check if the plotting script use this function's plot 
def calc_secant_lines(samp, cumsum = None,  ax = None):
    """ This function calculates the secant lines for the cumulative count curve for ranked barcode given a sample.
     A plot can also be made to show the secant lines and the standard deviation of the lines. 
     The ratio between the area under the curve and the area of the smallest rectangle that can fit the curve in will also be shown on the plot. 
    samp: the scRNA count sample 
    cumsum: can be pass-in if already calculated as a list, otherwise can be calculated from the sample data """
    
        
    #get cumulative sum from the data or by pass-in param
    if( cumsum is None):
        cumsum =  np.cumsum(samp.obs['total_counts'])

    x_vals=np.arange(0,samp.n_obs) #unit x values 

    #calculate secant line length
    secant_coef=cumsum[samp.obs.shape[0]-1]/samp.obs.shape[0]

    secant_line=secant_coef*x_vals
    secant_dist=abs(cumsum-secant_line)
    std_val = np.std(secant_dist)
    ratio = area_ratio_sample(samp)
    max_dist = max(secant_dist)
    max_ind = np.argmax(secant_dist)

    #plot the secant lines
    if (ax is None):
        _, ax = plt.subplots(figsize=(4, 4))
    ax.plot(x_vals, secant_line)
    ax.plot(x_vals, cumsum)
    vline_range = range(0, samp.n_obs, 200)
    ax.vlines(x_vals[vline_range], secant_line[vline_range],cumsum[vline_range], colors="lightgrey" )
    ax.vlines(x_vals[max_ind], secant_line[max_ind], cumsum[max_ind], colors = 'green')
    ax.text(int(0.6*max(x_vals) ), 0.4*max(cumsum) , f"max distance: {max_dist:.2f}\n stdev: {std_val:.2e}\n area ratio: {ratio:.2f}", fontsize = 'medium')
    ax.set_xlabel('Ranked Barcode')
    ax.set_ylabel('Scaled Cumulative Count')



    #return the standard deviation for secant lines 
    return [max_dist, std_val, ratio ] 






#TODO: checking all functions can input the dropout_threshold starting from the formatted figure function
def dropkick_curves_dropout_cutoff(s2, dropout_thresh = 2, ax= None):
    """This is a helper function called when plotting the ambient gene curves .
    It makes the dropkick ambiant gene plot when there's a dropout rate specified
    
    return: axes of dropout plot, a list of ambient genes and the anndata object with calculated metrics
    """
    s2.raw = s2
    num_ambient = len( s2.var["pct_dropout_by_counts"][ s2.var["pct_dropout_by_counts"]<=dropout_thresh]) 
    s2 = dk.recipe_dropkick(s2, filter=False, n_hvgs=None, X_final="raw_counts", n_ambient= num_ambient)
    qc_plt, amb_genes = revized_dropout_plot(s2, show=False, ax=ax)
    
    return qc_plt, amb_genes, s2

#copied 5/17/22 from 2_metrics notebook
def dropkick_curves(s2, num_ambient = 10, ax = None):
    """This is a helper function called by plot_distribution
    It makes the dropkick ambient gene plot when there's no specified dropout rate.
    The default ambient gene selection is based on its dropout rate ranking, where the top 10 lowest rate genes are selected.
    
    return: axes of dropout plot, a list of ambient genes and the anndata object with calculated metrics"""
    s2.raw = s2
    s2 = dk.recipe_dropkick(s2, filter=False, n_hvgs=None, X_final="raw_counts", n_ambient = num_ambient)
    #qc_plt = dk.qc_summary(s2)
    qc_plt, amb_genes = revized_dropout_plot(s2, show=False, ax=ax)
    
    return qc_plt, amb_genes, s2




def plot_distribution_v2(sample, drop_cutoff = True, dropout_thresh = 2):
    """This function make the following plots: 
     pct_count_ambiant distribution, read per barcode distribution and dropkick ambiant gene plots.
     @param sample: a anndata object for the sample
     @param drop_cutoff: if set True, ambiant genes will not be determined by its dropout rate ranking, but rather an absolute value specified in dropout_thresh
     @param dropout_thresh: the dropout percentage we want to use to select ambiant genes. default set to 2 percent """
    
    fig = plt.figure(figsize = [20,5]) #TODO: adjust figure sizes and ax2's xlim
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    fig2 = plt.figure(figsize = [6,4])
    ax3 = fig.add_subplot(111)
    
    recalc_boolean = not ( ('pct_counts_ambient'  in sample.obs) and ("arcsinh_total_counts"  in sample.obs ) ) #if the sample does not have fields, need to run dropkick_curves to get them first, so this boolean will be True 
    
    #if we want to cut off droprate based on an absolute value
    if(drop_cutoff):
        if( recalc_boolean ):
            sample.var["pct_dropout_by_counts"] = np.array( 
                (1 - (sample.X.astype(bool).sum(axis=0)/ sample.n_obs)) * 100 
            ).squeeze()
            qc_plt, amb_genes, sample = dropkick_curves_dropout_cutoff(sample,dropout_thresh , ax = ax3) #set the threshold 
        else:
            #have already calculated metrics based on cut-off 
            qc_plt, amb_genes = revized_dropout_plot(sample, show=False, ax=ax3)
            
    else:  
        if(recalc_boolean):
            qc_plt, amb_genes, sample = dropkick_curves(sample, ax = ax3) 
        else: 
            #directly plotting 
            qc_plt, amb_genes = revized_dropout_plot(sample, show=False, ax=ax3)
        
    if(sum(sample.obs['pct_counts_ambient']) != 0 ): #no need to plot if there's no ambient gene
        mean_pct  = np.mean(sample.obs['pct_counts_ambient'] )
        sn.histplot(sample.obs['pct_counts_ambient'] , kde=True, color = 'grey', ax=ax1)
        ax1.axvline(x = mean_pct, ymin=0, ymax = 1,color = 'salmon' )
        ax1.text(0.7, 0.3, f"avg = {mean_pct: .2f}", transform = ax1.transAxes, )
        sample.obs["ambient_count"] = sample[:,sample.var.ambient].X.sum(axis=1)
        sample.obs["ambient_count"] = sample[:,sample.var.ambient].X.sum(axis=1)
    
    sn.histplot(sample.obs["arcsinh_total_counts"], kde = True, color = 'grey', ax = ax2)
    ax2.set_xlabel("reads per barcode")
    ax2.set_xlim(2, 12)
    ax2.set_xticks(list(range(2,13,2)))
    ax2.set_xticklabels([f"10^{i}" for i in range(2,13,2)], fontsize = 8)
    ax2.set_ylabel("# of barcodes")

    plt.show()
    return amb_genes




def revized_dropout_plot(adata, show=False, ax=None):
    """
    Plots dropout rates for all genes
    the revised versoin plot out the number of ambient genes instead of the gene names

    Parameters
    ----------

    adata : anndata.AnnData
        object containing unfiltered scRNA-seq data
    show : bool, optional (default=False)
        show plot or return object
    ax : matplotlib.axes.Axes, optional (default=None)
        axes object for plotting. if None, create new.

    Returns
    -------

    ax : matplotlib.axes.Axes
        plot of gene dropout rates. return Axes object only if `show`==False, 
        otherwise output plot.
    genes: a list of gene names identified as ambient 
    """
    if not ax:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.plot(
        np.arange(adata.n_vars)+1,
        adata.var.pct_dropout_by_counts[
            np.argsort(adata.var.pct_dropout_by_counts)
        ].values,
        color="gray",
        linewidth=1,
    )
    # get range of values for positioning text
    val_max = adata.var.pct_dropout_by_counts.max()
    val_range = val_max - adata.var.pct_dropout_by_counts.min()
    
    ax.text(
        x=1,
        y=(val_max - (0.08 * val_range)),
        s=" Number of Ambient\n Genes:",
        fontweight="bold",
        fontsize=10,
    )
    
    # plot all ambient gene names if they'll fit
    ax.text(
                x=1,
                y=((val_max - (0.10 * val_range)) - ((0.05 * val_range))),
                s= adata.var.ambient.sum(),
                fontsize=12,
        
    
            )
        
    ax.set_xscale("log")
    ax.set_ylabel("Dropout Rate (%)", fontsize=12)
    ax.set_xlabel("Ranked Genes", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)
    
    genes = [adata.var_names[adata.var.ambient][x] for x in range(adata.var.ambient.sum())]
    if not show:
        return ax, genes
    else:
        return genes

#---------------------------------------------------------------------------------------------------------------
#------------------------------------ SIMULATION RELATED--------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

#copied 5/22/22 from 3_simulation
def multiply_data( folder_path, data_fold =2, inflection_fold = 5, 
                  sample_AnnData = None,
                  initial_keep = 12000, mito_tag = 'mt-'):
    """"this function will read in RNA seq data and multiply the dataset by a fold number specified in the pass-in parameter
    this function will also find the inflection point of the data's cumulative count matrix 
    and multiply the point's value with the inflection_fold value. 
    This product value will be the number of cells to keep after reading from the raw count data
    
    param folder_path: the path to the folder that has gene names, barcodes and count matrix
    param data_fold: how many more times of real cell do we want for the new dataset
    param inflection_fold: how many more times of cells we want to keep as a multiple for the first inflection point 
    param initial_keep: number of cells to keep before cut-off and concatenating the multiples of the dataset to prevent large memory usage 
    param mito_tag: monuse sample should use 'mt-', human "MT-"
    
    return the multiplied dataset as an AnnData obj"""
    
    #read in data
    if(sample_AnnData):
        s1 = sample_AnnData
    else:
        s1 = QCPipe.qc.read_dropest(folder_path)
    
    #keep only the the top 12000 cells / other number specified by initial_keep 
    if s1.n_obs > initial_keep:
        s1 = s1[0:initial_keep]
    
    #multiply the data matrix
    X2 = s1.to_df()
    barcodes = pd.Series( s1.to_df().index )
    barcodes_new = pd.Series( s1.to_df().index )
    for i in range(data_fold-1):
        X2 = pd.concat([X2, s1.to_df()], axis = 0, ignore_index= True)
        barcodes2 = barcodes + f"_{i+1}"
        barcodes_new = barcodes_new.append(barcodes2)
    X2.index = barcodes_new
    
    #creating the AnnData obj for the multiplied dataset
    dat2 = ad.AnnData(X2)
    dat2.raw = dat2
    dat2 = QCPipe.qc.reorder_AnnData(dat2, descending= True)
    
    #find inflection points
    inflection = QCPipe.qc.find_inflection(dat2,mito_tag = mito_tag,run_qc=True)
    dat2 = dat2[0:inflection_fold*inflection[0]]
    dat2.raw = dat2
    
    return dat2
    

    
#---------------------------------------------------------------------------------------------------------------
#------------------------------------ Gaussian Formula Functions------------------------------------------------
#------------------------------------------------------------------------------------------------------------    
#constant strings that can be called to print out the format of each function's parameter
gauss_bimodal_2m_params = "[m1,  m2, s1, s2, w, A]"
gauss_bimodal_params = "[ m, s, w, delta, A]"
gauss_bimodal2_params = "[m, s1, s2, w, delta, A]"
gauss_uni_param = "[m, sigma, A]"



#updated 6/13/22 from 4_3_est_2nd_gauss_peak notebook
def gauss_bimodal_2m (x, m1, m2, s1, s2, w, A):
    """This function is used to estimate bimodal gussian with 2 means and 2 stdevs """
    return w* np.exp(-(x-m1/2)**2 / (2. * s1**2)) / np.sqrt(2. * np.pi * s1**2) + \
               (A - w) *np.exp(-(x-m2/2)**2 / (2. * s2**2)) / np.sqrt(2. * np.pi * s2**2) 


def gauss_bimodal( x, m, s, w, delta, A):
    """This function is used to estimate bimodal gaussians with 1 mean estimator and 1 stdev value for each peak"""
    return w* np.exp(-(x-m+delta/2)**2 / (2. * s**2)) / np.sqrt(2. * np.pi * s**2) + \
               (A-w) *np.exp(-(x-m-delta/2)**2 / (2. * s**2)) / np.sqrt(2. * np.pi * s**2) 


def gauss_bimodal2 (x, m, s1, s2, w, delta,A):
    """This function is used to estimate bimodal gussian with 1 means and 2 stdevs """
    return w* np.exp(-(x-m+delta/2)**2 / (2. * s1**2)) / np.sqrt(2. * np.pi * s1**2) + \
              (A-w)* np.exp(-(x-m-delta/2)**2 / (2. * s2**2)) / np.sqrt(2. * np.pi * s2**2) 

def gauss_uni(x, m, sigma, A):
    """The unimodal guassian function with one mean, one stdev and a constant for peak height"""
    return A*np.exp(-(x-m)**2/(2*sigma**2))/np.sqrt(2. * np.pi * sigma**2) 
  

#---------------------------------------------------------------------------------------------------------------
#------------------------------------ Gaussian Curve Fitting, Plotting and Areas ------------------------
#------------------------------------------------------------------------------------------------------------  
FOLD_STD_TO_COVER_PEAK = 3

#from 4_3_est_2nd_gauss_peak notebook
#updated 6/13/22

def get_gauss_param(data_points, pdf = gauss_bimodal, x = np.arange(4,15,0.005), bounds = None):
    """This function git gaussian distribution from the datapoints' distribution and returns the parameters learned for the passed in gaussian function. The resulting parameters will predict a curve with a total area under curve of 1 as kernel density estimation was used
    
    param data_points: a list of values whose distribution is intended to fit to a given gaussian function
        eg. sample.obs["arcsinh_total_counts"]
    param pdf: gaussian formula function to be used, default set to gauss_bimodal 
    param x: a list of x values used to evaluate a kernel density distribution
    param bound: a tuple of lists in the format of ([gauss_param lower bounds], [gauss_param upper bounds])
    
    return parameters: parameters for the gauss function fit from the data points"""
    
    kernel = stats.gaussian_kde(data_points )
    den = kernel.evaluate(x)

    if(bounds):
        parameters, covariance = curve_fit(pdf, x, den, bounds=bounds)
    else:
        parameters, covariance = curve_fit(pdf, x, den)
    
    
    return parameters



def get_kernel_line_x( sample_dat, kernel_line):
    """This function returns a list of x values for the passed-in kernel line, where the kernel line represents a list of y values for a gaussian kernel fit from the sample_dat, and the returning value will be a list of x values matching the y values and the sample_dat"""
    ret_x= np.linspace( min(sample_dat), max(sample_dat), num =  len(kernel_line))
    return ret_x



#from 4_3_est_2nd_gauss_peak notebook
#updated 6/13/22
def get_gaussian_from_kernel_line(kernel_line, x, pdf, bounds = None):
    """This function fits gaussian curve from a given kernel line and its x value
    param kernel_line: y values of a gaussian kernel 
    param x: the x positions for each y value, same dimention with kernel_line
    param pdf: the gaussian formula function intended to fit the data on
    param bounds: a tuple of lists in the format of ([gauss_param lower bounds], [gauss_param upper bounds])
    
    return parameters: parameters for the gauss function fit from the data points"""
    
    if(bounds):
        parameters, covariance = curve_fit(pdf, x, kernel_line, bounds=bounds)
    else:
        parameters, covariance = curve_fit(pdf, x, kernel_line)
        
    return parameters




#from 4_3_est_2nd_gauss_peak notebook
#updated 6/13/22

def predict_gauss( param,  samp_dat, func, x2 = np.arange(5,15,0.01), plot = False, plot_hist = False, n_bins = 500):
    """This function returns a list of outputs from the func function given a list of x values and parameters for the function func. Plots can be made tp visualize the prediction line 
    
    param param: the parameters intended to use as input to the function func
    param samp_dat: values (eg. dat.obs["log_total_counts"]) to be visualized. 
    param func: the gaussian function used to do predictions
    param x2: a list of x values whose y values will be predicted from func and the passed-in parameters
    param plot: set True if want to visualize the predicted line
    param plot_hist: set True if want to visualize the original histogram from the data
    param n_bins: number of bins used for the histogram 
    
    return the predicted list of y values """

    y2 =  func(x2, *param) 
    
    if(plot):
        ax = plt.subplot()
        plt.plot(x2,y2, color = 'gold', label = "predicted")
        ax.set_xlabel("log reads per barcode")
        ax.set_title("Predicted Probability Distribution")
        if(plot_hist):
            sn.histplot(samp_dat, ax = ax, bins = n_bins, label = "data", color = 'salmon')
            ax.set_title("Predicted Probability and Real Data Distribution")
        plt.legend() 

    return y2






#from 4_3_est_2nd_gauss_peak notebook
#updated 6/13/22
def get_mean_std_w( fit_func, param):
    
    if(fit_func == gauss_bimodal):
        #m, s, w, delta, A
        m1 = param[0]+ param[3]/2
        m2 = param[0] - param[3]/2
        s1 = s2 = abs( param[1] )
        w1 = param[2]
        w2 = param[4]-w1
    
    elif(fit_func == gauss_bimodal2):
        # m, s1, s2, w, delta, A
        m1 = param[0]+ param[4]/-2
        m2 = param[0] - param[4]/-2
        s1 = abs( param[1] )
        s2 = abs( param[2])
        w1 = param[3]
        w2 = param[5]-w1
    elif(fit_func == gauss_bimodal_2m):
        # 0:m1, 1:m2, 2:s1, 3:s2, 4:w, 5:A,
        
        m1 = param[0]/2
        m2 = param[1]/2
        s1 = abs(param[2])
        s2 = abs(param[3])
        w1 =  param[4]
        w2 = ( param[5] - param[4] )
        
    return [m1,m2, s1,s2,w1,w2]



#from 4_3_est_2nd_gauss_peak notebook
#updated 6/13/22
def get_1st_peak_mean_std(sample, param, func, func_input_param):
    """this function find the mean and std values that is supposed to be the 1st peak of the gaussian curve
    param sample: the anndata obj for the sample
    param param: parameter learned from guassian fitting AND in the format of [m1,m2,s1,s2,w1,w2]
        this param list should be the returned list from get_mea_std_w function
    param func: the gaussian equation function used to fit gaussian
    param func_input_param: the parameters output from gaussian fitting, specific to each gaussian formula
    return [m1,s1]: the fist peak mean and std, or None if none found """

    xmin = min(sample.obs["log_total_counts"])
    xmax = max(sample.obs["log_total_counts"])
    ms = param[0:2]
    ss = param[2:4]
    
    peak1_x = []
    chosen_m = None
    
    for m in ms:
        #append mean within range to the peak1 list 
        if(m >= xmin and m <= xmax):
            peak1_x.append(m)
    
    #if both values are within range, chose the one with higer y value 
    if(len(peak1_x)>1):
        y1 = func(peak1_x[0], *func_input_param)
        y2 = func(peak1_x[1], *func_input_param)
        #print("both within range")
        if(y1>y2):
            chosen_m = peak1_x[0]
        else:
            chosen_m = peak1_x[1]
    elif len( peak1_x) == 1:
        chosen_m = peak1_x[0]
        #print("one within range")
    else:
        print("no value within range found")
        return
    
    if(chosen_m == ms[0]):
        print(f"from get_1st_peak_mean_std: m1, s1 = {param[0]:.3f}, {param[2]:.3f}")
        return [param[0], param[2]] #m1, s1
    else:
        print(f"from get_1st_peak_mean_std: m1, s1 = {param[1]:.3f}, {param[3]:.3f}")
        return [param[1], param[3]] #m2,s2
    

    
#from 4_3_est_2nd_gauss_peak notebook
#updated 6/13/22
    
#this function calulates the difference between histogram bars and curves at each midpoint of the histgram bars, 
#and there is another version of using n_bins number of points 
def get_bars_areas_and_range(samp_values,  m, s, func, func_input_param, n_bins):
    
    
    #get each bar's height and position 
    n, b,p =plt.hist(samp_values, bins = n_bins)
    sn.histplot(samp_values, kde = False, bins = n_bins, color = 'lightgrey')
    #calculate a midpoin tfor each bar, should be same len with n ( and 1 less element than b)
    bar_midpoints = [0.5*( b[i] + b[i+1]) for i in range(len(b)-1)]
    
    #get the range for the 1st gaussian peak
    #mean +/- 2* std but within the bin ranges 
    p1 = max(m-FOLD_STD_TO_COVER_PEAK*s, b[0]) #peak lower bound
    p2 = min( m+FOLD_STD_TO_COVER_PEAK*s, b[-1]) #peak upper bound 
    
    #calc the sum of bar area higher than the gaussian curve
    higher_area = []
    total_area = []
    for i in range(len(bar_midpoints)):
        
        total_area.append((b[i+1] - b[i]) * ( n[i]) )
        #bin lower tick should > p1, bin upper tick should <p2 
        if(bar_midpoints[i] < p1 or bar_midpoints[i]> p2):
            higher_area.append(0)
            continue
        
        else: #within the 1st peak range
            #get the curve height
            curve_value = func(bar_midpoints[i], *func_input_param)
            
            
            extra_area = (b[i+1] - b[i]) * ( n[i] - curve_value) #width*height = rectangle area
            higher_area.append(extra_area)
    
    print(f"from get_bars_area: p1, p2 = {p1:.2f}, {p2:.2f}")
    return [p1,p2, total_area, higher_area]


#from 4_3_est_2nd_gauss_peak notebook
#updated 6/13/22
def cal_real_cell_area(sample,gaussian_func, learned_param, n_bins = 500 ):
    """ returns a list [ total area, extra_areas, peak_aea, real_cell]"""
    
    #set up plot
    fig = plt.figure(figsize = [4,4])
    ax = fig.add_subplot()
    
    formatted_param = get_mean_std_w( gaussian_func, learned_param)
    [m1,std1] = get_1st_peak_mean_std(sample, formatted_param, gaussian_func, learned_param) 

    p1,p2, total_areas, extra_areas = get_bars_areas_and_range(
        sample.obs["log_total_counts"],  m1, std1, gaussian_func, learned_param, n_bins)
    #y_pred_peak1 also from the function so that the plotted predicted y values match the ones 
    #we used to cauculate the difference between predicted curve and the histogram bard
    
    x_peak1 = np.linspace(p1,p2,n_bins) 
    y_peak1 = predict_gauss(learned_param,sample, gaussian_func,x2 = x_peak1, n_bins= n_bins)
    peak_area = auc(x_peak1, y_peak1 )
    ax.plot(x_peak1, y_peak1, label = "curve_to_calc", color = "green")
    
    vline_closest_x = min(x_peak1, key=lambda x:abs(x-m1))
    ax.vlines( x = m1, ymin = 0, ymax = y_peak1[np.argwhere(x_peak1 == vline_closest_x).squeeze()], colors = ["green"])
    ax.text(0.6, 0.65, f'm1 = {m1:.3f}', transform=ax.transAxes)
    
    
    real_cell_area = sum(total_areas) - sum(extra_areas) - peak_area
    
    print(f"real cell area = total( {sum(total_areas): .2f}) - peak( {peak_area:.2f}) - higher_bar( {sum(extra_areas):.2f}) = {real_cell_area:.2f}")
    
    #print("\n ploted_y values: ")
    #print(y_pred_peak1)
    return [sum(total_areas) ,sum(extra_areas) ,peak_area, real_cell_area] 






#----------------------------other functions----------------------------------------------------------------

def check_transpose( dat, test_gene_name = 'Muc2'):
    if( test_gene_name not in dat.var.index):
        dat = dat.T
    print(f"obs: {dat.obs.index[0]}")
    return dat
            
            
    
    
    
        
        
        