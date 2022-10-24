import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt




def re_cluster(adata, p_barcode, patient_list = False):
    """this function redo the pca, neighboring and umap for a subset specified by the p_barcode(patient barcode)
    p_barcode can be a string of barcode or a list of strings of barcodes
    if p_barcode is a list, set patient_list should be set to True """
    
    if (patient_list):
        adata2 = adata[adata.obs["PatientBarcode"].isin( p_barcode ) == True]
    else: 
        adata2 = adata[adata.obs["PatientBarcode"] == p_barcode]
    print(adata2.n_obs)
    sc.tl.pca(adata2, return_info=False )
    sc.pp.neighbors( adata2, n_neighbors=int( np.sqrt(adata2.n_obs) ) , n_pcs=40)
    sc.tl.umap(adata2)
    
    return adata2



def normalization( dat_ct):
    """this function normalize the data so that each cell has the same 
    number of total counts as the median value of the total counts among all cells.
    The data will also be log-like transformed
    Count values will also be transformed to z-scores for each gene"""
    sc.pp.normalize_total(dat_ct) 
    dat_ct.X = np.arcsinh(dat_ct.X).copy()
    sc.pp.scale(dat_ct)
    
    return dat_ct


def scale_column( df, col_name, scaled_col_name):
    """This function scaled a column in a dataframe so that all values is between 0 and 1 
    the new dataframe will be returned """
    min_value = min( df[col_name])
    max_value = max(df[col_name])
    df[scaled_col_name] = (df[col_name] - min_value) / (max_value - min_value)
    
    return df




def save_h5ad_compresseed(dat_ls, name_ls, compression_opt = None):
    """This function save a list of anndata objs with names/paths specified in name_ls
    Files will be saved in compressed form, and the compression option is 'gzip for all files unless specified 
    The function will return the number of files saved successfully'"""
    
    if (not compression_opt):
        compression_opt = ['gzip' for i in range(len(dat_ls))]
    saved_num = 0
    for j in range(len(dat_ls)):
        try:
            dat_ls[j].write(name_ls[j], compression = compression_opt[j])
        except BaseException as err:
            print(err)
            continue
        else:
            saved_num +=1
        
    return saved_num



def clustering(dat, n_pcs = 50, n_neighbors= None):
    """This function calculate pcs, and get the umap of the dataset
    dat will be edited in place
    
    param dat: the anndata object that has transformed data
    param n_pcs: number of PCs to pass in to sc.tl.pca() function
    param n_neighbors: number of neightbors to pass in to the sc.pp.neighbors() function
    """
    
    if(n_neighbors == None):
        n_neighbors = int( np.sqrt(dat.n_obs) )
    sc.tl.pca(dat, return_info=False )
    sc.pp.neighbors( dat, n_neighbors= n_neighbors , n_pcs=n_pcs)
    sc.tl.umap(dat)
    
    return 



