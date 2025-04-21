# AmbiQuant: Ambient contamination quantification metrics
- Quantitative ambient contamination-based metrics to evaluate scRNA-seq data quality
- Check out our publication <font color = brown>**A contamination focused approach for optimizing the single-cell RNA-seq experiment**</font> [here](https://doi.org/10.1016/j.isci.2023.107242)


- The environment for this pipeline can be installed through the included .yml with the following, which will also initialize a jupyter kernel: 
```
conda env create -f qc_pipe.yml
python -m ipykernel install --user --name=qc_pipe
```
- this yml file is adapted from [QCPipe](https://github.com/Ken-Lau-Lab/STAR_Protocol.git) pipeline published in [this protocol](https://pubmed.ncbi.nlm.nih.gov/33982010/)


- Follow the steps in [ExamplePlots.ipynb](https://github.com/Ken-Lau-Lab/AmbientContaminationMetrics/blob/main/ExamplePlots.ipynb) to generate ambient contamination metrics plots
- example dataset in the notebook can be downloaded:
    + dropset option:
    <br> make sure you are at the right working directory
    <br>```mkdir 5394_YX_2``` (make a folder for the files)
    <br>```cd 5394_YX_2 ``` (change directory to the folder) 
    <br>```curl -O -J -L https://www.dropbox.com/sh/1z2nc7v3pp9o286/AACdWSa5uswk1pBLVn9yjhDna?dl=0 ``` (download as a zip file) 
    <br>```unzip 5394_YX_2.zip``` (unzip)
    + h5ad option:
    <br>```curl -O -J -L https://www.dropbox.com/s/s2h2t5uyd9ygud3/5394_YX_2_full.h5ad?dl=0```

