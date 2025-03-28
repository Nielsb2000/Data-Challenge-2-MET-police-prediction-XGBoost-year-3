# JBG040-Group9
This repository contains the code for the TU/e course JBG050 Data Challenge 2. Please read this document carefully as it has been filled out with important information on how to run the code.

## Get Started
to get started working on this open a terminal, type in `cd <theDirectory/youWant/toWorkin>`, then only the first time you download the repo do:
```
git clone https://github.com/Miesjell/JBG040-Group9.git
cd JBG040-Group19/
conda create --name dbl2 python=3.10 
conda activate dbl2 
pip install -r requirements.txt
```
this downloads the repository, creates a new environment for this project and then installs all the dependencies. Now anytime you want to run anything related to this project before make sure you are in the correct environment by typing in the terminal `conda activate dbl2`, once activated navigate to the JBG050-Group9 folder with `cd /path/to/JBG050-Group9`. 

<!-- For pytorch_geometric, Pytorch 1.12.0 or above is required. 
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
-->


## Download data

Now to download the data go to `https://data.police.uk/data/`, click on Custom Dowload.

Then select the following:

![alt text](https://github.com/Miesjell/JBG050-Group9/blob/2662829140c471a3ae1f934c391a46b651818721/data.png)

Then go on `https://data.police.uk/data/archive/` by clicking on archive and download the following:
- `April 2020`
- `April 2017`

then copy the 3 zipfiles in the `data/raw/` folder of this repository.

This is data from December 2010 to March 2023, we will consider the Metropolitan area and 

<!-- Hertfordshire too? -->

Now to prepare the data in that terminal run:

```
python3 process_crimes.py
```
you can choose between wards and LSOA codes as aggregation level

this will save the resulting datasets in the `/data/` folder as `df_crimes_{aggregation level}.csv`

