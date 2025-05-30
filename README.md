# Taming the heterogeneous dynamics in ocean chlorophyll-a concentration prediction with a deep learning model
This repository implements STD-Hunter method, a novel machine learning approach designed to facilitate long-term, large-scale prediction of chlorophyll-a (Chl_a) concentration. The core innovation is that STD-Hunter forms a predictive model for each task by integrating commonly shared basis models with the specific spatiotemporal characteristics of Chl_a.  STD-Hunter strikes an optimal balance between effectively leveraging data and preserving idiosyncratic dynamics with an interpretable mechanism, offering a trustworthy tool for heterogeneous data exploration.
![](https://github.com/fazhangmath/STD-Hunter/blob/main/Framework.png)

## Data
The chlorophyll-a (Chl_a) concentration and sea surface temperature (SST) data is publicly available in [MODIS Aqua projects](https://search.earthdata.nasa.gov/search?q=10.5067/AQUA/MODIS/L3M). We also provide the used data in [MODIS data](https://hkustconnect-my.sharepoint.com/:f:/g/personal/fzhangat_connect_ust_hk/EsembQtGI_5HlT6MeA2n89wBvqHtvkwkg7TlBJw-An9rmw?e=6S7VTM).

## Reference
If you find the ```STD-Hunter``` package or any of the source code in this repository useful for your work, please cite:
> Taming the heterogeneous dynamics in ocean chlorophyll-a concentration prediction with a deep learning model.
> 
> Fa Zhang, Hiusuet Kung, Fan Zhang, Zhiwei Wang, Can Yang#, and Jianping Gan#. 2025.

## Development
The python repository ```STD-Hunter``` is developed and maintained by Fa Zhang.

## Contact
Please feel free to contact Fa Zhang (fa.zhang@connect.ust.hk), Prof. Can Yang (macyang@ust.hk), or Prof. Jianping Gan (magan@ust.hk) if any inquiries.
