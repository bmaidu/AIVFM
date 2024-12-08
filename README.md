# AI-VFM
We present AI-VFM, a new vector flow mapping (VFM) method enabled by recent advances in artificial intelligence (AI). AI-VFM uses physics-informed neural networks (PINNs) encoding mass conservation, momentum balance, and boundary conditions to recover intraventricular flow and pressure fields from standard echocardiographic scans. AI-VFM performs phase unwrapping and recovers missing data in the form of spatial and temporal gaps in the input color-Doppler data, thereby producing super-resolution flow maps. AI-VFM is solely informed by each patient's flow physics;  it does not utilize explicit smoothness constraints or incorporate data from other patients or flow models. AI-VFM shows good validation against ground-truth data from CFD, outperforming traditional VFM methods as well as similar PINN-based VFM formulations relying exclusively on mass conservation. 
# Data
Please download the following computational fluid dynamics validation [data](https://drive.google.com/drive/folders/1zjpoTHym4fZzLpfVVgzvdCT02a8nTvV2?usp=sharing) and save it to the local directory for use in the AI-VFM code.
# Citation
@article{maidu2024,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;title={Super-resolution Left Ventricular Flow and Pressure Mapping by Navier-Stokes-Informed Neural Networks},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;author={Maidu, Bahetihazi and Martinez-Legazpi, Pablo and Guerrero-Hurtado, Manuel and Nguyen, Cathleen M. and Gonzalo, Alejandro and Kahn, Andrew M. and Bermejo, Javier and Flores, Oscar and del Alamo, Juan C.},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;journal={Computers in Biology and Medicine},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;volume={},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;number={},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;page={},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;year={2025}
}
