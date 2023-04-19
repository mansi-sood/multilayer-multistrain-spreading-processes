# multilayer-multistrain-spreading-processes
The repository includes codes used in the paper 'Spreading Processes with Mutations over Multi-layer Networks', Sood, Sridhar, Eletreby, Wu, Levin, Poor, Yagan 2023

*In this work, we provide a mathematical framework to analyze spreading processes triggered by mutating contagions in light of non-pharmaceutical interventions such as lockdowns that reduce physical contact in different social settings (e.g., schools and offices). To this end, we analyze multi-strain spreading on multi-layer contact networks, where network layers repre- sent different social settings.*


### SIMULATION CODE FOR MULTI-STRAIN SPREADING ON MULTILAYER CONTACT NETWORKS: 1-**multistrain_spreading_multilayer-sim.py**

- 'multistrain_spreading_multilayer-sim.py' contains the code for experimentally simulating multi-strain spreading on a multi-layer network and obtaining the experimental values for probability of emergence, expected epidemic size and the fraction of the population infected with each strain type in Figure 3 and Figure 4.
- dependencies required: 
	- argparse, collections, random, sys, pdb, site, os, numpy, igraph, joblib, multiprocessing, time, matplotlib, itertools, mpmath
- recommended Python version: Python 3.9.1 64-bit
- in the current working directory, create a directory called experiment_files/ and run the python script 1-multistrain_spreading_multilayer-sim.py
- the default initialization of the code is for the data point \lambda_w=1 in Figure 3. Set parameters t1 = [0.6, 0.8], t2 = [0.7, 0.9], mu1=0.1, mu2=0.95 for Figure 3 and vary mean_degree2 in [0,5].Similarly, for Figure 4 change the parameters to 
t1 = [0.5, 0.7], t2 = [0.3, 0.4], mu1=0.2, mu2=0.5 and vary mean_degree1 in {1,2,3} and mean_degree2 in [0,3]. Note that here the index '1' and '2' respectively correspond to the two network layers.
- For SI Figure S1, please use the function runExp_pl_cutoff(:) in place of runExp(:)


### JOINT IMPACT OF MUTATIONS/LAYER OPENINGS: **2-joint-impact-mutations-opening.py**

- 'joint-impact-mutations-opening.py' provides the code for heatmaps for probability of emergence as predicted by Theorem 1 in Figure 6
- dependencies required:
	- numpy, scipy, matplotlib, mpl_toolkits
- recommended Python version: Python 3.9.1 64-bit
- the transmissibility parameters 'tf1', 'tf2', and 'c' can be changed to obtain plots in the figure, where,
	- tf1 :transmissibility of strain-1 in layer-f
	- tf2 :transmissibility of strain-2 in layer-f
	- c :the ratio tw1/tf1 and tw2/tf2 
- run 2-joint-impact-mutations-opening.py; the generated plot is saved with the file name "{c}\_{tf1}\_{tf2}.png" 
- the default initialisation of the code corresponds to the plot for c=1.0, tf1=0.4, tf2=0.8 in Figure 6
- For Figure 6 plots, vary c in {1.0,1.5} and (tf1,tf2) in {(0.4,0.8),(0.2,0.9),(0.3,0.5)}
- For SI Figure S3, vary c in {1.0,1.2,1.5}  while keeping tf1=0.4, tf2=0.8 constant. The probability $P_\mu^{1-->2}$ is obtained through Equations 21-23 in SI Appendix.




### ANALYTICAL PREDICTIONS OF PROBABILITY OF EMERGENCE: **3-probability-of-emergence-th.py** 

- 'probability-of-emergence-th.py' provides the code for analytical probability of emergence as obtained from Theorem 1 used in Figures 3, 4, and 7
- dependencies required:
	- numpy, scipy, math
- recommended Python version: Python 3.9.1 64-bit
- the default initialization of the code is for the data point \lambda_w=1 in Figure 3 with parameters tf1=0.6, tf2=0.8, tw1=0.7, tw2=0.9, mf1=mw1=0.1, mf2=mw2=0.95; to create the theoretical predictions for probability of emergence in Figure 3, vary poisson_mean_w in [0,5]. 
	- For Figure 4, change the parameters to tf1=0.5, tf2=0.7, tw1=0.3, tw2=0.4, mf1=mw1=0.2, mf2=mw2=0.5. 
	- For MS-ML probability values in Figure 7(a), set tf1=0.4, tf2=0.8, tw1=1.2\*tf1, tw2=1.2\*tf2, poisson_mean_f=poisson_mean_w=1.2, mf2=mw2=1-1e-10 and vary muf1=muw1 in [0,1]. For Figures 7(b)-(d) vary (mf1,mf2)=(mw1,mw2) in {(0.2,1-1e-10 ),(0.9,0.4),(0.5,0.5)} and keep tf2=0.8, and vary tf1/tf2 in [0,1]. For SS-ML probability values in Figure 7(a): For the transformation Tx-rho and Tx-J, use the effective transmissibility values tf = max(np.linalg.eigvals(J)) and tf = (tf1\*(tf1\*mf1+tf2\*(1-mf2))+tf2\*(tf1\*(1-mf1)+tf2\*mf2))/(tf1+tf2)respectively, and use tw = c\*tf.
	- To obtain the plot in SI Figure S1, update the PGF from Poisson to Power Law Distribution with exponential cutoff as specified in the comments in the code.
	- For SI Figure S2, set tf1 =0.5, tw1=0.3, tw2=0.6, mf1=mw1 =0.8 and mf2=mw2=1; for SS-ML with strain-2 predictions use the transmissibility value of strain-2 for both strains, for computing the probability of emergence of strain-2 in the chain of infections starting from strain-1, use Equations 21-23 in SI Appendix. 



### ANALYTICAL PREDICTION OF EXPECTED EPIDEMIC SIZE: **4-epidemic-size-th.py** 

- 'epidemic-size-th.py' provides the predicated expected size of the epidemic outbreak and fraction of the population infected with each strain as predicted by Theorem 3
- dependencies required:
	- numpy, scipy
- Note- please use the following Python version for running this code:  Python 2.7.15 64-bit
- the default initialization of the code is for the data point \lambda_w=1 in Figure 3 with parameters tf1=0.6, tf2=0.8, tw1=0.7, tw2=0.9, mf1=mw1=0.1, mf2=mw2=0.95; to create the theory values for Q, Q1 and Q2 in Figure 3, vary poisson_mean_w in [0,5]. 
	- Similarly, for Figure 4 change the parameters to tf1=0.5, tf2=0.7, tw1=0.3, tw2=0.4, mf1=mw1=0.2, mf2=mw2=0.5. 
	- In Figure 7, use 4-epidemic-size-th.py code for size predictions for MS-ML with the pmf for d_f updated for the mix distribution as updated in the comments in the code. In Figure 7(a), set tf1=0.4, tf2=0.8, tw1=1.2\*tf1, tw2=1.2\*tf2, poisson_mean_f=poisson_mean_w=1.2, mf2=mw2=1-1e-10 and vary muf1=muw1 in [0,1]. For Figures 7(b)-(d) vary (mf1,mf2)=(mw1,mw2) in {(0.2,1-1e-10 ),(0.9,0.4),(0.5,0.5)} and fix tf2=0.8, and vary tf1/tf2 in [0,1].
	- For SI Figure S1, use 4-epidemic-size-th.py with precomputing the pmf for the power law degree distribution and storing the pmf values in dictionaries as indicated in the comments.
	- For SI Figure S4, use tf1=0.6, tf2=0.8, tw1=0.7, tw2=0.9, mf1=mw1=0.1, mf2=mw2=0.95, lambda_f=1, and vary lambda_w in [0,4].



Note: Supplementary information with additional proofs and discussion are presented in the SI Appendix of the paper 'Spreading Processes with Mutations over Multi-layer Networks', Sood, Sridhar, Eletreby, Wu, Levin, Poor, Yagan 2023.
Please contact the authors for any questions/feedback.


















