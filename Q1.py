import numpy as np
import functions as fc
import matplotlib.pyplot as plt
import scipy.stats as sc
import astropy.stats
		
class Q1():
	
	def __init__(self):
		self.F = fc.functions(seed=123409)
	
	def _run_functions(self):
		self.Q1a()
		self.Q1b()
		self.Q1c()
		self.Q1d()
		self.Q1e()
	
	#Question 1a
	def Q1a(self):
		#Generating 1000 random numbers and scatter plot them
		N = 1000
	
		RNG = self.F.uniform(size=N)
		rand_nums = [i for i in RNG]
		
		f,ax=plt.subplots(1,2,figsize=(16,5))
		ax[0].scatter(rand_nums[0:999],rand_nums[1:1000],color='black',s=0.5)
		ax[0].set_xlabel(r'$x_i$')
		ax[0].set_ylabel(r'$x_{i+1}$')
		ax[1].plot(np.arange(0,len(rand_nums)),rand_nums)
		ax[1].set_xlabel(r'iterations, $i$')
		ax[1].set_ylabel('Random value')
		plt.savefig('plots/rand_nums.png')
		plt.close()
	
		#Generating one million random numbers and creating a histogram
		N = 1000000
	
		RNG = self.F.uniform(size=N)
	
		rand_nums = [i for i in RNG]

		plt.hist(rand_nums,bins=20)
		plt.xlabel('Random Value')
		plt.savefig('plots/unif_dist.png')
		plt.close()
	
	#Question 1b
	def Q1b(self):
		#Setting some constants
		sigma = 2.4
		mu = 3
		N = 1000
	
		#Generating 1000 random numbers that are normally distributed
		#with the use of the box-muller transform
		RNG = self.F.normal(mu,sigma,N)
		gauss = [i for i in RNG]
	
		#Creating a histogram of them and overplot it with the sigma lines,
		#and the actual gaussian distribution
		his = plt.hist(gauss,bins=14,density=True)
		maxim = max(his[0])+0.02*max(his[0])
		for i in range(5):
			plt.vlines((i+1)*sigma+mu,0,maxim,linestyle=':',color='black')
			plt.vlines(-(i+1)*sigma+mu,0,maxim,linestyle=':',color='black')
			plt.text((i+1)*sigma+mu,maxim,r'${0}\sigma$'.format(i+1))
			plt.text(-(i+1)*sigma+mu,maxim,r'$-{0}\sigma$'.format(i+1))
	
		x = np.linspace(-5*sigma,5*sigma,500)
		gauss = (1./np.sqrt(2*np.pi*(sigma**2)))*np.exp(-0.5*((x-mu)**2)/(sigma**2))
	
		plt.plot(x,gauss,color='red')
		plt.xlabel(r'$x$')
		plt.ylabel(r'$\mathcal{P}(x)$')
		plt.savefig('plots/normal_dist.png')
		plt.close()
	
	#Question 1c
	def Q1c(self):
		#Setting some constants
		sigma = 1
		mu = 0
		#Creating the decimal exponents
		exponents = np.arange(1,5.1,0.1)
		Ns = 10**exponents
		Ps = np.zeros(len(Ns))
		scipy_ps = np.zeros(len(Ns))
	
		#Pre-generate the random samples of the gaussian distribution
		RNG = self.F.normal(mu,sigma,int(max(Ns)))
		complete_norm = [i for i in RNG]
	
		#Looping over the different number of samples
		for i in range(len(Ns)):
			N = int(round(Ns[i]))
			gauss = complete_norm[0:N]
		
			#Sorting the array that contains the normal distribution using quicksort
			self.F.quicksort(gauss,0,len(gauss)-1,self.F.sorting)
			#Determining the actual cdf of this normal distribution
			real_cdf = self.F.cdf_sdnorm(gauss)
		
			#Calculating the Kolmogorov-Smirnov test statistic of this distribution	
			KS = self.KS_statistic(gauss,real_cdf)
			#Determining the p-value of this ks test statistic by subtracting the CDF from 1
			Ps[i] = 1 - self.F.KS_cdf(KS,N) #one minus de KS_cdf to convert it to a p-value
			
			#Determining the p-value as calculated by scipy
			scipy_ps[i] = sc.kstest(gauss,'norm')[1]

		plt.plot(Ns,scipy_ps,label='scipy',color='blue',alpha=0.7)
		plt.plot(Ns,Ps,label='own approx.',color='red',alpha=0.7)
		plt.hlines(0.05,min(Ns),max(Ns),color='green',label='p = 0.05',linestyle=':')
		plt.ylabel(r'$p$')
		plt.xlabel('Number of samples')
		plt.xscale('log')
		plt.legend()
		plt.savefig('plots/KS-test.png')
		plt.close()
	
	#The Kolmogorov-Smirnov test statistic 
	def KS_statistic(self,x,real_cdf):
		'''
		Parameters:
		x = sorted array of the distribution
		real_cdf = sorted array of the actual (discrete) cdf of the same distribution
		'''
		#The first approximation of the cdf is simply 1 over the total
		#number of elements in array x (the distribution).
		old_cdf = 1/len(x)
		max_diff = 0
		for k in range(1,len(x)):
			#update the new approximated cdf
			new_cdf = (k+1)/len(x)
			#calculate a potential new maximum, by comparing the previous
			#cdf element with the current cdf element
			pot_max = max(abs(new_cdf-real_cdf[k]),abs(old_cdf-real_cdf[k]))
			#if the new potential maximum is bigger than the current maximum
			#update the maximum
			if pot_max>max_diff:
				max_diff = pot_max
			old_cdf = new_cdf
		
		return max_diff	
		
	#Question 1d
	def Q1d(self):
		#Setting some constants
		sigma = 1
		mu = 0
		#Creating the decimal exponents
		exponents = np.arange(1,5.1,0.1)
		Ns = 10**exponents
		Ps = np.zeros(len(Ns))
		scipy_ps = np.zeros(len(Ns))
	
		#Pre-generate the random samples of the gaussian distribution
		RNG = self.F.normal(mu,sigma,int(max(Ns)))
		complete_norm = [i for i in RNG]
	
		#Looping over the different number of samples
		for i in range(len(Ns)):
			N = int(round(Ns[i]))
			gauss = complete_norm[0:N]
		
			#Sorting the array that contains the normal distribution using quicksort
			self.F.quicksort(gauss,0,len(gauss)-1,self.F.sorting)
			#Determining the actual cdf of this normal distribution
			real_cdf = self.F.cdf_sdnorm(gauss)
		
			#Calculating the Kuiper's test statistic of this distribution
			KP = self.kuiper_statistic(gauss,real_cdf)
			#Determining the p-value of this Kuiper's test statistic
			Ps[i] = self.kuiper_pval(KP,N)
		
			scipy_ps[i] = astropy.stats.kuiper(gauss,self.F.cdf_sdnorm)[1]

		plt.plot(Ns,scipy_ps,label='scipy',color='blue')
		plt.plot(Ns,Ps,label='own approx.',color='red')
		plt.hlines(0.05,min(Ns),max(Ns),linestyle=':',label='p = 0.05',color='green')
		plt.ylabel(r'$p$')
		plt.xlabel('Number of samples')
		plt.xscale('log')
		plt.legend()
		plt.savefig('plots/kuiper-test.png')
		plt.close()
	
	#The kuiper test statistic
	def kuiper_statistic(self,x,real_cdf):
		'''
		Parameters:
		x = sorted array of the distribution
		real_cdf = sorted array of the actual (discrete) cdf of the same distribution
		'''
		#Setting both maximums to an extremely small number
		max_diff_A = -2**32
		max_diff_B = -2**32
		old_cdf = 1/len(x)
		#Looping over the discrete distribution x
		for k in range(1,len(x)):
			#Determining the cdf value of the discrete distribution x
			new_cdf = (k+1)/len(x)
			#Finding what the maximum distances are
			pot_max_A = max(new_cdf-real_cdf[k],old_cdf-real_cdf[k]) 
			pot_max_B = max(real_cdf[k]-new_cdf,real_cdf[k]-old_cdf)
			#Accept newly found distances if it's bigger than previous maximums
			if pot_max_A>max_diff_A:
				max_diff_A = pot_max_A
			if pot_max_B>max_diff_B:
				max_diff_B = pot_max_B
			#Setting the old cdf to the new cdf (see solutions paper for explanation)
			old_cdf = new_cdf
		return max_diff_A+max_diff_B
	
	#Calculating the cdf of Kuiper's statistic
	def kuiper_pval(self,max_diff,N):
		'''
		Parameters:
		max_diff = maximum difference in distance between the two distributions that are 
				   being compared.
		N = number of samples from the distributions
		'''
		#See solutions paper for the equation that is calculated below
		L = (np.sqrt(N) + 0.155 + (0.24/np.sqrt(N)))*max_diff
		Qkp_old = 2*((4*(L**2)) - 1)*np.exp(-2*(L**2))
		Qkp_new = Qkp_old + (2*((4*4*(L**2)) - 1)*np.exp(-2*4*(L**2)))
		j = 3
		while abs(Qkp_old-Qkp_new) > 1e-8:
			Qkp_old = Qkp_new
			Qkp_new = Qkp_old + (2*((4*(j**2)*(L**2)) - 1)*np.exp(-2*(j**2)*(L**2)))
			j += 1
		return Qkp_new
	
	#Question 1e
	def Q1e(self):
		#Loading the txt file containing the columns with the random numbers
		numberfile = np.loadtxt('randomnumbers.txt',delimiter=' ')
		
		exponents = np.arange(1,5.1,0.1)
		Ns = 10**exponents
		
		#Pre-generate the random samples of the gaussian distribution 
		RNG = self.F.normal(0,1,int(max(Ns)))
		complete_norm = [i for i in RNG]
		
		#Looping over each column
		fig,ax = plt.subplots(1,2,figsize=(12,5))
		iters = 0
		for i in range(len(numberfile[0])):
			Ps_kolm = np.zeros(len(Ns))
			Ps_scipy = np.zeros(len(Ns))
			#Looping over the various sample sizes
			for j in range(len(Ns)):
				N = int(Ns[j])
				pot_gauss = numberfile[0:N,i]
				own_gauss = complete_norm[0:N]
				
				#Sorting both distributions
				self.F.quicksort(own_gauss,0,N-1,self.F.sorting)
				self.F.quicksort(pot_gauss,0,N-1,self.F.sorting)
				
				#Calculating the KS-statistic using a 'discrete' version, where
				#now both given distributions are discrete
				KS = self.KS_statistic_discrete(pot_gauss,own_gauss)
				
				#Calculating the cdf of the KS statistic
				N = len(pot_gauss)*len(own_gauss)/(len(pot_gauss)+len(own_gauss))
				Ps_kolm[j] = 1 - self.F.KS_cdf(KS,N)
			
			ax[iters].plot(Ns,Ps_kolm,label='KS-test, column {0}'.format(i))
			ax[iters].set_ylabel(r'$p$')
			ax[iters].set_xlabel('Number of samples')
			ax[iters].set_xscale('log')
			ax[iters].plot([min(Ns),max(Ns)],[0.05,0.05],linestyle=':',color='green',label='p = 0.05')
			ax[iters].legend(loc='upper center')
			iters += 1
			if i%2 != 0:
				
				plt.savefig('plots/KS-test_column_{0}.png'.format(i))
				plt.close()
				iters = 0
				fig,ax = plt.subplots(1,2,figsize=(12,5))
		
	#The KS-statistic given two discrete distributions
	def KS_statistic_discrete(self,dist_1,dist_2):
		'''
		Parameters: 
		dist_1 = values of distribution 1 (sorted)
		dist_2 = values of distribution 2 (sorted)
		'''
		N = len(dist_1)
		indx_a = 1
		indx_b = 1
		max_diff = 0
		#Go through all values of both distributions
		while indx_a < N and indx_b < N:
			if abs((indx_a/N)-(indx_b/N))>max_diff:
				max_diff = abs((indx_a/N)-(indx_b/N))
			if dist_1[indx_a] < dist_2[indx_b]:
				indx_a += 1
			else:
				indx_b += 1	 
		return max_diff
		
execute_question = Q1()
execute_question._run_functions()