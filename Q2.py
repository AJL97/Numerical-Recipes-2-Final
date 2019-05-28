import numpy as np
import functions as fc
import matplotlib.pyplot as plt
import scipy.fftpack

class Q2():
	
	def __init__(self):
	
		self.F = fc.functions(seed=3438941)
		self.maxd = 1024
		self.min_distance = 1.
	
	def _run_functions(self):
		#Creating the random numbers
		self.random_nums()
		#Creating a symmetric matrix and inverse fourier transforming it
		#for different power values
		for n in [-3,-2,-1]:
			self.IFFT(n)
	
	#Creating random numbers
	def random_nums(self):
		size = self.maxd*self.maxd
		#creating all the random numbers before hand, to save time
		RNG = self.F.uniform(size=2*size)
		nums = np.array([i for i in RNG])
	
		#Transforming them with the Box-Muller transform
		self.nums = np.sqrt(-2*np.log(nums[size:]))*np.sin(2*np.pi*nums[:size])
	
	#Inverse fourier transform 
	def IFFT(self,n):
		#Create a symmetric fourier plane with the given random numbers
		grid = self.symmetric_FS(self.maxd,self.nums,n,self.maxd*self.min_distance)
		#Inverse fourier transform it and denormalize it (since scipy normalizes it)
		inv_fourier = scipy.fftpack.ifft2(grid)*self.maxd
		plt.imshow(inv_fourier.real,cmap='plasma')
		plt.colorbar()
		plt.title(r'$n = ${0}'.format(n))
		plt.savefig('plots/density_field_{0}.png'.format(n+4))
		plt.close()
	
	#Creating a symmetric fourier matrix
	def symmetric_FS(self,maxd,nums,n,max_dist):
		'''
		Parameters:
		maxd = size of the grid (in one dimension)
		nums = the random numbers
		n = the power in the power spectrum 
		max_dist = maximum physical distance
		'''
	
		grid = np.zeros((maxd,maxd),dtype=np.complex)
	
		#initialize the wavevector, k_both
		k_1 = np.arange(0,int(maxd/2)+1,1)
		k_2 = np.arange(-int(maxd/2)+1,0,1)
		k_both = np.concatenate((k_1,k_2))/max_dist
	
		total = 0
		#This loop makes sure that all the values in the matrix are symmetric
		for i in range(1,int(maxd/2)):
			for j in range(1,maxd):
				k = np.float(np.sqrt((k_both[i]**2) + (k_both[j]**2)))
				s = np.sqrt(k**n)
				num_1 = nums[total]
				num_2 = nums[total+1]
				grid[i,j],grid[maxd-i,maxd-j] = complex(nums[total]*s,nums[total+1]*s),complex(nums[total]*s,-nums[total+1]*s)
				total += 2
		
		#This loop makes sure that the nyquist frequencies have the correct symmetry as well
		for i in range(0,int(maxd/2)):
			k = np.sqrt((k_both[i]**2) + ((maxd/2)**2))
			#for i = 0: setting the elements to zero that have elements 0 and nyquist
			#(see solution paper for more thorough explanation)
			if i == 0:
				fact_vals = [np.sqrt(k**n),k]
				s = np.sqrt(k**n)
				grid[int(maxd/2),i] = complex(nums[total]*s,0)
				s = np.sqrt((maxd/2)**n)
				grid[i,int(maxd/2)] = complex(nums[total+1]*s,0)
				total += 2
				continue
			s = np.sqrt(k**n)
			grid[int(maxd/2),i],grid[int(maxd/2),maxd-i] = complex(nums[total]*s,0),complex(nums[total]*s,0)
			grid[i,int(maxd/2)],grid[maxd-i,int(maxd/2)] = complex(nums[total+1]*s,0),complex(nums[total+1]*s,0)
			total += 2
		
		#Setting the most middle element (nyquist,nyquist) equal to a real valued number
		k = (maxd/2)*np.sqrt(2)
		s = np.sqrt(k**n)
		grid[int(maxd/2),int(maxd/2)] = complex(nums[total]*s,0)
		total += 1
		
		#This loop makes sure that the first row and the first column also have the correct
		#symmetry
		for i in range(1,int(maxd/2)):
			if i == int(maxd/2):
				continue
			k = np.sqrt(k_both[i]**2)
			s = np.sqrt(k**n)
			grid[0,i],grid[0,maxd-i] = complex(nums[total]*s,nums[total+1]*s),complex(nums[total]*s,-nums[total+1]*s)
			total += 2
			grid[i,0],grid[maxd-i,0] = complex(nums[total]*s,nums[total+1]*s),complex(nums[total]*s,-nums[total+1]*s)
			total += 2
		return grid

execute_question = Q2()
execute_question._run_functions()