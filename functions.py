import numpy as np

class functions(object):
	
	def __init__(self,seed=453678,D=None):
		
		#Seed of the PRNG
		self.x = np.uint64(seed)
		#Constant for the MWC
		self.a = np.uint64(4294957665) 
		self._SHIFT = np.uint64(32)
		
		#Constants for the XOR-shift
		self.b1 = np.uint64(21)
		self.b2 = np.uint64(35)
		self.b3 = np.uint64(4)
		self._MAX_INT = np.uint32((2**32)-1)
		
		#Constants needed to approximate the error function
		self.p =  0.3275911
		self.a1 = 0.254829592
		self.a2 = -0.284496736
		self.a3 = 1.421413741
		self.a4 = -1.453152027
		self.a5 = 1.061405429
		
		#constants question 4
		self.om_m = 0.3
		self.om_l = 0.7
		self.H_0 = 7.16e-11
	
	#The XOR-shift generator used by the RNG
	def XOR(self,number):
		'''
		Parameters:
		number = integer number put in the XOR generator
		'''
		#XOR operator on the initial bits integer and initial bits 
		#integer shifted b1 to the left 
		number ^= number << self.b1
		#XOR operator on the new bits integer and new bits 
		#integer shifted b2 to the right 
		number ^= number >> self.b2
		#XOR operator on the new bits integer and new bits 
		#integer shifted b3 to the left 
		number ^= number << self.b3
		
		return number
	
	#The MWC used by the RNG
	def MWC(self,x):
		'''
		Parameters:
		x = integer number 
		'''
		num = self.a*(x&self._MAX_INT) + (x>>self._SHIFT)
		return np.uint64(num)
	
	#The PRNG that creates a uniform distribution between a and b
	def uniform(self,a=0,b=1,size=1):
		'''
		Parameters:
		a = minimum value of the uniform distribution, standard 0
		b = maximum value of the uniform distribution, standard 1
		size = number of values to return
		'''
		for i in range(size):
			self.x = self.XOR(self.x)
			num = self.MWC(self.x)
			num = num&self._MAX_INT
			dec_rand_num = num/self._MAX_INT
				
			yield (dec_rand_num*(b-a))+a

	#Function that transforms a uniform distribution to a normal distribution
	def normal(self,mu=0,s=1,size=1):
		'''
		Parameters: 
		mu = mean value of the normal distribution, standard 0
		s = standard deviation of the normal distribution, standard 1
		size = number of values to return
		'''
		#Creating two random values sampled from a uniform distribution
		#NOTE: it is more efficient to return 2 values X,Y since the box-muller
		#transform creates two digits. But how this program is set up, it was more
		#convenient to use it like this. For better efficiency I'd do it differently
		#from the start
		RNG = self.uniform(size=size*2)
		unifs = [i for i in RNG]
		for i in range(size):
			yield (np.sqrt(-2*(s**2)*np.log(unifs[i]))*np.sin(2*np.pi*unifs[len(unifs)-1-i])) + mu 
	
	#Sorting an array using the Quicksort algorithm
	def sorting(self,array,indx_arr,min,max):
		'''
		Parameters:
		array = array that needs to be sorted
		indx_arr = index array that needs to be sorted
		min = minimum index from where it needs to be sorted
		max = maximum index untill where it needs to be sorted
		'''
		#Set the pivot and pivot index to the middle element of the array
		pivot_indx = int((min+max)/2)
		pivot = array[pivot_indx]
		i = min
		j = max
		
		#Continue while i and j have not crossed each other
		while j>=i:
			indx_i = i
			indx_j = j
			#Increase index i if it's element is smaller than the pivot
			if array[indx_i] < pivot:
				i+=1
			#Decrease index j if it's element is bigger than the pivot
			if array[indx_j] > pivot:
				j-=1
			#If element i is bigger (or equal to) than the pivot and element j
			#is smaller (or equal) than the pivot, swap the elements
			elif array[indx_i] >= pivot and array[indx_j] <= pivot:
				self.swap(array,indx_i,indx_j)
				if pivot == array[indx_i]:
					pivot_indx = indx_i
				elif pivot == array[indx_j]:
					pivot_indx = indx_j
				i+=1
				j-=1
		#If element i is bigger than the pivot and is left to the pivot index
		#Swap the pivot with element i
		if array[i] > pivot and i < pivot_indx:
			self.swap(array,i,pivot_indx)
			pivot_indx = i
		#If element j is bigger than the pivot and is right to the pivot index
		#Swap the pivot with element j
		elif array[j] < pivot and j > pivot_indx:
			self.swap(array,pivot_indx,j)
			pivot_indx = j

		return pivot_indx
	
	#Swapping two elements in an array
	def swap(self,array,i,j):
		'''
		Parameters:
		array = array that needs to be sorted
		i = index i of array
		j = index j of array
		'''
		swap_i = array[i]
		swap_j = array[j]
		array[i] = swap_j
		array[j] = swap_i
	
	#Sorting an array using the Quicksort algorithm
	def quicksort(self,array,min,max,sort,indx_arr=None):
		'''
		Quicksort algorithm where the middle element is used as pivot element
		Parameters:
		array = array to be sorted
		indx_arr = index array of the array to be sorted
		min = minimum index of array that needs to be sorted
		max = maximum index of array that needs to be sorted
		sort = sorting method (either 'sorting' (array sorting)
							   or 'index_sorting' (index array sorting))
		'''
		if min < max:	
			#Sort the pivot element to get it's right place
			new_piv = sort(array,indx_arr,min,max)
			#Sort everything on the left side of the starting pivot element
			self.quicksort(array,min,new_piv-1,sort,indx_arr)
			#Sort everything on the right side of the starting pivot element
			self.quicksort(array,new_piv+1,max,sort,indx_arr)	
	
	#Determining the cdf of a given discrete standard normal distribution with a numerical
	#approximation that is more thoroughly discussed in the solutions paper
	def cdf_sdnorm(self,x):
		'''
		Parameters:
		x = array containing the sorted discrete pdf of a standard normal dist.
		'''
		#The full function is given in the solution paper
		x = np.array(x)
		t = 1./(1-(self.p*(1./np.sqrt(2))*x[x<0]))
		x[x<0] = -1 + (((self.a1*t)+(self.a2*(t**2)) + (self.a3*(t**3)) + (self.a4*(t**4)) + (self.a5*(t**5)))*np.exp(-0.5*(x[x<0]**2)))
		t = 1./(1+(self.p*(1./np.sqrt(2))*x[x>=0]))
		x[x>=0] = 1 - (((self.a1*t)+(self.a2*(t**2)) + (self.a3*(t**3)) + (self.a4*(t**4)) + (self.a5*(t**5)))*np.exp(-0.5*(x[x>=0]**2)))
		return 0.5*(1+x)
	
	#The Kolmogorov-Smirnov CDF
	def KS_cdf(self,max_diff,N):
		'''
		Parameters:
		max_diff = maximum distance between actual cdf and estimated cdf
		N = total number of samples of the distribution
		'''	
		#See the solution paper for the full functions
		z = (np.sqrt(N) + 0.12 + (0.11/np.sqrt(N)))*max_diff
		if z < 1.18:
			exp = np.exp(-(np.pi**2)/(8*(z**2)))
			ps = (np.sqrt(2*np.pi)/z)*(exp + (exp**9) + (exp**25))
		elif z >= 1.18:
			exp = np.exp(-2*(z**2))
			ps = 1-(2*(exp - (exp**4) + (exp**9)))
		return ps
	
	#The function to be integrated
	def int_function(self,x):
		fact = 1./(x**3)
		return fact/(((self.om_m*fact)+self.om_l)**(3./2))
	
	#Romberg integration algorithm
	def ROM_integrator(self,m,function,xmin=0,xmax=1):
		'''
		Parameters:
		m = number of initial functions for Neville's algorithm
		function = function that needs to be integrated
		a,b,c = constants
		A = normalization factor
		Nsat = average total number of satellites
		'''
		#S is the number of initial functions used for Neville's Algorithm
		S = np.zeros(m)
		#Interval to integrate on
		x_min = xmin
		x_max = xmax
		
		N = len(S)
		#Looping over column k
		for k in range(N):
			#Looping over row l
			for l in range(N-k):
				#When in column k=0, define all the initial functions
				if k == 0:
					#Determining which x-points should be used
					x_n,h = self.new_x(l,x_min,x_max)
					#Determining the trapezoids given spacing parameter h
					S[l] = (h*np.sum(function(x_n)))
				#Use analogue Neville's algorithm to combine the initial integrals
				else:
					S[l] = (((4**(k))*S[l+1])-S[l])/((4**(k))-1)
		return (S[0])
	
	#Returning an equally spaced (h) array in the intervals x_min and x_max
	def new_x(self,n,a,b):
		'''
		Parameters:
		n = index of the array
		a = minimum value of the array
		b = maximum value of the array
		'''
		if n == 0: h = 0.5*(b-a) #spacing parameter (step sizes in the array)
		else: h = (1/(2**n))*(b-a) #spacing parameter (step sizes in the array)
		x = np.arange(a+(h/2),b,h)
		return x,h
	
	#The derivative of the growth factor
	def derivative_D(self,a,int):
		'''
		Parameters:
		a = expansion factor
		int = integral (see self.int_function)
		function = 
		'''
		first_factor = 5*self.H_0*self.om_m/(2*(a**2))
		second_factor = (-3*self.om_m/(2*a))*int
		third_factor = 1/(((self.om_m*(1/(a**3)))+self.om_l)**0.5)
		return first_factor*(second_factor + third_factor)
	
	#Function of the growth factor
	def D(self,a):
		'''
		Parameters:
		a = expansion factor
		'''
		return self.ROM_integrator(10,self.int_function,0,a)*(5/2.)*0.3*np.sqrt(((0.3*((1/a)**3))+0.7))
		
	#Creating the wave vector
	def wave_vect(self,grid_points,min_distance):
		'''
		Parameters:
		grid_points = number of grid points of the image
		min_distance = minimal distance of one grid point
		'''
		k_1 = np.arange(0,int(grid_points/2)+1,1)
		k_2 = np.arange(-int(grid_points/2)+1,0,1)
		k_both = 2*np.pi*np.concatenate((k_1,k_2))/(grid_points*min_distance)
		return k_both

	#Bisection algorithm
	def bisection_index_finding(self,x,x_data):
		'''
		Parameters:
		x = the to be evaluated x data point
		x_data = the given x data points
		'''
		#Starting indexes are at the borders of the given data
		left_idx = 0 #Left bound
		right_idx = len(x_data)-1 #Right bound
		
		#If the to be evaluated point falls outside of the bounds,
		#return the first two (or last two) indexes of the given data points
		if x < x_data[left_idx]: return left_idx+1,left_idx
		elif x > x_data[right_idx]: return right_idx-1,right_idx
	
		while True:
			#Stop if the difference between left and right is 1
			if right_idx-left_idx == 1:
				break
			#Divide the left plus right index by two, to find 
			#the middle index
			split = int((left_idx + right_idx)/2)
			
			#Determining on which side x is 
			if x_data[split] > x: right_idx = split
			else: left_idx = split
		
		return left_idx,right_idx
	
	#Nearest grid point method
	def NGP(self,x,y,z):
		'''
		Parameters:
		x = x positions of the particles
		y = y positions of the particles
		z = z positions of the particles
		'''
		ngd = np.zeros((1024,3),dtype=np.uint64)
		grid = np.zeros((16,16,16),dtype=np.float64)
		ranges = np.arange(0,16.1,1)
		
		#Loop over the particles
		for i in range(len(x)):
			#get the nearest grid points for particle i in direction x
			l_x,r_x = self.bisection_index_finding(x[i],ranges)
			#Check to which of the grid point it is closes to
			if x[i] - l_x > r_x - x[i]:
				pos_x = r_x%16
			else:
				pos_x = l_x%16
			#get the nearest grid points for particle i in direction y
			l_y,r_y = self.bisection_index_finding(y[i],ranges)
			#Check to which of the grid point it is closes to
			if y[i] - l_y > r_y - y[i]:
				pos_y = r_y%16
			else:
				pos_y = l_y%16
			#get the nearest grid points for particle i in direction z
			l_z,r_z = self.bisection_index_finding(z[i],ranges)
			#Check to which of the grid point it is closes to
			if z[i] - l_z > r_z - z[i]:
				pos_z = r_z%16
			else:
				pos_z = l_z%16
			#update the grid by adding the full fraction of the mass to this grid point
			#position
			grid[pos_x,pos_y,pos_z] += 1
		
		return grid	