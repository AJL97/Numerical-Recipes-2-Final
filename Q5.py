import numpy as np
import functions as fc
import matplotlib.pyplot as plt
import scipy.fftpack

class Q5():
	
	def __init__(self):
		self.F = fc.functions(seed=1234901)
		
	def _run_functions(self):
		self.Q5a()
		self.Q5b()
		self.Q5c()
		self.Q5d()
		self.Q5e()
		
	#Question 5a
	def Q5a(self):
		#Create the 1024 particle positions
		np.random.seed(121)
		positions = np.random.uniform(low=0,high=16,size=(3,1024))
		ranges = np.arange(0,16,1)
		ngd = np.zeros((1024,3),dtype=np.uint64)
		
		#Creating the scheme with the nearest grid point method
		grid = self.F.NGP(positions[0],positions[1],positions[2])/1024.
		
		iters = 0
		fig,ax = plt.subplots(1,2)
		#plot the x-y grids for these z-values
		for z in [4,9,11,14]:
			#set all 0 values to none to make the plot more clear
			grid_xy = np.copy(grid[:,:,z])
			idxs_none = np.where(grid_xy == 0.000)
			grid_xy[idxs_none] = None
			im = ax[iters].matshow(grid_xy)
			im.set_clim(0.,3./1024)
			ax[iters].invert_yaxis()
			ax[iters].xaxis.set_ticks_position('bottom')
			fig.colorbar(im,ax=ax[iters],fraction=0.046,pad=0.04)
			ax[iters].set_xlabel(r'$x$')
			ax[iters].set_ylabel(r'$y$')
			ax[iters].set_title(r'$z = {0}$'.format(z))
			if iters%2 != 0:
				fig.tight_layout()
				plt.savefig('plots/NGD_{0}'.format(z),bbox_inches='tight')
				plt.close()
				fig,ax = plt.subplots(1,2)
				iters = -1
			iters += 1	
		plt.close()
	
	#Question 5b
	def Q5b(self):
		#Creating the x positions
		x_position = np.linspace(0,16,300)
		ranges = np.arange(0,17,1)
		cell_4 = np.zeros(len(x_position))
		cell_0 = np.zeros(len(x_position))
		#Looping over the positions
		for i in range(len(x_position)):
			#set y and z position to 0 because we only have 1 dimension
			grid = self.F.NGP([x_position[i]],[0],[0])
			#Get the value of cell 4 and 0
			cell_4[i] = grid[4,0,0]
			cell_0[i] = grid[0,0,0]
			
		plt.plot(x_position,cell_4,label='cell 4')
		plt.plot(x_position,cell_0,label='cell 0')
		plt.xlabel(r'$x$')
		plt.ylabel('fraction of mass received')
		plt.legend()
		plt.savefig('plots/NGD_test.png')
		plt.close()
		
	#Question 5c
	def Q5c(self):
		#Create the 1024 particle positions
		np.random.seed(121)
		positions = np.random.uniform(low=0,high=16,size=(3,1024))
		
		#Creating the scheme with the cloud in cell method
		grid = self.cloud_in_cell(positions[0],positions[1],positions[2])

		fig,ax = plt.subplots(1,2)
		iters = 0
		#plot the x-y grids for these z-values
		for z in [4,9,11,14]:
			#set all 0 values to none to make the plot more clear
			idxs_none = np.where(grid[:,:,z] == 0.000)
			grid_copy = np.copy(grid[:,:,z])
			grid_copy[idxs_none] = None
			im = ax[iters].matshow(grid_copy)
			im.set_clim(0.,0.0015)
			ax[iters].invert_yaxis()
			ax[iters].xaxis.set_ticks_position('bottom')
			fig.colorbar(im,ax=ax[iters],fraction=0.046,pad=0.04)
			ax[iters].set_xlabel(r'$x$')
			ax[iters].set_ylabel(r'$y$')
			ax[iters].set_title('z = {0}'.format(z))
			if iters%2 != 0:
				fig.tight_layout()
				plt.savefig('plots/CIC_{0}'.format(z),bbox_inches='tight')
				plt.close()
				fig,ax = plt.subplots(1,2)
				iters = -1
			iters += 1

		#Checking the robustness of the implementation above in 1D
		x_position = np.arange(0,16,0.01)
		cell_4 = np.zeros(len(x_position))
		cell_0 = np.zeros(len(x_position))
		#Loop over all the positions again
		for i in range(len(x_position)):
			#set y and z position to 0 because we only have 1 dimension
			grid = self.cloud_in_cell([x_position[i]],[0],[0])
			#Get the value of cell 4 and 0
			cell_4[i] = grid[4,0,0]
			cell_0[i] = grid[0,0,0]
				
		fig,ax = plt.subplots(1,1)
		ax.plot(x_position,cell_4,label = 'cell 4')
		ax.plot(x_position,cell_0,label = 'cell 0')
		ax.set_xlabel(r'$x$')
		ax.set_ylabel('fraction of mass received')
		plt.legend()
		plt.savefig('plots/CIC_test.png')
		plt.close()
	
	#Cloud in cell method
	def cloud_in_cell(self,x,y,z):
		'''
		Parameters:
		x = x positions of the particles
		y = y positions of the particles
		z = z positions of the particles
		'''
		grid = np.zeros((16,16,16))
		
		for n in range(len(x)):
			#i,j,k represent the x0,y0,z0 gridpoint (in the cube figure that would
			#be equal to the green dot.
			i = int(x[n])%16
			j = int(y[n])%16
			k = int(z[n])%16
			#determining the distance of the particle in each direction to the 
			#x0,y0,z0 gridpoint
			dx = (x[n] - i)%16
			dy = (y[n] - j)%16
			dz = (z[n] - k)%16
			#assigning mass to each gridpoint (making sure that index 16 is assigned to index 0)
			#see solution paper for more thorough explanation for the masses
			grid[i][j][k] += (1-dx)*(1-dy)*(1-dz)/1024.
			grid[i][(j+1)%16][k] += (1-dx)*dy*(1-dz)/1024.
			grid[i][j][(k+1)%16] += (1-dx)*(1-dy)*dz/1024.
			grid[(i+1)%16][j][k] += dx*(1-dy)*(1-dz)/1024.
			grid[i][(j+1)%16][(k+1)%16] += (1-dx)*dy*dz/1024.
			grid[(i+1)%16][(j+1)%16][k] += dx*dy*(1-dz)/1024.
			grid[(i+1)%16][j][(k+1)%16] += dx*(1-dy)*dz/1024.
			grid[(i+1)%16][(j+1)%16][(k+1)%16] += dx*dy*dz/1024.
		
		return grid
		
	#Question 5d
	def Q5d(self):
		
		#creating a function (easy) to fourier transform
		N = 512
		x = np.linspace(0,N,N)
		y = np.cos(2*np.pi*20*x)
		y = np.array(y,dtype=complex)
		#creating the wavevector
		k_1 = np.arange(0,int(N/2)+1,1)
		k_2 = np.arange(-int(N/2)+1,0,1)
		k_both = 2*np.pi*np.concatenate((k_1,k_2))
	
		#fourier transforming the function with the own implementation and scipy
		
		FT_scipy = scipy.fftpack.fft(y)
		FT = self.fft(y)
		fig,ax = plt.subplots(1,1)
		plt.plot(k_both,np.abs(FT_scipy),label='scipy',linestyle=':')
		plt.plot(k_both,np.abs(FT),label='own',alpha=0.7)
		plt.xlabel(r'$f_x$')
		plt.ylabel('Amplitude')
		plt.vlines(40*np.pi,0,max(abs(FT)),color='red',label = 'analytical')
		plt.vlines(-40*np.pi,0,max(abs(FT)),color='red')
		plt.xlim(-200,200)
		plt.legend()
		plt.savefig('plots/FFT_1D.png')
		plt.close()
	
	#fast fourier transform
	def fft(self,x):
		'''
		Parameters:
		x = discrete data that has to be fourier transformed
		'''
		n = len(x)
		if n == 1:
			return x
		else:
			#recurse over the odd and even values of the data
			x_even = self.fft(x[0::2])
			x_odd = self.fft(x[1::2])
			
			x_new = np.zeros_like(x)
			x_new = np.asarray(x_new,dtype=complex)
			#Using the periodicity in N and N + (n/2), and using the even and odd parts
			#see solution paper for more thorough explanation.
			for N in np.arange(int(n/2)):
				x_new[N] = x_even[N] + np.exp(-2.*np.pi*1j*N/n)*x_odd[N]
				x_new[N + int(n/2)] = x_even[N] - np.exp(-2.*np.pi*1j*N/n)*x_odd[N]
			
			return x_new
	
	#Question 5e
	def Q5e(self):
		#2 dimensional case
		#creating the 2D function
		N = 64
		x = np.linspace(0,N,N)
		X,Y = np.meshgrid(x,x)
		y = np.sin(2*np.pi*(0.1*Y + 0.1*X))
		y = np.array(y,dtype = complex)
		
		#showing real space
		y_real = y.real
		fig,ax = plt.subplots(1,1)
		plt.imshow(y_real)
		plt.xlabel(r'$x$')
		plt.ylabel(r'$y$')
		plt.savefig('plots/real_space_2D.png')
		plt.close()
		
		#2D fourier transforming it with the own implementation and scipy
		FT_scipy = scipy.fftpack.fft2(y)
		FT = self.fft2(y)

		fig, ax = plt.subplots(1,2)
		im = ax[0].imshow(abs(FT_scipy))
		fig.colorbar(im,ax=ax[0],fraction=0.046,pad=0.04)
		ax[0].set_title('scipy')
		ax[0].invert_yaxis()
		ax[0].set_xlabel(r'$f_x$')
		ax[0].set_ylabel(r'$f_y$')
		
		im = ax[1].imshow(abs(FT))
		fig.colorbar(im,ax=ax[1],fraction=0.046,pad=0.04)
		ax[1].set_ylabel(r'$f_y$')
		ax[1].set_xlabel(r'$f_x$')
		ax[1].set_title('own')
		ax[1].invert_yaxis()
		fig.tight_layout()
		plt.savefig('plots/FFT_2D.png',bbox_inches='tight')
		plt.close()
		
		#3 dimensional case
		#creating the 3D multivariate gaussian
		N = 64
		x = np.linspace(-30,30,N)
		X,Y,Z = np.meshgrid(x,x,x)
		y = self.gauss(X,Y,Z)
		y = np.array(y,dtype=complex)
		FT_scipy = scipy.fftpack.fftn(y)
		FT = self.fft3(y)
		
		fig, ax = plt.subplots(1,2)
		im = ax[0].imshow(abs(FT_scipy[int(N/2),:,:]))
		fig.colorbar(im,ax=ax[0],fraction=0.046,pad=0.04)
		im = ax[1].imshow(abs(FT[int(N/2),:,:]))
		fig.colorbar(im,ax=ax[1],fraction=0.046,pad=0.04)
		ax[0].invert_yaxis()
		ax[1].invert_yaxis()
		ax[0].set_title('Scipy')
		ax[1].set_title('Own')
		ax[0].set_xlabel(r'$f_y$')
		ax[0].set_ylabel(r'$f_z$')
		ax[1].set_xlabel(r'$f_y$')
		ax[1].set_ylabel(r'$f_z$')
		fig.tight_layout()
		plt.savefig('plots/gauss_YZ.png',bbox_inches='tight')
		plt.close()
		fig, ax = plt.subplots(1,2)
		im = ax[0].imshow(abs(FT_scipy[:,int(N/2),:]))
		fig.colorbar(im,ax=ax[0],fraction=0.046,pad=0.04)
		im = ax[1].imshow(abs(FT[:,int(N/2),:]))
		fig.colorbar(im,ax=ax[1],fraction=0.046,pad=0.04)
		ax[0].invert_yaxis()
		ax[1].invert_yaxis()
		ax[0].set_title('Scipy')
		ax[1].set_title('Own')
		ax[0].set_xlabel(r'$f_x$')
		ax[0].set_ylabel(r'$f_z$')
		ax[1].set_xlabel(r'$f_x$')
		ax[1].set_ylabel(r'$f_z$')
		fig.tight_layout()
		plt.savefig('plots/gauss_XZ.png',bbox_inches='tight')
		plt.close()
		fig,ax = plt.subplots(1,2)
		im = ax[0].imshow(abs(FT_scipy[:,:,int(N/2)]))
		fig.colorbar(im,ax=ax[0],fraction=0.046,pad=0.04)
		im = ax[1].imshow(abs(FT[:,:,int(N/2)]))
		fig.colorbar(im,ax=ax[1],fraction=0.046,pad=0.04)
		ax[0].invert_yaxis()
		ax[1].invert_yaxis()
		ax[0].set_title('Scipy')
		ax[1].set_title('Own')
		ax[0].set_xlabel(r'$f_x$')
		ax[0].set_ylabel(r'$f_y$')
		ax[1].set_xlabel(r'$f_x$')
		ax[1].set_ylabel(r'$f_y$')
		fig.tight_layout()
		plt.savefig('plots/gauss_XY.png',bbox_inches='tight')
		plt.close()
	
	#2 dimensional FFT
	def fft2(self,x):
		'''
		Parameters:
		x = 2D matrix that contains the data that has to be fourier transformed
		'''
		#first fourier transforming the rows using 1D fft
		for rows in range(x.shape[0]):
			x[rows,:] = self.fft(x[rows,:])
		#transforming over the columns
		x = self.fft(x)
		
		return x
	
	#3 dimensional FFT
	def fft3(self,x):
		'''
		Parameters:
		x = 3D matrix that contains the data that has to be fourier transformed
		'''
		#first fourier transforming the first axis using 1D fft
		for i in range(x.shape[0]):
			x[i,:,:] = self.fft2(x[i,:,:])
		#transforming over the other two axes
		x = self.fft(x)
		
		return x
		
	#gaussian function
	def gauss(self,x,y,z):
		mu_x,mu_y,mu_z = 1,5,3
		s_x,s_y,s_z = 1,1,1 #have to be set to 1 otherwise the covariance is needed
		factor = (1/np.sqrt(2*np.pi)**3)*(1/s_x)*(1/s_y)*(1/s_z)
		exponent_x = np.exp(-((x-mu_x)**2)/(2*s_x**2))
		exponent_y =  np.exp(-((y-mu_y)**2)/(2*s_y**2))
		exponent_z = np.exp(-((z-mu_y)**2)/(2*s_z**2))
		return factor*exponent_x*exponent_y*exponent_z	

execute_question = Q5()
execute_question._run_functions()