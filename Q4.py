import numpy as np
import functions as fc
import matplotlib.pyplot as plt
import scipy.fftpack

class Q4():
	
	def __init__(self):
		self.F = fc.functions(seed=5239589)
		self.grid_points = 64
		self.min_distance = 1.
		self.n = -2
		
	def _run_functions(self):
		self.Q4a()
		self.Q4b()
		#Creating the wavevector used in 4c and 4d
		self.k_both = self.F.wave_vect(self.grid_points,self.min_distance)
		self.Q4c()
		self.Q4d()
	
	#Question 4a
	def Q4a(self):
	
		function = self.F.int_function
	
		a = 1./51

		final = self.F.D(a)
	
		print ('Growth factor at z = 50, D(50) = {0}'.format(final))
	
	#Question 4b
	def Q4b(self):
		function = self.F.int_function
	
		a = 1./51
	
		#calculate the integration factor that is used within the analytical derivative function
		int = self.F.ROM_integrator(10,function,0,a)
	
		final = self.F.derivative_D(a,int)
		
		deriv = self.Ridder(1/51.,10,self.F.D)[0]*(1/51.)*(7.16*(10**(-11)))*np.sqrt((0.3*(51**3))+0.7)
	
		print ("Analytical derivative at z = 50: {0}".format(final))
		print ("Numerical derivative at z = 50: {0}".format(deriv))
		print ("With a relative error of: {0}".format((final-deriv)/final))
	
	#Numerical derivative of a given function
	def Ridder(self,x,m,function,min_error=1e-12):
		'''
		Ridder's method for numerical differentiation
		Parameters:
		x = x value at which the derivate needs to be determined
		m = number of initial functions for Neville's algorithm
		function = the function to differentiate
		min_error = minimum error required - optional
		'''
		#Set the begin parameters
		h_new = 0.01 #Delta 'x' paramater
		d = 2 #For every iteration the delta 'x' paramater is decreased by a factor of d 
		M = list([0])*m #The number of initial functions used for analogue to Neville's algorithm
		error = 2**64 #Starting error set to maximum
		error_new = error
		
		#Loop over the columns k
		for k in range(m):
			#Loop over the rows l
			for l in range(m-k):
				#If in column 0, use the central difference method to find 
				#the derivative given certain delta x (=h)
				if k == 0:
					M[l] = (function(x+h_new) - function(x-h_new))/(2*h_new)
					#Reduce the delta x (h) by a factor d after every iteration over l
					h_new /= d
				#Use Neville's algorithm to combine previous answers 
				#and create better ones
				else:
					M_old = M[l]
					M[l] = (((4**(k))*M[l+1])-M[l])/((4**(k))-1)
					error_new = max(abs(M[l]-M_old),abs(M[l]-M[l+1]))
				#set the error to the new_error if it is smaller
				if error_new <= error:
					error = error_new
					answer = M[l]
				#When the desired accuracy is accomplished, exit the loops
				if error <= min_error:
					break
			if error > min_error:
				continue
			break
			
		return answer,error
		
	#Question 4c
	def Q4c(self):
		
		grid_points = self.grid_points
		size = self.grid_points*self.grid_points
		#Create random numbers for the whole matrix
		RNG = self.F.uniform(size=4*size)
		nums = np.array([i for i in RNG])
		nums = np.sqrt(-2*np.log(nums[2*size:]))*np.sin(2*np.pi*nums[:2*size])
		
		#Instead of filling the matrix and making it symmetric at the same time,
		#the matrix is first fillend and then made symmetric (inefficient but works)
		matrix = self.create_2D_matrix(nums)
		matrix_symmetric = self.make_matr_symmetric(matrix)
		
		#Creating the density field that it should converge to and plotting it
		density_field = scipy.fftpack.ifft2(matrix_symmetric)*(grid_points)
		plt.imshow(density_field.real,cmap='plasma')
		plt.gca().invert_yaxis()
		plt.xlabel(r'$x$')
		plt.ylabel(r'$y$')
		plt.colorbar()
		plt.savefig('plots/density_field_Q4c.png')
		plt.close()
		
		#Multiplying the wavenumbers with each matrix element
		grid_x = matrix_symmetric*self.k_both*1J
		grid_y = matrix_symmetric*self.k_both[:,np.newaxis]*1J
		
		#Making the nyquist entries symmetric again
		#in the x-grid
		grid_x[int(grid_points/2)+1:,int(grid_points/2)] = -1*grid_x[int(grid_points/2)+1:,int(grid_points/2)] 
		grid_x[0,int(grid_points/2)] = grid_x[0,int(grid_points/2)].imag + 0J
		grid_x[int(grid_points/2),int(grid_points/2)] = grid_x[int(grid_points/2),int(grid_points/2)].imag + 0J
		#in the y-grid
		grid_y[int(grid_points/2),int(grid_points/2)+1:] = -1*grid_y[int(grid_points/2),int(grid_points/2)+1:]
		grid_y[int(grid_points/2),0] = grid_y[0,int(grid_points/2)].imag + 0J
		grid_y[int(grid_points/2),int(grid_points/2)] = grid_y[int(grid_points/2),int(grid_points/2)].imag + 0J
		
		#Inverse fourier transform both grids
		Sx = scipy.fftpack.ifft2(grid_x)*(grid_points)
		Sy = scipy.fftpack.ifft2(grid_y)*(grid_points)
		
		#Creating the initialing X and Y positions of the particles
		xs = np.arange(0,grid_points,1)
		ys = np.arange(0,grid_points,1)
		X_ini,Y_ini = np.meshgrid(xs,ys)
		
		#Plotting the initial X and Y positions
		plt.figure(figsize=(5,5))
		plt.scatter(X_ini,Y_ini,s=0.8,color='black')
		plt.xlabel(r'$x$')
		plt.ylabel(r'$y$')
		plt.savefig('movie_2d/0.png')
		plt.close()

		#Creating the expansion factors 
		a = np.linspace(0.0025,1,90)
		a_prev = a[0]
	
		Pys = [[]]*len(a)
		y_positions = [[]]*len(a)
		
		function = self.F.int_function
		for i in range(len(a)):
			a_curr = a[i] #current a value
			delta_a = abs(a_prev-a_curr) #delta a value
			
			#integral for the calculation of D
			D = self.F.D(a_curr)
			#intgral for the calculation of dot(D)
			intgrl = self.F.ROM_integrator(10,function,0,a_curr-(0.5*delta_a))
			D_deriv = self.F.derivative_D(a_curr-(0.5*delta_a),intgrl)
			
			#Create the step sizes for each particles
			add_x = D*Sx.real
			add_y = D*Sy.real
			
			#Add them to the initial positions (with mod to make it circular)
			X = (X_ini + add_x)%grid_points
			Y = (Y_ini + add_y)%grid_points
		
			#Creating the momenta
			Px = -((a_curr-(0.5*delta_a))**2)*D_deriv*Sx.real
			Py = -((a_curr-(0.5*delta_a))**2)*D_deriv*Sy.real
	
			Pys[i] = Py[0:10,0]
			y_positions[i] = Y[0:10,0]
		
			a_prev = a_curr
		
			plt.figure(figsize=(5, 5))
			plt.scatter(X,Y,s=1.2,color='black')
			plt.xlabel(r'$x$')
			plt.ylabel(r'$y$')
			plt.title('a = {0}'.format(a[i]))
			plt.savefig('movie_2d/{0}.png'.format(i+1))
			plt.close()
		
		#Last scatter plot is plotted together with the density field
		plt.scatter(X,Y,s=1.2,color='black')
		plt.imshow(density_field.real,cmap='plasma')
		plt.gca().invert_yaxis()
		plt.xlabel(r'$x$')
		plt.ylabel(r'$y$')
		plt.colorbar()
		plt.savefig('plots/density_field_Q4c.png')
		plt.close()
		
		#plotting momentum
		Pys = np.array(Pys)
		for i in range(10):
			plt.plot(a,Pys[:,i],label='particle {0}'.format(i+1))
		plt.xlabel(r'$a$')
		plt.ylabel(r'$p_y$')
		plt.legend()
		plt.savefig('plots/py_a.png')
		plt.close()
		
		#plotting y positions
		y_positions = np.array(y_positions)
		for i in range(10):
			plt.plot(a,y_positions[:,i],label='particle {0}'.format(i+1))
		plt.xlabel(r'$a$')
		plt.ylabel(r'$y$')
		plt.legend()
		plt.savefig('plots/y_a.png')
		plt.close()
	
	#Creating a 2D matrix with complex entries
	def create_2D_matrix(self,nums):
		'''
		Parameters: 
		nums = the random numbers for filling up the entire matrix
		'''
		size = self.grid_points
		n = self.n
		matrix = np.zeros((size,size),dtype=complex)

		total = 0
		for i in range(size):
			for j in range(size):
				if i == 0 and j == 0:
					continue
				k = np.sqrt((self.k_both[i]**2) + (self.k_both[j]**2))
				#Each element is multiplied by sqrt(P(k))/(k^2)
				matrix[i,j] = 0.5*complex(nums[total],nums[total+1])*np.sqrt(k**n)/(k**2)
				total += 2
		return matrix
	
	#Making the matrix symmetric
	def make_matr_symmetric(self,matrix):
		'''
		Parameters:
		matrix = the matrix that has to be made symmetric
		'''
		size = self.grid_points
		
		for i in range(1,int(size/2)+1):
			#The first column and row are conjugate symmetric. (0,0) is skipped.
			matrix[i,0] = complex(matrix[-i,0].real, -matrix[-i,0].imag)
			matrix[0,i] = complex(matrix[0,-i].real, -matrix[0,-i].imag)
			#All the other elements are also conjugate symmetric
			for j in range(1,size):
				matrix[i,j] = complex(matrix[size-i,size-j].real, -matrix[size-i,size-j].imag)
		
		for i in range(0,size):
			if i == 0 or i == int(size/2):
				#The points (0,Nyq) (Nyq,0) (Nyq,Nyq) are real numbers
				matrix[int(size/2),i] = matrix[int(size/2),i].real + 0J
				matrix[i,int(size/2)] = matrix[i,int(size/2)].real + 0J
				continue
			#The other values in the nyquist row and column are complex numbers, but
			#conjugate symmetric since Nyq=-Nyq
			matrix[int(size/2),i] = complex(matrix[int(size/2),i].real, -matrix[int(size/2),i].imag)
			matrix[i,int(size/2)] = complex(matrix[i,int(size/2)].real, -matrix[i,int(size/2)].imag)
		
		return matrix
	
	#Question 4d
	def Q4d(self):
	
		grid_points = self.grid_points
		size = grid_points**3
		k_both = self.k_both
		n = self.n
		
		#Creating the random numbers that fills up the whole 3d matrix
		#Note: Inefficient to fill it up entirely, but seemed easiest way out for now
		RNG = self.F.uniform(size=2*2*size)
		nums = np.array([i for i in RNG])
		nums = np.sqrt(-2*np.log(nums[2*size:]))*np.sin(2*np.pi*nums[:2*size])
	
		#Create the 3D matrix and make it fourier symmetric
		matrix = self.create_matrix_3D(nums)
		matrix = self.make_matr_symmetric_3D(matrix)
		
		#This is not efficient since three matrices will take up more space
		#this solution did seem to be the most easy for now
		X,Y,Z = np.meshgrid(k_both,k_both,k_both)
		#Multiplying each matrix element with the wavenumbers
		grid_x = matrix*X*1.J
		grid_y = matrix*Y*1.J
		grid_z = matrix*Z*1.J

		#Nyquist symmetry is lost so it has to be made symmetric again
		#This method is extremely inefficient, but it works!
		#These symmetries have been found by trying out 4x4x4 matrices all the time
		#and checking where the symmetries in the nyquist frequencies go wrong
		nyq = int(grid_points/2)
		#For Y grid (for the nyquist rows/columns)
		grid_y[nyq,0,nyq+1:] = -1*grid_y[nyq,0,nyq+1:]
		grid_y[nyq,nyq,nyq+1:] = -1*grid_y[nyq,nyq,nyq+1:]
		grid_y[nyq,nyq+1:,:] = -1*grid_y[nyq,nyq+1:,:]
		#These entries should be real but due to the *1j multiplication
		#they became imaginary, therefore to make them real again, the imaginary
		#part is taken to be the real part. (The same holds for the z and x grid, but at
		#different entries)
		grid_y[nyq,0,0] = grid_y[nyq,0,0].imag
		grid_y[nyq,nyq,0] = grid_y[nyq,nyq,0].imag
		grid_y[nyq,0,nyq] = grid_y[nyq,0,nyq].imag
		grid_y[nyq,nyq,nyq] = grid_y[nyq,nyq,nyq].imag
		#For Z grid - slicing (for the nyquist rows/columns)
		grid_z[:,nyq+1:,nyq]= -1*grid_z[:,nyq+1:,nyq]
		grid_z[nyq+1:,nyq,nyq] = -1*grid_z[nyq+1:,nyq,nyq]
		grid_z[nyq+1:,0,nyq] = -1*grid_z[nyq+1:,0,nyq]
		#single entries (for the imaginary part)
		grid_z[0,0,nyq] = grid_z[0,nyq,0].imag
		grid_z[0,nyq,nyq] = grid_z[0,nyq,nyq].imag
		grid_z[nyq,0,nyq] = grid_z[nyq,nyq,0].imag
		grid_z[nyq,nyq,nyq] = grid_z[nyq,nyq,nyq].imag
		#And for X grid - slicing (for the nyquist rows/columns)
		grid_x[nyq+1:,nyq,0] = -1*grid_x[nyq+1,nyq,0]
		grid_x[nyq+1,nyq,nyq] = -1*grid_x[nyq+1,nyq,nyq]
		grid_x[:,nyq,nyq+1:] = -1*grid_x[:,nyq,nyq+1:]
		#single entries (for the imaginary part)
		grid_x[0,nyq,0] = grid_x[0,nyq,0].imag
		grid_x[0,nyq,nyq] = grid_x[0,nyq,nyq].imag
		grid_x[nyq,nyq,0] = grid_x[nyq,nyq,0].imag
		grid_x[nyq,nyq,nyq] = grid_x[nyq,nyq,nyq].imag
		
		#Inverse fourier transform all three grids (with denormalization constant)
		Sx = scipy.fftpack.ifftn(grid_x)*(grid_points**(3/2.))
		Sy = scipy.fftpack.ifftn(grid_y)*(grid_points**(3/2.))
		Sz = scipy.fftpack.ifftn(grid_z)*(grid_points**(3/2.))

		#Creating the initial positions
		xs = np.arange(0,grid_points,1)
		ys = np.arange(0,grid_points,1)
		zs = np.arange(0,grid_points,1)
		X_ini,Y_ini,Z_ini = np.meshgrid(xs,ys,zs)
		
		#Plotting the first 3 initial positions by taking a slice of the 3D matrix
		x_slice = np.where(abs(X_ini - 32) <= 0.5)
		y_slice = np.where(abs(Y_ini - 32) <= 0.5)
		z_slice = np.where(abs(Z_ini - 32) <= 0.5)
		
		plt.figure(figsize=(5, 5))
		plt.scatter(X_ini[z_slice],Y_ini[z_slice],s=0.5,color='black')
		plt.savefig('movie_3d/XY.png'.format(0))
		plt.close()
		plt.figure(figsize=(5, 5))
		plt.scatter(X_ini[y_slice],Z_ini[y_slice],s=0.5,color='black')
		plt.savefig('movie_3d/XZ.png'.format(0))
		plt.close()
		plt.figure(figsize=(5, 5))
		plt.scatter(Y_ini[x_slice],Z_ini[x_slice],s=0.5,color='black')
		plt.savefig('movie_3d/YZ_{0}.png'.format(0))
		plt.close()

		#Creating the expansion factors between the given boundaries
		a = np.linspace(1/51.,1,90)
	
		a_prev = a[0]
	
		Pzs = [[]]*len(a)
		z_positions = [[]]*len(a)
		function = self.F.int_function
		for i in range(len(a)):		
			a_curr = a[i] #current a value
			delta_a = abs(a_prev-a_curr) #delta a value
		
			#integral for the calculation of D
			#intgrl = self.F.ROM_integrator(10,function,0,a_curr)
			D = self.F.D(a_curr)
			#integral for the calculation of dot(D)
			intgrl = self.F.ROM_integrator(10,function,0,a_curr-(0.5*delta_a))
			D_deriv = self.F.derivative_D(a_curr-(0.5*delta_a),intgrl)
			
			#Creating the step sizes in each direction
			add_x = D*Sx.real
			add_y = D*Sy.real
			add_z = D*Sz.real
			
			#Adding them to the initial positions and taking the mod to make
			#the positions circular
			X = np.add(X_ini,add_x)%grid_points
			Y = np.add(Y_ini,add_y)%grid_points
			Z = np.add(Z_ini,add_z)%grid_points
			
			#Calculating the momenta
			Px = -((a_curr-(0.5*delta_a))**2)*D_deriv*Sx.real
			Py = -((a_curr-(0.5*delta_a))**2)*D_deriv*Sy.real
			Pz = -((a_curr-(0.5*delta_a))**2)*D_deriv*Sz.real
		
			Pzs[i] = Pz[0][0][0:10]
			z_positions[i] = Z[0][0][0:10]
			
			#Slicing the matrices and plotting them
			x_slice = np.where(abs(X - 32) <= 0.5)
			y_slice = np.where(abs(Y - 32) <= 0.5)
			z_slice = np.where(abs(Z - 32) <= 0.5)
		
			plt.figure(figsize=(5, 5))
			plt.title('a = {0}'.format(a[i]))
			plt.scatter(X[z_slice],Y[z_slice],s=0.5,color='black')
			plt.savefig('movie_3d/XY_{0}.png'.format(i+1))
			plt.close()
			plt.figure(figsize=(5, 5))
			plt.title('a = {0}'.format(a[i]))
			plt.scatter(X[y_slice],Z[y_slice],s=0.5,color='black')
			plt.savefig('movie_3d/XZ_{0}.png'.format(i+1))
			plt.close()
			plt.figure(figsize=(5, 5))
			plt.title('a = {0}'.format(a[i]))
			plt.scatter(Y[x_slice],Z[x_slice],s=0.5,color='black')
			plt.savefig('movie_3d/YZ_{0}.png'.format(i+1))
			plt.close()
		
		#Plotting momenta
		Pzs = np.array(Pzs)
		for i in range(10):
			plt.plot(a,Pzs[:,i],label='particle {0}'.format(i+1))
		plt.xlabel(r'$a$')
		plt.ylabel(r'$p_z$')
		plt.legend()
		plt.savefig('plots/pz_a.png')
		plt.close()
		
		#Plotting z positions
		z_positions = np.array(z_positions)
		for i in range(10):
			plt.plot(a,z_positions[:,i],label='particle {0}'.format(i+1))
		plt.xlabel(r'$a$')
		plt.ylabel(r'$z$')
		plt.legend()
		plt.savefig('plots/z_a.png')
		plt.close()
	
	#Filling a 3D matrix
	def create_matrix_3D(self,nums):
		'''
		Parameters: 
		nums = all the random numbers in an array
		'''
		size = self.grid_points
		n = self.n
		matrix = np.zeros((size,size,size),dtype=complex)
		total = 0 
		for i in range(size):
			for j in range(size):
				for l in range(size):
					if i == 0 and j == 0 and l == 0:
						continue
					#Each element is multiplied by sqrt(P(k))/(k^2)
					k = np.sqrt((self.k_both[i]**2) + (self.k_both[j]**2) + (self.k_both[l]**2))
					matrix[i,j,l] = 0.5*complex(nums[total],nums[total+1])*np.sqrt(k**n)/(k**2)
					total += 1
		return matrix
	
	#Making a 3D matrix symmetric
	def make_matr_symmetric_3D(self,matrix):
		'''
		Parameters:
		matrix = 3D matrix that has to be made conjugate symmetric
		'''
		size = self.grid_points
		
		#This method is done hardcoded, because it seemed easiest in 3D
		for x in range(0,int(size/2)+1):
			for y in range(0,size):
				for z in range(0,size):
					#Do nothing with first element, since it is 0 + 0J anyways
					if x == 0 and y == 0 and z==0:
						continue
					#At (Nyq,Nyq,Nyq) make it a real number
					elif x == int(size/2) and y == int(size/2) and z == int(size/2):
						matrix[x,y,z] = matrix[x,y,z].real + 0J
					#At (Nyq,0,0) make it a real number
					elif x == int(size/2) and y == 0 and z == 0:
						matrix[x,y,z] = matrix[x,-y,-z].real + 0J
					#At (0,Nyq,0) make it a real number
					elif y == int(size/2) and x == 0 and z == 0:
						matrix[x,y,z] = matrix[-x,y,-z].real + 0J
					#At (0,0,Nyq) make it a real number
					elif z == int(size/2) and y == 0 and x == 0:
						matrix[x,y,z] = matrix[-x,-y,z].real + 0J
					#At (Nyq,Nyq,0) make it a real number
					elif x == int(size/2) and y == int(size/2) and z == 0:
						matrix[x,y,z] = matrix[x,y,-z].real + 0J
					#At (Nyq,0,Nyq) make it a real number
					elif x == int(size/2) and y == 0 and z == int(size/2):
						matrix[x,y,z] = matrix[x,-y,z].real + 0J
					#At (0,Nyq,Nyq) make it a real number
					elif x == 0 and y == int(size/2) and z == int(size/2):
						matrix[x,y,z] = matrix[-x,y,z].real + 0J
					#When ever in a nyquist row: make it conjugate symmetric!
					elif x == int(size/2) and y != int(size/2) and z != int(size/2):
						matrix[x,y,z] = complex(matrix[x,-y,-z].real, -matrix[x,-y,-z].imag)
					elif y == int(size/2) and x != int(size/2) and z != int(size/2):
						matrix[x,y,z] = complex(matrix[-x,y,-z].real, -matrix[-x,y,-z].imag)
					elif z == int(size/2) and y != int(size/2) and x != int(size/2):
						matrix[x,y,z] = complex(matrix[-x,-y,z].real, -matrix[-x,-y,z].imag)
					elif x == int(size/2) and y == int(size/2) and z != int(size/2):
						matrix[x,y,z] = complex(matrix[x,y,-z].real, -matrix[x,y,-z].imag)
					elif x == int(size/2) and y != int(size/2) and z == int(size/2):
						matrix[x,y,z] = complex(matrix[x,-y,z].real, -matrix[x,-y,z].imag)
					elif x != int(size/2) and y == int(size/2) and z == int(size/2):
						matrix[x,y,z] = complex(matrix[-x,y,z].real, -matrix[-x,y,z].imag)
					#Make everyting else in the whole matrix also conjugate symmetric
					else:
						matrix[x,y,z] = complex(matrix[-x,-y,-z].real,-matrix[-x,-y,-z].imag)		
		
		return matrix
	
execute_question = Q4()
execute_question._run_functions()