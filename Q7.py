import numpy as np
import matplotlib.pyplot as plt
import h5py

class Q7():

	def __init__(self):
		self.datafile = h5py.File('colliding.hdf5','r')
		
	def _prepare_data(self):
		#Get the coordinates of all particles
		group_keys = list(self.datafile.keys()) #contains: 'Header', 'PartType4' (data), 'Units'
		data = list(self.datafile[group_keys[1]]) #contains: 'Coordinates' (x,y,z), 'Masses', 'ParticleIDs', 'Velocities'
		self.coordinates = np.array((self.datafile[group_keys[1]][data[0]]))
			
	def _run_functions(self):
		#First build the tree
		tree = Tree(0,0,150,self.coordinates)
		#Now make a plot of the tree
		tree.make_plot()
		plt.savefig('plots/tree.png')
		plt.close()
		#Get the correct children of the tree (for particle i = 100)
		tree.get_children()

class Tree():

	def __init__(self,x,y,size,coords):
		self.x = x #x-coordinate of the square (left lower corner x0)
		self.y = y #y-coordinate of the square (left lower corner y0)
		self.size = size #size of the square (width = height = size)
		self.coords = coords #coordinates of the particles
		self.build_tree(x,y,size,coords) 
	
	#Build the tree
	def build_tree(self,x,y,size,coords):
		#Start at the root (x = 0, y = 0, size = width = height = 150) and go through
		#the tree
		self.root = Leaf(self.x,self.y,self.size,self.coords)
		#Get the multipole moments of all nodes
		self.root.mp()
	
	#Get the children for i = 100, and print out the n = 0 multipole moments
	def get_children(self):
		self.root.get_children(self.coords[:,0][100],self.coords[:,1][100],0)
	
	#Function to make the plot
	def make_plot(self):
		fig,ax = plt.subplots(1,1,figsize=(7,7))
		self.root.make_plot(ax)
		ax.set_xlim(0,150)
		ax.set_ylim(0,150)
		ax.set_xlabel(r'$x$')
		ax.set_ylabel(r'$y$')
		
class Leaf():
	
	def __init__(self,x,y,size,coords):
		self.x = x #x-coordinate of the square (left lower corner x0)
		self.y = y #y-coordinate of the square (left lower corner y0)
		self.size = size #size of the square
		#If the number of particles within this square is more than 12: split 
		#the tree in 4
		if len(coords[:,0]) > 12:
			#calculate the center coordinates of the square
			x_center = x+(0.5*size)
			y_center = y+(0.5*size)
			#Create the 4 children where x0 and y0 are now divided over 4 points
			self.children = [Leaf(x,y,size/2,coords[(coords[:,0] < x_center) & (coords[:,1] < y_center)]),
							 Leaf(x,y_center,size/2,coords[(coords[:,0] < x_center) & (coords[:,1] > y_center)]),
							 Leaf(x_center,y,size/2,coords[(coords[:,0] > x_center) & (coords[:,1] < y_center)]),
							 Leaf(x_center,y_center,size/2,coords[(coords[:,0] > x_center) & (coords[:,1] > y_center)])]
			self.multipole = None		
		#If less than 12 particles in the node, save the coordinates of these particles
		#and their children, and multipole moment			 	
		else:
			self.coords = coords
			self.children = []
			self.multipole = None
	
	#Get the nodes and leaf given an x and y coordinate of a particle
	def get_children(self,x,y,i):
		
		if self.children:
			print ('n = 0 multipole moment in node {0} = {1}'.format(i,self.multipole*0.012500000186264515))
			i+=1
			x_center = self.x + (0.5*self.size)
			y_center = self.y + (0.5*self.size)
			#Check in which node the particle belongs:
			#South-west node
			if x < x_center and y < y_center:
				self.children[0].get_children(x,y,i)
			#North-west node
			elif x < x_center and y > y_center:
				self.children[1].get_children(x,y,i)
			#South-east node
			elif x > x_center and y < y_center:
				self.children[2].get_children(x,y,i)
			#North-east node
			elif x > x_center and y > y_center:
				self.children[3].get_children(x,y,i)
		else:
			print ('n = 0 multipole moment in leaf = {0}'.format(self.multipole*0.012500000186264515))
		
		
	#Create a plot of the tree
	def make_plot(self,ax):
		#If the node has children, calculate their square sizes
		if self.children:
			x_center = self.x + (0.5*self.size)
			y_center = self.y + (0.5*self.size)
			for child in self.children:
				child.make_plot(ax)
			ax.plot([x_center,x_center],[self.y,self.y+self.size],color='black',alpha=0.8)
			ax.plot([self.x,self.x+self.size] ,[y_center,y_center],color='black',alpha=0.8)
		else:
			ax.scatter(self.coords[:,0],self.coords[:,1],s=0.8,color='green')
	
	#Calculate the multipole moment of a node
	def mp(self):
		if self.children:
			self.multipole = np.sum([child.mp() for child in self.children])
		else:
			self.multipole = len(self.coords[:,0])
		return self.multipole

execute_question = Q7()
execute_question._prepare_data()
execute_question._run_functions()