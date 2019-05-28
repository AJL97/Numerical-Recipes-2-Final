import numpy as np
import matplotlib.pyplot as plt

class Q6():

	def __init__(self):
		self.datafile = np.loadtxt('GRBs.txt',skiprows=2,usecols=(2,3,4,5,6,7,8))
		self.names = np.loadtxt('GRBs.txt',skiprows=2,usecols=(0),dtype=str)
		self.thetas = np.zeros(5)
		self.step_size = 0.001
		self.Max_epochs = 1e5
	
	def _run_functions(self):
		self._prepare_data()
		self._train()
		self._acc()
		self._plot()
	
	def _prepare_data(self):
		#The shape of the train set is 220 rows and 6 columns
		train = np.ones((self.datafile.shape[0]-15,(self.datafile.shape[1]-3)+1))
		#for the labels it is 220 rows
		labels = np.zeros(self.datafile.shape[0]-15)
		
		#going over all the data
		iters_2 = 0
		for i in range(self.datafile.shape[1]):
			iters = 0
			for j in range(len(self.datafile[:,i])):
				#exclude the data point if it contains the name 'XRF'
				if self.names[j] == 'XRF':
					continue
				#Making an exponent of the log-columns
				if i == 2 or i == 3:
					self.datafile[:,i][j] = np.exp(self.datafile[:,i][j])
				#If the entry is exp(-1), set it to zero	
				if self.datafile[:,i][j] == np.exp(-1):
					self.datafile[:,i][j] = 0
				#If the entry is -1, set it to zero
				elif self.datafile[:,i][j] == -1:
					self.datafile[:,i][j] = 0
				#don't include the last two columns of the data set
				if i < 5 and i != 1:
					train[iters][iters_2] = self.datafile[:,i][j]
				iters += 1
			if i <5 and i != 1:
				iters_2+=1

		#Getting the labels
		iters = 0
		for i in range(len(self.datafile[:,1])):
			if self.datafile[:,1][i] >= 10 and self.names[i] != 'XRF':
				labels[iters] = 1
			else:
				labels[iters] = 0
			if self.names[i] == 'GRB':
				iters+=1
	
		self.train = np.array(train)
		self.labels = np.array(labels)
	
	#sigmoid function
	def sigmoid(self,x):
		return 1/(1+np.exp(-x))
	
	#Loss function
	def L(self,y,y_hat):
		'''
		Parameters:
		y = true labels
		y_hat = estimated labels
		'''
		return -((y*np.log(y_hat+0.0001)) - ((1-y)*np.log(1.0001-y_hat)))
	
	#cost function
	def cost_func(self,Ls):
		'''
		Parameters:
		Ls = losses 
		'''
		return -(1./len(Ls))*np.sum(Ls)
	
	#training the network
	def _train(self):

		M = len(self.train)
		#Estimate the labels for the first time
		y_hat = self.sigmoid(np.sum(np.array(self.train*self.thetas),axis=1))
		#Calculate the loss for the first time
		Ls = self.L(self.labels,y_hat)
		#Calculate the cost for the first time
		cost = self.cost_func(Ls)

		err = (2**32)-1
		epochs = 0
		losses = []
		#Starting the training
		while err > 1e-3 and epochs < self.Max_epochs:
    		
    		#Append losses
			losses.append(Ls)
			#Update the weights (thetas)
			for i in range(len(self.thetas)):
				self.thetas[i] = self.thetas[i] - ((self.step_size/M)* np.sum((y_hat - self.labels)*self.train[:,i]))
    		
    		#Estimate the labels with the newly updated weights
			y_hat = self.sigmoid(np.sum(np.array(self.train*self.thetas),axis=1))
			#Determine the loss function
			Ls = self.L(self.labels,y_hat)
			#Determine the cost function
			cost_new = self.cost_func(Ls)
    
			err = cost_new - cost
			cost = cost_new
			epochs += 1

	def _acc(self):
    	
		train = self.train
		labels = self.labels
		weights = self.thetas
		#predict the labels with the final weights
		y_est = self.sigmoid(np.sum(np.array(train*weights),axis=1))
		#round the labels of to either 0 or 1
		y_est_rounded = np.zeros(len(y_est))
		for i in range(len(y_est)):
			if y_est[i] >= 0.5:
				y_est_rounded[i] = 1
			else:	
				y_est_rounded[i] = 0
		
		self.y_est = y_est_rounded
		self.accuracy = (len(labels) - np.sum(np.abs(labels-y_est_rounded)))/len(labels)
    		
	def _plot(self):
		hist = plt.hist(self.labels,alpha=0.5,label='actual')
		hist = plt.hist(self.y_est,alpha=0.5,label='predicted')
		plt.ylabel('Number of GRBs')
		plt.xlabel('Classification (0=short, 1=long)')
		plt.legend()
		plt.savefig('plots/logistic_regression.png')
    			
execute_question = Q6()
execute_question._run_functions()	