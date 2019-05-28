import numpy as np
import matplotlib.pyplot as plt

class Q3():
	
	def __init__(self):
		self.vals_cases = np.array([[2,3],[-10,10],[0,5]])
		self.analyt_vals = np.array([[3,0],[0,10],[3,2]])
		self.rtol = 1e-3
		self.atol = 1e-3
		self.S = 0.9
		
	def _run_functions(self):
		#Looping over the different cases
		for i in range(len(self.vals_cases)):
			self.calculate_ODE(self.vals_cases[i],self.analyt_vals[i],i)
		
	def calculate_ODE(self,vals,analyt_vals,i):
	
		t = 1
		h = 0.01
		function = self.ODE
		
		#For this exercise appending has been used, unfortunately, because it is not 
		#known how many steps the ODE solver will make before hand.
		us = []
		us.append(vals[0])
		Ds = []
		Ds.append(vals[1])
		ts = []
		ts.append(t)
	
		while t < 1000:
			#Calculate the new and embedded values
			vals_n,vals_emb = self.RK5(h,t,vals,function)
			
			#The difference between the new and embedded values
			tri = vals_n-vals_emb
			#The scale factors
			scale = np.array([self.atol + (max(abs(vals[0]),abs(vals_n[0]))*self.rtol),self.atol + (max(abs(vals[1]),abs(vals_n[1]))*self.rtol)])
			#Dividing the difference between new and embedded by the scale factors to calculate
			#the error
			err = np.sqrt(0.5*np.sum((tri/scale)**2))
			
			#If the error is small enough: accept step
			if err <= 1:
				t = t + h
				h = h*self.S*((1/err)**(1/5.))
				ts.append(t)
				Ds.append(vals_n[1])
				us.append(vals_n[0])
				vals = vals_n
			#else reject it and introduce a new step size h
			else:
				h = h*self.S*((1/err)**(1/5.))	

		plt.plot(ts,Ds,label='numerical', color='blue',alpha=0.7)
		analytical_Ds = self.analytical(np.array(ts),*analyt_vals)
		plt.plot(ts,analytical_Ds,label='analytical',color='red',alpha=0.7)
		plt.legend()
		plt.loglog()
		plt.title('case {0}'.format(i+1))
		plt.savefig('plots/ODE_case_{0}.png'.format(i+1))
		plt.close()
		
	#The ODE that has to be solved (consists of two first order ODEs but when
	#combined it is one second order ODE
	def ODE(self,t,vals):
		return np.array([(-4./(3*t))*vals[0] + vals[1]*(2./(3*(t**2))), vals[0]])
	
	#The analytical solution to the second order ODE
	def analytical(self,t,A,B):
		return (A*(t**(2./3))) + (B*(t**(-1)))
	
	#The fifth order Runge-Kutta method (hard-coded because it seemed easiest)
	def RK5(self,h,t,vals,func):
		
		#Calculating all the k factors and combining them
		k1 = h*func(t,vals)
		k2 = h*func(t+(0.2*h),vals+(k1*0.2))
		k3 = h*func(t+(0.3*h),vals+((3/40)*k1)+((9/40)*k2))
		k4 = h*func(t+(0.8*h),vals+((44/45)*k1)-((56/15)*k2)+((32/9)*k3))
		k5 = h*func(t+((8/9)*h),vals+((19372/6561)*k1)-((25360/2187)*k2)+((64448/6561)*k3)-((212/729)*k4))
		k6 = h*func(t+h,vals+((9017/3168)*k1)-((355/33)*k2)+((46732/5247)*k3)+((49/176)*k4)-((5103/18656)*k5))
		
		#Adding them (with pre-factors) to the original values to create the new value and
		#the embedded value
		vals_n = vals + ((35/384)*k1) + ((500/1113)*k3) + ((125/192)*k4) - ((2187/6784)*k5) + ((11/84)*k6)
		vals_emb = vals + ((5179/57600)*k1) + ((7571/16695)*k3) + ((393/640)*k4) - ((92097/339200)*k5) + ((187/2100)*k6)
		
		return vals_n,vals_emb
execute_question = Q3()
execute_question._run_functions()