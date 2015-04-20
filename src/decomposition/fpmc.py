import numpy as np

class FPMC:
	def __init__(self):
		self.data = None
		self.a = self.vui = self.viu = self.vil = self.vli = None
		pass

	def factorize(self, data, kui, kil, alpha, lambda_):
		self.data = data
		lambda_ui = lambda_iu = lambda_li = lambda_il = lambda_
		print 'begin fpmc\n','u_num:',data['u_num'],'\n','i_num:',data['i_num'],'\n',\
		'kui',kui,'\n','kil',kil

		sigma = .2
		vui = self.vui = np.random.normal(0,sigma,(data['u_num'], kui))
		viu = self.viu = np.random.normal(0,sigma,(data['i_num'], kui))
		vil = self.vil = np.random.normal(0,sigma,(data['i_num'], kil))
		vli = self.vli = np.random.normal(0,sigma,(data['i_num'], kil))

		print (data['u_num']*data['i_num'])
		for it in range(data['u_num']*data['i_num']):
			u = np.random.randint(0, data['u_num'])
			t = np.random.randint(1, len(data['transactions'][u]))
			i = data['transactions'][u][t]

			j = np.random.randint(0, data['i_num']-1)
			if j >= i:
				j += 1

			delta = 1.-sigma*(self.__x(u,t,i) - self.__x(u,t,j))
			if np.isnan(delta):
				raise
			for f in range(kui):
				vui[u,f] += alpha*(delta*(viu[i,f]-viu[j,f])-lambda_ui*vui[u,f])
				viu[i,f] += alpha*(delta*vui[u,f]-lambda_iu*viu[i,f])
				viu[j,f] += alpha*(-delta*vui[u,f]-lambda_iu*viu[j,f])
			
			for f in range(kil):
				eta = vli[data['transactions'][u][t-1],f]
				vil[i,f] += alpha*(delta*eta-lambda_il*vil[i,f])
				vil[j,f] += alpha*(-delta*eta - lambda_il*vil[j,f])
				l = data['transactions'][u][t-1]
				vli[l, f] += alpha*(delta*(vil[i,f]-vil[j,f])-lambda_li*vli[l,f])

		return (vui,viu,vil,vli)

	def __x(self, u,t,i):
		return self.vui[u,:].dot(self.viu[i,:]) + self.vil[i,:].dot(self.vli[self.data['transactions'][u][t-1],:])

	def cal_a(self):
		vui = self.vui
		viu = self.viu
		vil = self.vil
		vli = self.vli
		a = self.a = np.zeros((self.data['u_num'], self.data['i_num'], self.data['i_num']))
		for u in range(self.data['u_num']):
			for l in range(self.data['i_num']):
				for i in range(self.data['i_num']):
					a[u,l,i] = np.dot(vui[u,:], viu[i,:])+np.dot(vil[i,:], vli[l,:])

			

