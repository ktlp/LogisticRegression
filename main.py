import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\Kostas\Documents\αναγνώριση προτύπων\MLR.data",header=None, sep=' ')
X = df.to_numpy()
y = np.float_(X[:,-1])
X = np.float_(X[:,:-1])
print(np.version.version)
class Logistic_Model():
	def __init__(self,X,y):
		self.X = X
		self.y = y
		self.classes = np.unique(y).shape[0]
		self.dim = self.X.shape[1]
		self.MAX_ITERATIONS = 5
		self.t = self.construct_targets()

	def construct_targets(self):
		t = np.zeros((self.y.shape[0],self.classes))
		for i in range(self.y.shape[0]):
			t[i,int(self.y[i])] = 1
		return t

	def train(self):
		# initialize before iterative procedure
		w_new = np.random.random((self.classes,self.dim))

		for i in range(self.MAX_ITERATIONS):
			self.w = w_new
			# calculate output - y_nk
			self.y_hat = self.softmax()

			# calculate metrics
			self.success_rate = np.where(np.argmax(self.y_hat, axis=0) - self.y == 0)[0].size/self.y.shape[0]

			# calculate derivative vector K x D
			E = self.derivative()

			# calculate hessian matrix - K x K
			H = self.hessian()

			# update weights
			try:
				w_new = self.w + np.matmul(np.linalg.inv(H),E)
			except np.linalg.LinAlgError:
				print('Singular Matrix, using another initial weighting matrix..')
				w_new = np.random.random((self.classes, self.dim))

		self.w = w_new
		pass

	def hessian(self):
		res = np.empty((self.classes,self.classes))
		identity = np.diag(np.ones(self.classes))
		for i in range(self.classes):
			for j in range(self.classes):
				tmp = self.y_hat[i,:]*(identity[i,j] - self.y_hat[j,:])*np.sum(np.square(self.X), axis=1)
				tmp = - np.sum(tmp)
				res[i,j] = tmp
		return res

	def derivative(self):
		# self.y_hat 3 x N
		res = np.empty((self.classes,self.dim))
		for i in range(self.classes):
			# tmp -- 1 x N
			tmp = self.y_hat[i,:] - self.t[:,i]
			# tmp N x 2
			tmp = tmp.reshape(-1,1)*self.X
			# tmp - 2 x 1
			tmp = np.sum(tmp, axis = 0)
			res[i,:] = tmp
		return res

	def softmax(self, x = None):
		# w is 3 x 2
		# x is n x 2
		if x is None:
			tmp = np.exp(np.matmul(self.w,np.transpose(self.X)))
			sumation = np.sum(tmp, axis=0)
			#sumation +=1e+16
			#sumation += 1e-6
			tmp = tmp/sumation
		else:
			tmp = np.exp(np.matmul(self.w, x))
			sumation = np.sum(tmp, axis=0)
			#sumation[sumation > 1e+16] =1e+16
			#sumation += 1e-6
			tmp = tmp /sumation
		return tmp

	def predict(self, x):
		return np.argmax(self.softmax(x), axis=0)
class Linear_Model():
	def __init__(self,X,y):
		self.X = X
		self.y = y
		self.classes = np.unique(y).shape[0]
		self.MAX_ITERATIONS = 30
		self.t = self.construct_targets()
	def construct_targets(self):
		t = np.zeros((self.y.shape[0],self.classes))
		for i in range(self.y.shape[0]):
			t[i,int(self.y[i])] = 1
		return t
	def train(self):
		tmp = np.matmul(np.transpose(self.X),self.X)
		tmp = np.matmul(np.linalg.inv(tmp),np.transpose(X))
		tmp = np.matmul(tmp,self.t)
		self.w = tmp
		return
	def predict(self, x):
		tmp = np.matmul(np.transpose(self.w), x)
		return np.argmax(tmp)
ml = Linear_Model(X,y)
ml.train()

m = Logistic_Model(X, y)
m.train()

# meshgrid
u1, l1 = max(X[:,0]), min(X[:,0])
u2, l2 = max(X[:,1]), min(X[:,1])

x1 = np.linspace(l1,u1,50)
x2 = np.linspace(l2,u2,50)
xv, yv = np.meshgrid(x1, x2)
xv, yv = xv.reshape(-1,1), yv.reshape(-1,1)

fig = plt.figure()
col = ['r', 'c', 'g']
for i in range(xv.shape[0]):
	tmp = m.predict(np.array([xv[i],yv[i]]))
	plt.scatter(xv[i], yv[i], c= col[tmp[0]])
plt.show()

fig2 = plt.figure()
for i in range(y.shape[0]):
	plt.scatter(X[i,0],X[i,1], c= col[int(y[i])])
plt.show()

#fig3 = plt.figure()
#for i in range(xv.shape[0]):
#	tmp = ml.predict(np.array([xv[i],yv[i]]))
#	plt.scatter(xv[i], yv[i], c= col[tmp])
#plt.show()