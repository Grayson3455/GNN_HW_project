import numpy as np
from matplotlib import pyplot as plt

# Problem 1

# paras
K  = 1
d  = [.5,.5]
R  = 3
J  = 8
lambda_max = 2

a = R * np.log(lambda_max) / (J - R + 1)


def Indicator(a,b):
	if -a >= 0 and -a < b:
		return 1
	else:
		return 0 

# define half cosine kernel
def half_cosine_kernel(lmd):

	# init
	g = 0

	for k in range(K + 1):

		g += d[k] * np.cos( 2. * np.pi * k * (lmd/a + 0.5) ) * Indicator(lmd, a)

	return g

# define spectral filter
def spec_filter(j,lmd):

	def j_more_than_2(j,lmd):

		return half_cosine_kernel(np.log(lmd) - a*(j-1)/R )

	if j ==  1:
		
		g_inside = R*d[0]**2 + R/2* d[1]**2

		for i in range(2,J+1):
			g_inside -= abs(j_more_than_2(i, lmd))**2 

		return np.sqrt(g_inside + 1e-12)

	if j>=2 and j<=J:
		return j_more_than_2(j, lmd)



if __name__=="__main__":

	# start to plot
	plt.figure(figsize=(10,10))

	num   = 1000
	g_hat = np.zeros((J, num))
	LMD   = np.linspace(1e-12,lambda_max,num)

	for j in range(1,J+1):
	#for j in range(1,2):
		for i, lmd in enumerate(LMD):
			g_hat[j-1,i] = spec_filter(j, lmd)

		plt.plot(LMD, g_hat[j-1,:], label = '$j$='+str(j),linewidth=2)

	fs = 24
	plt.rc('text',  usetex=True)
	plt.xlabel(r'$\lambda$', fontsize=fs)
	plt.ylabel(r'$\widehat{g}_j$', fontsize=fs)
	plt.tick_params(labelsize=fs-2)
	plt.legend(fontsize=fs-3)
	plt.savefig('P1.pdf')

