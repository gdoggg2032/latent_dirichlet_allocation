import numpy as np
import numpy.random
from collections import Counter




def fakeDataGenerate(K, D, V, Dlen):

	# answer is here

	# phi is topic-word combination

	phi = np.random.uniform(0, 1, (K, V))
	phi = phi / (np.sum(phi, 1).T)[:, np.newaxis]
	theta = np.random.uniform(0, 1, (D, K))
	theta = theta / (np.sum(theta, 1).T)[:, np.newaxis]


	# phi = np.array( # theta[t, w]
	# 	[	[0.1, 0.1, 0.4, 0.4],
	# 		[0.5, 0.5, 0.0, 0.0]])

	# # theta is topic-document combination
	# theta = np.array( # phi[d, t]
	# 	[	[0.1, 0.9],
	# 		[0.1, 0.9],
	# 		[0.1, 0.9],
	# 		[0.9, 0.1],
	# 		[0.9, 0.1]])

	Docs = []
	for d in range(0, D):
		doc = []
		for i in range(0, Dlen):
			t_tmp = numpy.random.choice(range(0, K), p=theta[d, :])
			w = numpy.random.choice(range(0, V), p=phi[t_tmp, :])
			doc.append(w)
		Docs.append(doc)

	return Docs, phi, theta

class LDA:

	def __init__(self, K, alpha, beta, docs, V, smartinit=None):
		self.K = K
		self.alpha = alpha
		self.beta = beta
		self.docs = docs
		self.V = V
		self.Dlen = Dlen
		self.z_m_n = np.zeros((len(self.docs), Dlen)) # topic of the n-th word in the m-th document
		self.n_m_z = np.zeros((len(self.docs), K)) + alpha
		self.n_m = np.zeros(len(self.docs)) + K * alpha
		# word count of each doc and topic
		self.n_z_w = np.zeros((K, V)) + beta
		self.n_z = np.zeros(K) + V*beta

		self.N = 0

		for m, doc in enumerate(docs):

			self.N += len(doc)

			#z_n = [] # topic assignment in this doc
			self.n_m[m] = len(doc)

			for n, t in enumerate(doc):

				z = numpy.random.randint(0, K)

				#z_n.append(z)
				self.z_m_n[m, n] = z
				self.n_m_z[m, z] += 1.
				self.n_z_w[z, t] += 1.
				self.n_z[z] += 1.

	def inference(self):
		for m, doc in enumerate(self.docs):
			for n, w in enumerate(doc):
				z = self.z_m_n[m, n]

				# resampling
				self.n_m_z[m, z] -= 1.
				self.n_z_w[z, w] -= 1.
				self.n_z[z] -= 1.
				self.n_m[m] -= 1.

				p_z = self.n_z_w[:, w] / self.n_z * self.n_m_z[m, :]

				new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()

				self.n_z[new_z] += 1.
				self.n_m_z[m, new_z] += 1.
				self.n_z_w[new_z, w] += 1.
				self.n_m[m] += 1.
				self.z_m_n[m, n] = new_z

	def phi(self):

		return self.n_z_w / self.n_z[:, np.newaxis]

	def theta(self):

		return self.n_m_z / self.n_m[:, np.newaxis]

	def perplexity(self):

		phi = self.phi()
		theta = self.theta()

		log_per = 0

		for m, doc in enumerate(self.docs):
			for n, w in enumerate(doc):
				log_per -= np.log(np.dot(phi[:, w], theta[m, :]))

		return np.exp(log_per / self.N)

	
	def result(self, docs):

		return ""



	def training(self, iteration):


		pre_perp = self.perplexity()

		print "initial perplexity=", pre_perp

		for t in range(iteration):
			self.inference()

			perp = self.perplexity()

			print "iter ", t, "perplexity=", perp






if __name__ == "__main__":

	V = 4 # vocabulary size
	K = 2 # number of topics
	D = 5 # number of documents
	Dlen = 1000 # word numbers for each document

	alpha = 1.
	beta = 10.

	iteration = 100


	docs, phi, theta = fakeDataGenerate(K, D, V, Dlen)

	lda = LDA(K, alpha, beta, docs, V)


	lda.training(iteration)



	print lda.phi()
	print lda.theta()


	print phi
	print theta
	







