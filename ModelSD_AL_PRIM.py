import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import gpytorch
import torch
import prim
import random
from Model import model


###### Load initial LHScd data
with open('Data/output200.pickle', 'rb') as handle:
    y = pickle.load(handle)
with open('Data/X200.pickle', 'rb') as handle: #labeled points
    X = pickle.load(handle)


#### Active learning loop
n_iter= 200 # number of iterations in the active learning loop
k= 1 # number of elements to query in each loop

X_labeled = X
y_labeled = y


y_labeled= y_labeled.astype(float)
y_labeled = torch.from_numpy(y_labeled)
X_labeled = torch.from_numpy(X_labeled)


# Define a Gaussian process regression model with GPYtorch
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model1 = GPRegressionModel(X_labeled, y_labeled, likelihood)

# Train the model on the initial labeled dataset
model1.train()
likelihood.train()
optimizer = torch.optim.Adam(model1.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model1)

for i in range(5000):
    optimizer.zero_grad()
    output = model1(X_labeled)
    loss = -mll(output, y_labeled)
    print('Iter: ' , i , 'Loss:  %.3f'  %loss.item())
    loss.backward()
    optimizer.step()

#These are the sampling points that I return at the end
X_new_labeled = X_labeled
y_new_labeled = y_labeled


with open('Data/X800.pickle', 'rb') as handle:
    X_test = pickle.load(handle)

X_test = torch.from_numpy(X_test)


for i in range(n_iter):

    # Use the model to make predictions on the unlabeled dataset
    model1.eval()
    likelihood.eval()

    #Make predictions of the posterior in another LHS 
    with torch.no_grad():
        output = likelihood(model1(X_test))

    output = output.mean
    percentil_20 = np.percentile(output, 20)

    Y= [1 if output[i] < percentil_20 else 0 for i in range(len(output))]
    Y= np.array(Y).astype(float)
    print("Number of vulnerable scenarios: ", sum(Y), " where the leasure trips are less than ", percentil_20, "in the whole population")

    #perform PRIM in the posterior LHS
    p = prim.Prim(X_test, Y, threshold=0.3, threshold_type=">")
    box = p.find_box()

    df = box.limits

    #create an array to store the contrain dimesions
    dimension = np.zeros([(X.shape[-1]),2])
    dimension[:,1] = 1

    #store the restricted dimension from PRIM for each dimension
    for i in range(len(df)):
        dimension[int(df.index[i]),0] = df.iloc[i,0] #minumum
        dimension[int(df.index[i]),1] = df.iloc[i,1] #maximum


    #sample a random point within the restricted dimensions
    new_X_point = np.zeros(X.shape[-1])
    for i in range(X.shape[-1]):
        new_X_point[i] = random.uniform(dimension[i,0],dimension[i,1])

    #compute the value of the new point given the model
    new_Y_point = model(new_X_point)
    new_X_point = np.array([new_X_point])

    #convert array to tensor
    new_X_point = torch.from_numpy(new_X_point)

    #Add point to the X train samples
    X_new_labeled = torch.cat([X_new_labeled, new_X_point])
    y_new_labeled = y_new_labeled.numpy()
    y_new_labeled = np.append(y_new_labeled, new_Y_point)
    y_new_labeled = torch.from_numpy(y_new_labeled)

    print('X_new_labeled size: ', X_new_labeled.size())
    print('y_new_labeled size: ', y_new_labeled.size())

    # Re train the model on the updated labeled dataset
    model1.set_train_data(X_new_labeled, y_new_labeled, strict=False)
    model1.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model1.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model1)

    for i in range(300):
        optimizer.zero_grad()
        output = model1(X_new_labeled)
        loss = -mll(output, y_new_labeled)
        print('Iter: ' , i , 'Loss:  %.3f'  %loss.item())
        loss.backward()
        optimizer.step()


#Get the final value of the posterior for the another LHS samples
model1.eval()
likelihood.eval()
print('Final loss:',loss)


with torch.no_grad():
    output = likelihood(model1(X_test))

#devuelve todos los puntos usados en la posterior y la posterior evaluada en LHS lista para aplicar PRIM
with open('Output_AL_PRIM.pickle', 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('X_AL_PRIM.pickle', 'wb') as handle:
    pickle.dump(X_new_labeled, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('Y_AL_PRIM.pickle', 'wb') as handle:
    pickle.dump(y_new_labeled, handle, protocol=pickle.HIGHEST_PROTOCOL)



