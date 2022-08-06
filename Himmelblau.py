from src.surrogateLib import KSModel
from src.DOELib import Design
from src.scaleLib import scalability
import numpy as np
import pandas as pd

def himmelblau(X):
    x = X[:,0]; y = X[:,1]

    f = (((x**2) + y - 11)**2) + ((x + (y**2) -7)**2)
    
    f_grad = np.zeros((X.shape[1],1,X.shape[0]))
    for n in range(len(x)):
        f_grad[:,:,n] = np.array([
            4*x[n]*((x[n]**2) + y[n] - 11) + 2*(x[n] + (y[n]**2) -7),
            2*((x[n]**2) + y[n] - 11) + 4*y[n]*(x[n] + (y[n]**2) -7)
            ])[:,None]
    
    return f[:,None], f_grad

bounds = np.array([[-5, 5],[-5, 5]])
lob = bounds[:,0]; upb = bounds[:,1]
doe = Design(lob,upb,2000,'LHS')

f, f_grad = himmelblau(doe.unscale())

df=pd.DataFrame(np.hstack((doe.unscale(),f)), columns= ['x%i'%(i+1) for i in range(doe.unscale().shape[1])] + ['f%i'%(i+1) for i in range(f.shape[1])]) 
df.to_csv('himmelblau.csv')

model = KSModel('PSpace')
model.train(doe.unscale(),f,1,lb=lob,ub=upb)
# model.view([0,1],0)

# scalability assessment
p = np.array([1,-1])
m = np.array([1,])
s = scalability(p,m,model,'Himmelblau')
s.view([0,1],0,cstrs=[(0,0),(1,0),])
