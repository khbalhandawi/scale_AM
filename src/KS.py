import numpy as np
import pickle
from typing import Union, List

class KSModel:
    """
    A kernel smoothing model using Gaussian kernels: 
    Y = exp(-pi*l||Xi-Z||)Yi/exp(-pi*l||Xi-Z||), 
    """

    # Kernels
    # norm2 function -> x.nrows
    dist_norm_2 = lambda self,x,x1 : np.linalg.norm(np.matmul((x-x1),self.B),axis=1)
    # Gradient of kernel function (divide each column of (x-x1) by norm2(x-x1)-> [x.nrows * x.ncols]
    dist_norm_2_grad = lambda self,x,x1 : np.matmul((x-x1),self.B**2)/(self.dist_norm_2(x,x1)[:,None]) 
    # Gaussian kernel function -> x.nrows
    kerf = lambda self,x : np.exp(-np.pi*x)

    def __init__(self):
        self.n_features = None
        self.X = None
        self.Y = None
        self._bandwidth = np.empty(0)
        self._lb = np.empty(0)
        self._ub = np.empty(0)
        self.B = np.empty(0)
        self.data_available = False


    @property
    def lb(self) -> np.ndarray:
        """
        Returns the lower bound of the model supports

        Returns
        -------
        np.ndarray
            lower model supports
        """

        return self._lb

    @lb.setter
    def lb(self, lb:Union[int,float,np.ndarray]):
        """
        Sets the lb parameter and asserts it is compatible with model dimensions

        Parameters
        ----------
        Union[int,float,np.ndarray]
            lower pdf support(s)
        """

        # convert to floats
        if isinstance(lb, (int, float)):
            self._lb = lb * np.ones(self.n_features)
        else:
            self._lb = lb  # reshape 1D arrays to 2D

        # number of dimensions in lower and upper bounds must be equal
        assert self._lb.shape == (self.n_features,)

    @property
    def ub(self) -> np.ndarray:
        """
        Returns the upper bound of the model supports

        Returns
        -------
        np.ndarray
            upper pdf supports
        """

        return self._ub

    @ub.setter
    def ub(self, ub:Union[int,float,np.ndarray]):
        """
        Sets the ub parameter and asserts it is compatible with model dimensions

        Parameters
        ----------
        Union[int,float,np.ndarray]
            upper pdf support(s)
        """

        if isinstance(ub, (int, float)):
            self._ub = ub * np.ones(self.n_features)
        else:
            self._ub = ub  # reshape 1D arrays to 2D

        # number of dimensions in lower and upper bounds must be equal
        assert self._ub.shape == (self.n_features,)

    @property
    def bandwidth(self,) -> np.ndarray:
        """
        returns the bandwidth vector _bandwidth

        Returns
        -------
        np.ndarray
            the bandwidth vector
        """
        return self._bandwidth
    
    @bandwidth.setter
    def bandwidth(self,b:Union[float,List[float],np.ndarray]):
        """
        sets the bandwidth vector _bandwidth

        Parameters
        ----------
        b : Union[float,List[float],np.ndarray]
            value to use for the bandwidth
        """
        if isinstance(b,(int,float,np.floating)):
            self._bandwidth = b*np.ones(self.n_features)
        else:
            if isinstance(b,(np.ndarray)):
                assert b.ndim == 1, 'expected %i dimensions, got %i dimensions' %(1,b.ndim)
            assert len(b) == self.n_features, 'expected %i parameters, got %i parameters' %(self.n_features,len(b))
            self._bandwidth = np.array(b)

    def predict(self, Z:np.ndarray) -> List[np.ndarray]:
        """
        Predicts the function value and gradient at the locations Z
        Z must be of shape [n_rows * n_cols], where n_cols must be equal to the 
        number of cols of X provided during the `train` method

        Parameters
        ----------
        Z : np.ndarray
            input vector to approximate function at

        Returns
        -------
        List[np.ndarray]
            two arrays, the first is f(x) the second is nabla_f(x)
        """
        assert self.data_available, 'must provide data for prediction using the `train` method'
        assert Z.shape[1] == self.X.shape[1], \
            'expected %i dimensions, got %i dimensions' %(self.X.shape[1],Z.shape[1])

        # # [X.nrows * Z.nrows * X.cols]
        # diff = np.tile(Z,(5,1,1)) - self.X[:,:,None]
        # # [Z.nrows * X.cols]
        # xx_3D=np.linalg.norm(diff,axis=2)
        # # select neighbors using exp(-pi*norm2(x-xi))<5e-6
        # # [Z.nrows * X.cols]
        # idx = xx_3D < 3.8853
        # # [Z.nrows * X.cols]
        # z_3D=np.where(idx, self.kerf(xx_3D),0)

        # [Z.nrows * Y.ncols]
        self.F=np.zeros((Z.shape[0],self.Y.shape[1]))

        # [X.ncols * Y.ncols * Z.nrows]
        self.grad_F = np.zeros((self.Y.shape[1], self.X.shape[1], Z.shape[0]))

        # Loop through each regression point
        for k in range(Z.shape[0]):

            # [X.nrows * Z.ncols]
            x_p = np.tile(Z[k,:],(self.X.shape[0],1))

            # scaled deference from regression point
            # X.nrows
            xx=self.dist_norm_2(x_p,self.X)
            
            # select neighbors using exp(-pi*norm2(x-xi))<5e-6
            # X.nrows
            idx = xx < 3.8853
            # kernel function

            # X.nrows
            z=self.kerf(xx[idx])
            
            # regression
            S_phi = np.sum(z) # 1

            # [1 * Y.ncols]
            S_phi_y = np.dot(z,self.Y[idx,:])[None,:]

            # Y.ncols
            self.F[k,:] = S_phi_y / S_phi # a 1 x X vector

            #############################################
            # [X.nrows * X.ncols]
            xx_grad = self.dist_norm_2_grad(x_p,self.X)

            # [X.nrows * X.ncols]
            z_grad = -np.pi*xx_grad[idx,:] * z[:,None]

            # [1 * X.ncols]
            S_grad_phi = np.sum(z_grad,axis=0)[None,:]
            
            # [Y.ncols * X.ncols]
            S_grad_phi_y = np.dot(self.Y[idx,:].T,z_grad)

            # [X.ncols * Y.ncols]
            self.grad_F[:,:,k] = ( (S_phi * S_grad_phi_y) - (S_phi_y.T * S_grad_phi) ) / (S_phi**2) # an X x P Jacobian

        return self.F, self.grad_F

    def R2(self,X:np.ndarray, Y:np.ndarray) -> float:
        """
        Compute the R2 error

        Parameters
        ----------
        X : np.ndarray
            input array at which to evaluate surrogate
        Y : np.ndarray
            output array providing the ground truth values

        Returns
        -------
        float
            the coefficient of determination (R2)
        """
        pred = self.predict(X)

        RSS = np.linalg.norm((pred - Y))**2 # Compute loss
        TSS = np.linalg.norm((np.mean(Y) - Y))**2 # Compute loss

        loss = 1 - (RSS/TSS) # Compute loss
        return loss

    def error(self, X:np.ndarray, Y:np.ndarray, loss:str ='MSE') -> float:
        """
        Compute the squared error

        Parameters
        ----------
        X : np.ndarray
            input array at which to evaluate surrogate
        Y : np.ndarray
            output array providing the ground truth values
        loss : str, optional
            type of error metric. For now only `MSE` is available, by default 'MSE'

        Returns
        -------
        float
            the desired error metric
        """
        return np.linalg.norm((self.predict(X) - Y))**2 # Compute loss

    def train(self, X:np.ndarray, Y:np.ndarray, bandwidth:Union[float,List[float],np.ndarray] = 1e-3):
        """
        `trains` the kernel regression model (just stores the training data)

        Parameters
        ----------
        X : np.ndarray
            input training data
        Y : np.ndarray
            output training data
        """
        assert X.shape[0] == Y.shape[0], 'x and y have different rows.'
        self.X = X
        self.Y = Y
        self.n_features = self.X.shape[1]
        self.bandwidth = bandwidth

        # Scaling first
        self.B=np.diag(1./self._bandwidth) # bandwidth matrix

        self.data_available = True

    def save(self, filename='model.pkl'):
        with open(filename,'wb') as fid:
            pickle.dump(self.X,fid)
            pickle.dump(self.Y,fid)

    def load(self, filename):
        with open(filename,'rb') as fid:
            self.X = pickle.load(fid)
            self.Y = pickle.load(fid)