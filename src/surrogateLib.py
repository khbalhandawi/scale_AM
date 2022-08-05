import numpy as np
import pickle
from typing import Union, List
import matplotlib.pyplot as plt

from .utilities import check_folder
from .DOELib import Design, scaling

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

    def __init__(self, key: str = ''):
        self.n_features = None
        self._X = None
        self._Y = None
        self._bandwidth = np.empty(0)
        self._lb = np.empty(0)
        self._ub = np.empty(0)
        self.B = np.empty(0)
        self.data_available = False
        self.key = key

    @property
    def X(self) -> np.ndarray:
        """
        Returns the X training inputs unscaled

        Returns
        -------
        np.ndarray
            Y training outputs
        """

        return scaling(self._X,self.lb,self.ub,2)

    @X.setter
    def X(self, x:np.ndarray):
        """
        Sets the X training inputs to be scaled by their upper and lower bounds

        Parameters
        ----------
        x : np.ndarray
            Y training outputs
        """

        # scales the values
        self._X = scaling(x,self.lb,self.ub,1)

    @property
    def Y(self) -> np.ndarray:
        """
        Returns the Y training outputs

        Returns
        -------
        np.ndarray
            Y training outputs
        """

        return self._Y

    @Y.setter
    def Y(self, y:np.ndarray):
        """
        Sets the Y training outputs to be a 2 dimensional array

        Parameters
        ----------
        y : np.ndarray
            Y training outputs
        """

        # convert to floats
        if y.ndim == 1:
            self._Y = y[:,None]
        else:
            self._Y = y  # reshape 1D arrays to 2D

        assert self._X.shape[0] == self._Y.shape[0], 'number of rows in input and output training data must be equal'

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
        lb : Union[int,float,np.ndarray]
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
        ub : Union[int,float,np.ndarray]
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
        pred,_ = self.predict(X)

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
        return np.linalg.norm((self.predict(X)[0] - Y))**2 # Compute loss

    def train(self, X:np.ndarray, Y:np.ndarray, 
        bandwidth:Union[float,List[float],np.ndarray] = 1e-3, 
        lb:np.ndarray = None, ub:np.ndarray = None):
        """
        `trains` the kernel regression model (just stores the training data)

        Parameters
        ----------
        X : np.ndarray
            input training data
        Y : np.ndarray
            output training data
        bandwidth : Union[float,List[float],np.ndarray], optional
            _description_, by default 1e-3
        lb : np.ndarray, optional
            sets the lower bounds of the model inputs, by default None
        ub : np.ndarray, optional
            sets the upper bounds of the model inputs, by default None
        """
        # Set the bounds

        assert X.ndim == 2, 'X: expected 2 dimensions got %i dimensions' %X.ndim

        self.n_features = X.shape[1]
        self.lb = np.min(X,axis=0) if lb is None else lb
        self.ub = np.max(X,axis=0) if ub is None else ub
        self.X = X
        self.Y = Y
        self.bandwidth = bandwidth
        self.B=np.diag(1./self._bandwidth) # bandwidth matrix
        self.data_available = True

    def view(self, xi: List[int], yi: int, nominal: Union[float,np.ndarray] = None, label_x1: str = None, label_x2: str = None,
                    label_y: str = None, n_levels: int = 100, folder: str = '', file: str = None,
                    img_format: str = 'pdf'):
        """
        Shows the estimated performance 

        Parameters
        ----------
        xi : list[int]
            index of the margin node values to be viewed on the plot
        yi : int
            index of the performance parameter to be viewed on the plot
        nominal : Union[float,np.ndarray], optional
            default value of features outside viewing scope, if None then default is the mid point of the model bounds
            by default None
        label_x1 : str, optional
            axis label of excess value 1if not provided uses the key of MarginNode, 
            by default None
        label_x2 : str, optional
            axis label of excess value 2, if not provided uses the key of MarginNode, 
            by default None
        label_y : str, optional
            z-axis label of performance parameter, if not provided uses the key of Performance object, 
            by default None
        n_levels : int, optional
            resolution of the plot (how many full factorial levels in each direction are sampled), 
            by default 100
        folder : str, optional
            folder in which to store image, by default ''
        file : str, optional
            name of image file, if not provide then an image is not saved, by default None
        img_format : str, optional
            format of the image to be stored, by default 'pdf'
        """
        if nominal is not None:
            if isinstance(nominal, (int, float)):
                nominal = nominal * np.ones(self.n_features)
            else:
                assert nominal.ndim == 1, 'expected %i dimensions, got %i dimensions' %(1,nominal.ndim)
                assert len(nominal) == self.n_features, 'expected %i parameters, got %i parameters' %(self.n_features,len(nominal))
        else:
            nominal = ((self.ub - self.lb)/2)  + self.lb

        # Plot the result in 2D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot()

        sampling_vector = np.ones(self.n_features)
        # only sample the selected variables while holding the other variables at their nominal values
        sampling_vector[xi] = n_levels

        lb_x, ub_x = np.empty(0), np.empty(0)
        for i in range(self.n_features):
            lb_x = np.append(lb_x, nominal[i])
            ub_x = np.append(ub_x, nominal[i] + 1e-3)
        lb_x[xi] = self.lb[xi]
        ub_x[xi] = self.ub[xi]

        plot_grid = Design(lb_x, ub_x, sampling_vector, 'fullfact')
        estimate,_ = self.predict(plot_grid.unscale())

        x = plot_grid.unscale()[:, xi[0]].reshape((n_levels, n_levels))
        y = plot_grid.unscale()[:, xi[1]].reshape((n_levels, n_levels))
        z = estimate[:, yi].reshape((n_levels, n_levels))

        label_x1 = 'x%i' %(xi[0]+1) if label_x1 is None else label_x1
        label_x2 = 'x%i' %(xi[1]+1) if label_x2 is None else label_x2
        label_y = 'f' if label_y is None else label_y

        ax.contourf(x, y, z, cmap=plt.cm.jet, )
        # ax.plot(self.xt[:50,xi[0]],self.xt[:50,xi[1]],'.k', markersize = 10) # plot DOE points for surrogate (first 50 only)
        ax.plot(self.X[:,xi[0]],self.X[:,xi[1]],'.k') # plot DOE points for surrogate

        ax.set_xlabel(label_x1)
        ax.set_ylabel(label_x2)

        cbar = plt.cm.ScalarMappable(cmap=plt.cm.jet)
        cbar.set_array(z)

        boundaries = np.linspace(np.min(estimate[:, yi]), np.max(estimate[:, yi]), 51)
        cbar_h = fig.colorbar(cbar, boundaries=boundaries)
        cbar_h.set_label(label_y, rotation=90, labelpad=3)

        if file is not None:
            # Save figure to image
            check_folder('images/%s' % folder)
            self.fig.savefig('images/%s/%s.%s' % (folder, file, img_format),
                             format=img_format, dpi=200, bbox_inches='tight')

        plt.show()

    def save(self, filename: str = 'KS'):
        """
        saves all the model parameters and training data

        Parameters
        ----------
        filename : str, optional
            Prefix of filename to save, by default 'KS'
        """
        with open('%s_%s.pkl' %(filename,self.key),'wb') as fid:
            pickle.dump(self.n_features,fid)
            pickle.dump(self.lb,fid)
            pickle.dump(self.ub,fid)
            pickle.dump(self.X,fid)
            pickle.dump(self.Y,fid)
            pickle.dump(self.bandwidth,fid)
            pickle.dump(self.B,fid)
            pickle.dump(self.data_available,fid)

    def load(self, filename: str = 'KS'):
        """
        saves all the model parameters and training data

        Parameters
        ----------
        filename : str, optional
            Prefix of filename to load, by default 'KS'
        """
        with open('%s_%s.pkl' %(filename,self.key),'rb') as fid:
            self.n_features = pickle.load(fid)
            self.lb = pickle.load(fid)
            self.ub = pickle.load(fid)
            self.X = pickle.load(fid)
            self.Y = pickle.load(fid)
            self.bandwidth = pickle.load(fid)
            self.B = pickle.load(fid)
            self.data_available = pickle.load(fid)