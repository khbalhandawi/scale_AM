import numpy as np
from typing import Union, List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines

from .surrogateLib import KSModel

class scalability:

    def __init__(self, p: np.ndarray, m: np.ndarray, model: KSModel, key: str = ''):
        """
        Checks the scalability of a design problem

        Parameters
        ----------
        p : np.ndarray
            change effect vector
        m : np.ndarray
            monotonicity vector
        model : KSModel
            a surrogate model that return gradients
        key : str, optional
            key to label design scenario with, by default ''
        """
        assert m.ndim == 1 and p.ndim == 1, 'monotonicity and change effect must be one dimensional arrays'
        assert model.data_available, 'model must be trained'
        assert len(m) == model.n_outputs, 'monotonicity vector of length %i must match the \
            number of outputs %i from surrogate' %(len(m),model.n_outputs)
        assert len(p) == model.n_features, 'change effect vector of length %i must match the \
            number of variables %i in surrogate' %(len(p),model.n_features)

        self.m = m
        self.p = p
        self.model = model
        self.key = key

        cstrs = {}
        for i in range(self.model.n_features):
            for j in range(self.model.n_outputs):
                cstrs[(i,j)] = None
        cstrs['all'] = None
        self.cstrs:Dict[Union[str,Tuple(int,int)],np.ndarray] = cstrs

    def compute_scalability(self, xi: List[int], yi: int, nominal: float = None, n_levels: int = 100) \
        -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        computes scalability at the give cross-section

        Parameters
        ----------
        xi : List[int]
            _description_
        yi : int
            _description_
        nominal : float, optional
            _description_, by default None
        n_levels : int, optional
            _description_, by default 100
        """
        x,y,z,_,f,grad_f = self.model.cross_section(xi,yi,nominal,n_levels)
        j = grad_f
        pjm = np.diag(self.p)[None,...,] @ j.swapaxes(0,2) @ np.diag(self.m)[None,...,]

        for i in range(self.model.n_features):
            for j in range(self.model.n_outputs):
                self.cstrs[(i,j)] = pjm[:,i,j] < 0

        self.cstrs['all'] = np.any(pjm < 0,axis=(1,2))

        return x,y,z,f,grad_f

    def view(self, xi: List[int], yi: int, nominal: float = None, label_x1: str = None, 
        label_x2: str = None, label_y: str = None, n_levels: int = 100, 
        cstrs: List[Union[str,Tuple[int,int]]] = ['all',], show_training: bool = False):
        """
        Views the surrogate model overlayed with scalability

        Parameters
        ----------
        xi : List[int]
            _description_
        yi : int
            _description_
        nominal : float, optional
            _description_, by default None
        label_x1 : str, optional
            _description_, by default None
        label_x2 : str, optional
            _description_, by default None
        label_y : str, optional
            _description_, by default None
        n_levels : int, optional
            _description_, by default 100
        cstrs : List[Union[str,Tuple, optional
            the tuple index of the Jacobian constraint to plot, by default ['all']
        show_training : bool, optional
            whether to show training data, by default False
        """

        assert all([ci in self.cstrs.keys() for ci in cstrs]), 'invalid keys provided, \
            make sure i and j do not exceed n_features - 1 and n_outputs - 1, respectively'

        x,y,_,_,_ = self.compute_scalability(xi,yi,nominal,n_levels)
        fig,ax = self.model.view(xi,yi,nominal,label_x1,label_x2,label_y,n_levels,handles=True,show_training=show_training)
        #======================== NONLINEAR CONSTRAINTS ============================#	
        # Nonlinear constraints
        handles = []; labels = []
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color'][1:]
        hatches=['.', '/', '\\', '\\\\', '*']

        for i,ci in enumerate(cstrs):
            color = colors[i % len(colors)]
            hatch = hatches[i % len(hatches)]
            cstr_plot = np.reshape(self.cstrs[ci], np.shape(x))

            plt.contour(x, y, cstr_plot, levels=[-20, 0, 20], colors=color, linestyles='-')
            cf = plt.contourf(
                x, y, cstr_plot, levels=[-20, 0, 20], colors='none',
                hatches=[hatch, None],
                extend='lower',
            )

            # https://github.com/matplotlib/matplotlib/issues/2789/#issuecomment-604599060
            for i, collection in enumerate(cf.collections):
                collection.set_edgecolor(color)

            a_hatch = patches.Rectangle((20,20), 20, 20, linewidth=2, edgecolor=color, facecolor='none', fill='None', hatch=3*hatch)
            handles += [a_hatch]
            labels += [str(ci)]

        a_data = mlines.Line2D([], [], color='black', marker='.', markersize=5, linestyle='')
        handles += [a_data]
        labels += ["training data"]

        lx = fig.legend(handles, labels, loc='upper center', ncol=3, fontsize = 14)

        plt.show()
