import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
from .pytorchtools import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
from ..MMI.IC.AIC import TableEntropy
import os

from .mine import Mine, MineNet, sample_joint_marginal, sample_joint_marginal

class Mine_ent(Mine):
    def __init__(self, lr, batch_size, patience=int(20), iter_num=int(1e+3), log_freq=int(100), avg_freq=int(10), ma_rate=0.01, prefix="", verbose=True, resp=0, cond=[1], log=True, objName="", ParamName="", ParamValue=np.inf, X_GroundTruth=np.inf):
        """
        can only support bivariate mutual information now
        """
        obj_XY = "{0}_XY".format(objName)
        super().__init__(lr, batch_size, patience, iter_num, log_freq, avg_freq, ma_rate, prefix, verbose, resp, cond, log, obj_XY, ParamName, ParamValue, X_GroundTruth)
        self.marginal_mode = 'unif'

        obj_X = "{0}_X".format(objName)
        self.Mine_resp = Mine(lr, batch_size, patience, iter_num, log_freq, avg_freq, ma_rate, prefix, verbose, resp, [], log, obj_X, ParamName, ParamValue, X_GroundTruth)
        self.Mine_resp.marginal_mode = 'unif'

        obj_Y = "{0}_Y".format(objName)
        self.Mine_cond = Mine(lr, batch_size, patience, iter_num, log_freq, avg_freq, ma_rate, prefix, verbose, cond[0], [], log, obj_Y, ParamName, ParamValue, X_GroundTruth)
        self.Mine_cond.marginal_mode = 'unif'

    def predict(self, X):
        """[summary]
        
        Arguments:
            X {[numpy array]} -- [N X 2]

        Return:
            mutual information estimate
        """
    
        HXY = super(Mine_ent, self).predict(X)
        HX = self.Mine_resp.predict(X)
        HY = self.Mine_cond.predict(X)
        return HX + HY - HXY

    def setVaryingParamInfo(self, ParamName, ParamValue, X_gt):
        self.ParamName = ParamName
        self.ParamValue = ParamValue
        self.X_GroundTruth = X_gt

    def getTrainCurve(self, objMine, ax):
        ax.plot(range(1,len(self.avg_train_mi_lb)+1),self.avg_train_mi_lb, label='Training Loss')
        ax.plot(range(1,len(self.avg_valid_mi_lb)+1),self.avg_valid_mi_lb,label='Validation Loss')
        # find position of lowest validation loss
        minposs = self.avg_valid_mi_lb.index(min(self.avg_valid_mi_lb))+1 
        ax.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
        ax.grid(True)
        ax.legend()
        ax.set_title('train curve of HXY')
        return ax


    def savefigAli(self, X, X_est):
        if len(self.cond) > 1:
            raise ValueError("Only support 2-dim or 1-dim")
        fig, ax = plt.subplots(3,4, figsize=(90, 90))
        #plot Data
        axCur = ax[0,0]
        axCur.scatter(X[:,self.resp], X[:,self.cond], color='red', marker='o')
        axCur.set_title('scatter plot of data')

        #plot training curve
        axCur = ax[0,1]
        axCur = super(Mine_ent, self).getTrainCurve(axCur)
        axCur.set_title('train curve of HXY')
        axCur = ax[1,1]
        axCur = self.Mine_resp.getTrainCurve(axCur)
        axCur.set_title('train curve of HX')
        axCur = ax[2,1]
        axCur = self.Mine_cond.getTrainCurve(axCur)
        axCur.set_title('train curve of HXY')

        # Trained Function contour plot
        Xmin = min(X[:,0])
        Xmax = max(X[:,0])
        Ymin = min(X[:,1])
        Ymax = max(X[:,1])
        x = np.linspace(Xmin, Xmax, 300)
        y = np.linspace(Ymin, Ymax, 300)
        xs, ys = np.meshgrid(x,y)
        # Z = [self.mine_net(torch.FloatTensor([[xs[i,j], ys[i,j]]])).item() for j in range(ys.shape[0]) for i in range(xs.shape[1])]
        # Z = np.array(Z).reshape(xs.shape[1], ys.shape[0])
        # # x and y are bounds, so z should be the value *inside* those bounds.
        # # Therefore, remove the last value from the z array.
        # Z = Z[:-1, :-1]
        # z_min, z_max = -np.abs(Z).max(), np.abs(Z).max()
        # c = ax[2].pcolormesh(xs, ys, Z, cmap='RdBu', vmin=z_min, vmax=z_max)
        # # set the limits of the plot to the limits of the data
        # ax[0,2].axis([xs.min(), xs.max(), ys.min(), ys.max()])
        axCur = ax[0,2]
        axCur, HXY, c = super(Mine_ent, self).getHeatMap(axCur, xs, ys, sampleNum=5)
        fig.colorbar(c, ax=axCur)
        axCur.set_title('heatmap T(X,Y)')

        axCur = ax[0,3]
        axCur, _, c = super(Mine_ent, self).getHeatMap(axCur, xs, ys, Z=HXY)
        fig.colorbar(c, ax=axCur)
        axCur.set_title('heatmap H(X,Y)')

        axCur = ax[1,2]
        axCur, HX = self.Mine_resp.getResultPlot(axCur, x, sampleNum=5)
        axCur.set_title('plot of T(X)')

        axCur = ax[1,3]
        axCur, _ = self.Mine_resp.getResultPlot(axCur, x, Z=HX)
        axCur.set_title('plot of H(X)')

        axCur = ax[2,2]
        axCur, HY = self.Mine_cond.getResultPlot(axCur, y, sampleNum=5)
        axCur.set_title('plot of T(Y)')

        axCur = ax[2,3]
        axCur, _ = self.Mine_resp.getResultPlot(axCur, y, Z=HY)
        axCur.set_title('plot of H(Y)')

        axCur = ax[1,0]
        HX = HX[:-1]
        HY = HY[:-1]
        MI_XY = [HX[i]+HY[j]-HXY[i,j] for i in range(HX.shape[0]) for j in range(HY.shape[0])]
        MI_XY = np.array(MI_XY).reshape(HX.shape[0], HY.shape[0])
        axCur, _, c = super(Mine_ent, self).getHeatMap(axCur, xs, ys, MI_XY)
        fig.colorbar(c, ax=axCur)
        axCur.set_title('heatmap of MI_XY')


        # Plot result with ground truth
        axCur = ax[2,0]
        axCur.scatter(0, self.X_GroundTruth, edgecolors='red', facecolors='none', label='Ground Truth')
        axCur.scatter(0, X_est, edgecolors='green', facecolors='none', label="MINE_{0}".format(self.objName))
        axCur.set_xlabel(self.ParamName)
        axCur.set_ylabel(self.y_label)
        axCur.legend()
        axCur.set_title('MI of XY')
        figName = "{0}MINE".format(self.prefix)
        fig.savefig(figName, bbox_inches='tight')
        plt.close()
        
