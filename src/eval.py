import pandas as pd 
from pandas.api.types import is_numeric_dtype
import copy 
import numpy as np
from scipy.stats import ks_2samp, stats 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import auc, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
#from optbinning import OptimalBinning

pd.set_option("mode.chained_assignment", None)

def calculate_ks(x, y, pipeline, TARGET):
    
    """ Function to calculate de KS test for the dataset """
    
    data = x.copy()
    
    pred_prob = pipeline.predict_proba(data)
    pred_prob = pred_prob[:,1]
    
    data['pred_prob'] = pred_prob
    data[TARGET] = y
    
    ks = ks_2samp(data.loc[data[TARGET] == 0, 'pred_prob'],
                  data.loc[data[TARGET] == 1, 'pred_prob'])[0]
    ks = np.round(ks,3)
    
    return ks

def iv_woe(data, target, bins=10, show_woe=False):
    
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    #Extract Column Names
    cols = data.columns
    
    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d0 = d0.astype({"x": str})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Non-Events']/d['% of Events'])
        d['IV'] = d['WoE'] * (d['% of Non-Events']-d['% of Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)
    
    newDF = newDF.sort_values(by="IV", ascending=False).reset_index(drop=True)
        
    return newDF

class BinsType:
    
    QCUT = 0
    
class Bins:
    
    """
        Generate Bins
    """
    
    def generate_kbins(self, scores: np.array, q=10, strategy = 'kmeans'):
        
        """
            Generate score bins 
            
            Args:
                score: list of scores from the model.
                labels: bins labels.
                q: quantity of bins.
            
            Return:
                Bins generated.
        """

        discretizer = KBinsDiscretizer(
            n_bins=q, encode='ordinal', strategy = strategy, subsample=None
        )
        discretizer.fit(scores.reshape(-1,1))
        bins = list(discretizer.bin_edges_[0])
        
        return bins
    
    def generate(
        self,
        scores: np.array,
        bin_type=BinsType.QCUT,
        q=10,
        y: np.array = None,
    ):
        """ 
            Generate score bins
            
            Args:
                scores: scores one dimensional array.
                bin_type: QCUT 
                q: quantity of bins.
                y: targets array.
                
            Return:
                Bins
        """
        
        if bin_type == BinsType.QCUT:
            return self.generate_kbins(scores, q, strategy='quantile')
        
        raise ValueError("Error")

class Modelplots:
    
    """
        Graph plots to analyse the performance of the model
    """
    
    def _get_axis(self, ax):
        with_ax = True
    
        if ax is None:
            _, ax = plt.subplots()
            with_ax = False

        return ax, with_ax  

    def plot_roc(
        self, y: np.array, pred_prob: np.array, ax = None, title = "AUC Curve"
    ):
        
        """
            Plot of the AUC roc curve 
            
            Args:
            y: target series.
            pred_prob: predicted probability
            ax: axis of matplotlib
            title: title of the plot
        """
        ax, with_ax = self._get_axis(ax)

        fpr,tpr, _ = roc_curve(y, pred_prob)
        ax.plot(fpr,tpr)
        ax.plot(np.array([0,1]), np.array([0,1]))
        
        ax.set_title(f"{title}: %0.3f" % np.round(auc(fpr,tpr),3))
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        
        if not with_ax:
            plt.show()
    
    def plot_calibration(
        self,
        data: pd.DataFrame,
        target: str,
        probability: str = 'pred_prob',
        decil: str = 'decil_score',
        ax = None,
        title = 'Calibration curve',
        xlabel = "% of Event",
        ylabel = "Probability of default"
    ):
        
        """
        
        Plot of model calibration curve

        Args:

            data: dataset.
            target: target feature.
            probability: probability feature.
            decil: decil feature.
            ax: matplotlib axis.
            title: title of plot.
            xlabel: label of x.
            ylabel: label of y.
        """
    
        ax, with_ax = self._get_axis(ax)
        
        calibr_df = data.groupby(decil, observed = False)[[target, probability]].agg("mean")
        
        ax.plot([0,1], [0,1], linestyle = "--")
        
        ax.plot(calibr_df[target], calibr_df[probability], marker = ".")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if not with_ax:
            plt.show()
            
    def generate_model_plots(
        self,
        x:pd.DataFrame,
        y:np.array,
        pipeline: Pipeline,
        bins: list = None,
        labels: list = None,
        normalize: bool = False,
        scaler = None
    ):
        
        """
            Function to calculate de probability of default of the model, the bins and the execute the plot functions
            
            Args:
        
        """
        
        data = x.copy()
        
        pred_prob = pipeline.predict_proba(x)
        pred_prob_ones = pred_prob[:,1]
        
        if normalize:
            pred_prob_ones = scaler.transform(pred_prob_ones)
            
        data['target'] = y
        data['pred_prob'] = pred_prob_ones 
        
        if bins is None:
            gen_bins = Bins()
            bins = gen_bins.generate(pred_prob_ones, BinsType.QCUT, q=10)
            
        if labels is None:
            labels = [x for x in range(1, len(bins))]
            
        data['decil_score'] = pd.cut(
            data['pred_prob'], bins = bins, labels = labels, duplicates="drop"
        )
        
        _, axes = plt.subplots(figsize=(10, 6), nrows=1, ncols=2)

        self.plot_roc(y, pred_prob_ones, axes[0])  # Acessando o primeiro subplot
        
        self.plot_calibration(
            data, "target", probability="pred_prob", decil="decil_score", ax=axes[1]  # Acessando o segundo subplot
        )
        
        plt.show()