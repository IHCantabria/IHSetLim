import numpy as np
import xarray as xr
from datetime import datetime
from spotpy.parameter import Uniform
from IHSetLim import lim
from IHSetCalibration import objective_functions
from IHSetUtils import BreakingPropagation, ADEAN

class cal_Lim(object):
    """
    cal_Lim
    
    Configuration to calibfalse,and run the Lim et al. (2022) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path, **kwargs):

        self.path = path
        
        mkTime = np.vectorize(lambda Y, M, D, h: datetime(int(Y), int(M), int(D), int(h), 0, 0))

        cfg = xr.open_dataset(path+'config.nc')
        wav = xr.open_dataset(path+'wav.nc')
        ens = xr.open_dataset(path+'ens.nc')

        self.cal_alg = cfg['cal_alg'].values
        self.metrics = cfg['metrics'].values
        self.dt = cfg['dt'].values
        self.switch_Yini = cfg['switch_Yini'].values

        if self.cal_alg == 'NSGAII':
            self.n_pop = cfg['n_pop'].values
            self.generations = cfg['generations'].values
            self.n_obj = cfg['n_obj'].values
            self.cal_obj = objective_functions(self.cal_alg, self.metrics, n_pop=self.n_pop, generations=self.generations, n_obj=self.n_obj)
        else:
            self.repetitions = cfg['repetitions'].values
            self.cal_obj = objective_functions(self.cal_alg, self.metrics, repetitions=self.repetitions)

        self.Hs = wav['Hs'].values
        self.Tp = wav['Tp'].values
        self.Dir = wav['Dir'].values
        self.time = mkTime(wav['Y'].values, wav['M'].values, wav['D'].values, wav['h'].values)
        self.Eb = self.Hs**2 / 16
        
        self.depth = kwargs['depth']
        self.angleBathy = kwargs['angleBathy']
        self.D50 = kwargs['D50']
        self.mf = kwargs['mf']
        self.A = ADEAN(self.D50)
        
        breakType = "spectral"
        self.Hb, self.theb, self.depthb = BreakingPropagation(self.Hs, self.Tp, self.Dir,
                                                              np.repeat(self.depth, (len(self.Hs))), np.repeat(self.angleBathy, (len(self.Hs))),  breakType)
        
        self.Obs = ens['Obs'].values
        self.time_obs = mkTime(ens['Y'].values, ens['M'].values, ens['D'].values, ens['h'].values)

        self.start_date = datetime(int(cfg['Ysi'].values), int(cfg['Msi'].values), int(cfg['Dsi'].values))
        self.end_date = datetime(int(cfg['Ysf'].values), int(cfg['Msf'].values), int(cfg['Dsf'].values))

        self.split_data()

        if self.switch_Yini == 0:
            self.S0 = self.Obs_splited[0]
        self.Sm = np.mean(self.Obs_splited)
        
        cfg.close()
        wav.close()
        ens.close()
        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))
        self.idx_obs = mkIdx(self.time_obs)

        if self.switch_Yini == 0:
            def model_simulation(par):
                # ar = par['ar']
                kr = par['kr']
                mu = par['mu']
                Ymd = lim.lim(self.Hb_splited,
                                    self.dt,
                                    self.A,
                                    self.mf,
                                    kr,
                                    mu,
                                    self.Sm,
                                    self.S0)
                
                return Ymd[self.idx_obs_splited]
            
            self.params = [
                # Uniform('ar', 0.0001, 1.0),
                Uniform('kr', 0.0001, 1.0),
                Uniform('mu', 0.0001, 1.0)
            ]
            self.model_sim = model_simulation

        elif self.switch_Yini == 1:
            def model_simulation(par):
                # ar = par['ar']
                kr = par['kr']
                mu = par['mu']    
                S0 = par['S0']
                Ymd = lim.lim(self.Hb_splited,
                                    self.dt,
                                    self.A,
                                    self.mf,
                                    kr,
                                    mu,
                                    self.Sm,
                                    S0)
                
                return Ymd[self.idx_obs_splited]
                
            self.params = [
                # Uniform('ar', 0.0001, 1.0),
                Uniform('kr', 0.0001, 1.0),
                Uniform('mu', 0.0001, 1.0),
                Uniform('S0', np.min(self.Obs), np.max(self.Obs))
            ]
            self.model_sim = model_simulation


    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """
        idx = np.where((self.time < self.start_date) | (self.time > self.end_date))
        self.idx_validation = idx

        idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))
        self.idx_calibration = idx
        self.Hb_splited = self.Hb[idx]
        self.time_splited = self.time[idx]

        idx = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))
        self.Obs_splited = self.Obs[idx]
        self.time_obs_splited = self.time_obs[idx]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time_splited - t)))
        self.idx_obs_splited = mkIdx(self.time_obs_splited)
        self.observations = self.Obs_splited

        # Validation
        idx = np.where((self.time_obs < self.start_date) | (self.time_obs > self.end_date))[0]
        self.idx_validation_obs = idx
        if len(self.idx_validation)>0:
            mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
            if len(self.idx_validation_obs)>0:
                self.idx_validation_for_obs = mkIdx(self.time_obs[idx])
            else:
                self.idx_validation_for_obs = []
        else:
            self.idx_validation_for_obs = []


            
    # def split_data(self):
    #     """
    #     Split the data into calibration and validation datasets.
    #     """
    #     idx = np.where((self.time < self.start_date) | (self.time > self.end_date))
    #     self.idx_validation = idx

    #     idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))
    #     self.idx_calibration = idx
    #     self.Hb_splited = self.Hb[idx]
    #     self.time_splited = self.time[idx]

    #     idx = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))
    #     self.Obs_splited = self.Obs[idx]
    #     self.time_obs_splited = self.time_obs[idx]

    #     mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time_splited - t)))
    #     self.idx_obs_splited = mkIdx(self.time_obs_splited)
    #     self.observations = self.Obs_splited

    #     # Validation
    #     idx = np.where((self.time_obs < self.start_date) | (self.time_obs > self.end_date))
    #     self.idx_validation_obs = idx
    #     mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
    #     self.idx_validation_for_obs = mkIdx(self.time_obs[idx])
