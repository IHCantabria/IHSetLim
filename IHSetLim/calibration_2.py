import numpy as np
import xarray as xr
import fast_optimization as fo
import pandas as pd
from .lim import lim
import json
from IHSetUtils import BreakingPropagation, ADEAN

class cal_Lim_2(object):
    """
    cal_Lim_2
    
    Configuration to calibfalse,and run the Lim et al. (2022) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):

        self.path = path
     
        data = xr.open_dataset(path)
        
        cfg = json.loads(data.attrs['Lim'])

        self.cal_alg = cfg['cal_alg']
        self.metrics = cfg['metrics']
        self.switch_Yini = cfg['switch_Yini']
        self.switch_brk = cfg['switch_brk']
        if self.switch_brk == 1:
            self.bathy_angle = cfg['bathy_angle']
            self.breakType = cfg['break_type']
            self.depth = cfg['depth']
        self.D50 = cfg['D50']
        self.mf = cfg['mf']
        self.lb = cfg['lb']
        self.ub = cfg['ub']

        self.calibr_cfg = fo.config_cal(cfg)  

        if cfg['trs'] == 'Average':
            self.hs = np.mean(data.hs.values, axis=1)
            self.tp = np.mean(data.tp.values, axis=1)
            self.dir = np.mean(data.dir.values, axis=1)
            self.time = pd.to_datetime(data.time.values)
            self.Obs = data.average_obs.values
            self.Obs = self.Obs[~data.mask_nan_average_obs]
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.time_obs = self.time_obs[~data.mask_nan_average_obs]
        else:
            self.hs = data.hs.values[:, cfg['trs']]
            self.tp = data.tp.values[:, cfg['trs']]
            self.dir = data.dir.values[:, cfg['trs']]
            self.time = pd.to_datetime(data.time.values)
            self.Obs = data.obs.values[:, cfg['trs']]
            self.Obs = self.Obs[~data.mask_nan_obs[:, cfg['trs']]]
            self.time_obs = pd.to_datetime(data.time_obs.values)
            self.time_obs = self.time_obs[~data.mask_nan_obs[:, cfg['trs']]]
        
        self.start_date = pd.to_datetime(cfg['start_date'])
        self.end_date = pd.to_datetime(cfg['end_date'])
        
        data.close()

        if self.switch_brk == 0:
            self.hb = self.hs
        elif self.switch_brk == 1:
            self.hb, _, _ = BreakingPropagation(self.hs, self.tp, self.dir, np.repeat(self.depth, len(self.hs)), np.repeat(self.bathy_angle, len(self.hs)), self.breakType)
        
        self.hb[self.hb < 0.1] = 0.1

        self.split_data()

        if self.switch_Yini == 0:
            self.Yini = self.Obs_splited[0]


        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))

        
        self.idx_obs = mkIdx(self.time_obs)

        # Now we calculate the dt from the time variable
        mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
        self.dt = mkDT(np.arange(0, len(self.time)-1))
        mkDTsplited = np.vectorize(lambda i: (self.time_splited[i+1] - self.time_splited[i]).total_seconds()/3600)
        self.dt_splited = mkDTsplited(np.arange(0, len(self.time_splited)-1))

        self.Sm = np.mean(self.Obs_splited)
        self.A = ADEAN(self.D50)

        if self.switch_Yini== 0:
            # @jit
            def model_simulation(par):
                kr = np.exp(par[0])
                mu = np.exp(par[1])
                Ymd = lim(self.hb_splited,
                             self.dt_splited,
                             self.A,
                             self.mf,
                             kr,
                             mu,
                             self.Sm,
                             self.Yini)
                return Ymd[self.idx_obs_splited]

            self.model_sim = model_simulation

            def run_model(par):
                kr = np.exp(par[0])
                mu = np.exp(par[1])
                Ymd = lim(self.hb,
                             self.dt,
                             self.A,
                             self.mf,
                             kr,
                             mu,
                             self.Sm,
                             self.Yini)
                return Ymd

            self.run_model = run_model

            # @jit
            def init_par(population_size):
                log_lower_bounds = np.array([np.log(self.lb[0]), np.log(self.lb[1])])
                log_upper_bounds = np.array([np.log(self.ub[0]), np.log(self.ub[1])])
                population = np.zeros((population_size, 2))
                for i in range(2):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

        elif self.switch_Yini == 1:
            def model_simulation(par):
                kr = np.exp(par[0])
                mu = np.exp(par[1])
                Yini = par[2]
                Ymd = lim(self.hb_splited,
                             self.dt_splited,
                             self.A,
                             self.mf,
                             kr,
                             mu,
                             self.Sm,
                             Yini)
                return Ymd[self.idx_obs_splited]
            
            self.model_sim = model_simulation

            def run_model(par):
                kr = np.exp(par[0])
                mu = np.exp(par[1])
                Yini = par[2]
                Ymd = lim(self.hb,
                             self.dt,
                             self.A,
                             self.mf,
                             kr,
                             mu,
                             self.Sm,
                             Yini)
                return Ymd
            
            self.run_model = run_model

            def init_par(population_size):
                log_lower_bounds = np.array([np.log(self.lb[0]), np.log(self.lb[1]), 0.75*np.min(self.Obs_splited)])
                log_upper_bounds = np.array([np.log(self.ub[0]), np.log(self.ub[1]), 1.25*np.max(self.Obs_splited)])
                population = np.zeros((population_size, 3))
                for i in range(3):
                    population[:,i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
                
                return population, log_lower_bounds, log_upper_bounds
            
            self.init_par = init_par

    def split_data(self):
        """
        Split the data into calibration and validation datasets.
        """
        ii = np.where(self.time>=self.start_date)[0][0]
        self.hb = self.hb[ii:]
        self.time = self.time[ii:]

        idx = np.where((self.time < self.start_date) | (self.time > self.end_date))
        self.idx_validation = idx

        idx = np.where((self.time >= self.start_date) & (self.time <= self.end_date))
        self.idx_calibration = idx
        self.hb_splited = self.hb[idx]
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

    def calibrate(self):
        """
        Calibrate the model.
        """
        self.solution, self.objectives, self.hist = self.calibr_cfg.calibrate(self)