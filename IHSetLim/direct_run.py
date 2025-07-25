# import numpy as np
# import xarray as xr
# import fast_optimization as fo
# import pandas as pd
# from scipy.stats import circmean
# from .lim import lim_njit
# import json
# from IHSetUtils import BreakingPropagation, ADEAN

# class Lim_run(object):
#     """
#     Lim_run
    
#     Configuration to calibfalse,and run the Lim et al. (2022) Shoreline Evolution Model.
    
#     This class reads input datasets, performs its calibration.
#     """

#     def __init__(self, path):

#         self.path = path
#         self.name = 'Lim et al. (2022)'
#         self.mode = 'standalone'
#         self.type = 'CS'
     
#         data = xr.open_dataset(path)
        
#         cfg = json.loads(data.attrs['run_Lim'])
#         self.cfg = cfg

#         self.switch_Yini = cfg['switch_Yini']
#         self.switch_brk = cfg['switch_brk']
#         if self.switch_brk == 1:
#             self.breakType = cfg['break_type']
#         self.D50 = cfg['D50']
#         self.mf = cfg['mf']

#         if cfg['trs'] == 'Average':
#             self.hs = np.mean(data.hs.values, axis=1)
#             self.tp = np.mean(data.tp.values, axis=1)
#             self.dir = np.mean(data.dir.values, axis=1)
#             self.time = pd.to_datetime(data.time.values)
#             self.Obs = data.average_obs.values
#             self.Obs = self.Obs[~data.mask_nan_average_obs]
#             self.time_obs = pd.to_datetime(data.time_obs.values)
#             self.time_obs = self.time_obs[~data.mask_nan_average_obs]
#             self.depth = np.mean(data.waves_depth.values)
#             self.bathy_angle = circmean(data.phi.values, high=360, low=0)
#         else:
#             self.hs = data.hs.values[:, cfg['trs']]
#             self.tp = data.tp.values[:, cfg['trs']]
#             self.dir = data.dir.values[:, cfg['trs']]
#             self.time = pd.to_datetime(data.time.values)
#             self.Obs = data.obs.values[:, cfg['trs']]
#             self.Obs = self.Obs[~data.mask_nan_obs[:, cfg['trs']]]
#             self.time_obs = pd.to_datetime(data.time_obs.values)
#             self.time_obs = self.time_obs[~data.mask_nan_obs[:, cfg['trs']]]
#             self.depth = data.waves_depth.values[cfg['trs']]
#             self.bathy_angle = data.phi.values[cfg['trs']]
        
#         self.start_date = pd.to_datetime(cfg['start_date'])
#         self.end_date = pd.to_datetime(cfg['end_date'])
        
#         data.close()

#         if self.switch_brk == 0:
#             self.hb = self.hs
#         elif self.switch_brk == 1:
#             self.hb, _, _ = BreakingPropagation(self.hs, self.tp, self.dir, np.repeat(self.depth, len(self.hs)), np.repeat(self.bathy_angle, len(self.hs)), self.breakType)
        
#         self.hb[self.hb < 0.1] = 0.1

#         self.split_data()

#         if self.switch_Yini == 1:
#             ii = np.argmin(np.abs(self.time_obs - self.time[0]))
#             self.Yini = self.Obs[ii]

#         mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))

#         self.idx_obs = mkIdx(self.time_obs)

#         # Now we calculate the dt from the time variable
#         mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
#         self.dt = mkDT(np.arange(0, len(self.time)-1))

#         self.Sm = np.mean(self.Obs)
#         self.A = ADEAN(self.D50)

#         if self.switch_Yini== 1:
#             def run_model(par):
#                 kr = par[0]
#                 mu = par[1]
#                 Ymd = lim_njit(self.hb,
#                              self.dt,
#                              self.A,
#                              self.mf,
#                              kr,
#                              mu,
#                              self.Sm,
#                              self.Yini)
#                 return Ymd

#             self.run_model = run_model

#         elif self.switch_Yini == 0:
#             def run_model(par):
#                 kr = par[0]
#                 mu = par[1]
#                 Yini = par[2]
#                 Ymd = lim_njit(self.hb,
#                              self.dt,
#                              self.A,
#                              self.mf,
#                              kr,
#                              mu,
#                              self.Sm,
#                              Yini)
#                 return Ymd
            
#             self.run_model = run_model

#     def split_data(self):
#         """
#         Split the data into calibration and validation datasets.
#         """
#         ii = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
#         self.hb = self.hb[ii]
#         self.time = self.time[ii]

#         ii = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]
#         self.Obs = self.Obs[ii]
#         self.time_obs = self.time_obs[ii]

#     def run(self, par):
#         self.full_run = self.run_model(par)
#         if self.switch_Yini == 1:
#             self.par_names = [r'k_r', r'mu']
#             self.par_values = par
#         elif self.switch_Yini == 0:
#             self.par_names = [r'k_r', r'mu', r'Y_{i}']
#             self.par_values = par
            
#         # self.calculate_metrics()

#     def calculate_metrics(self):
#         self.metrics_names = fo.backtot()[0]
#         self.indexes = fo.multi_obj_indexes(self.metrics_names)
#         self.metrics = fo.multi_obj_func(self.Obs, self.full_run[self.idx_obs], self.indexes)


import numpy as np
from .lim import lim
from IHSetUtils import ADEAN
from IHSetUtils.CoastlineModel import CoastlineModel

class Lim_run(object):
    """
    Lim_run
    
    Configuration to calibfalse,and run the Lim et al. (2022) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Lim et al. (2022)',
            mode='standalone',
            model_type='CS',
            model_key='run_Lim'
        )

        self.setup_forcing()

    def setup_forcing(self):
        self.switch_Yini = self.cfg['switch_Yini']
        self.D50 = self.cfg['D50']
        self.mf = self.cfg['mf']

        self.hb[self.hb < 0.1] = 0.1

        if self.switch_Yini == 0:
            self.Yini = self.Obs[0]

        self.Sm = np.mean(self.Obs)
        self.A = ADEAN(self.D50)
    
    def run_model(self, par: np.ndarray) -> np.ndarray:
        kr = par[0]
        mu = par[1]
        if self.switch_Yini == 1:
            Yini = self.Yini
        elif self.switch_Yini == 0:
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

    def _set_parameter_names(self):
        if self.switch_Yini == 1:
            self.par_names = [r'k_r', r'mu']
        elif self.switch_Yini == 0:
            self.par_names = [r'k_r', r'mu', r'Y_i']
