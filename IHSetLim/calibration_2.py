import numpy as np
from .lim import lim
from IHSetUtils import ADEAN
from IHSetUtils.CoastlineModel import CoastlineModel

class cal_Lim_2(CoastlineModel):
    """
    cal_Lim_2
    
    Configuration to calibfalse,and run the Lim et al. (2022) Shoreline Evolution Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Lim et al. (2022)',
            mode='calibration',
            model_type='CS',
            model_key='Lim'
        )

        self.setup_forcing()

    def setup_forcing(self):
        self.switch_Yini = self.cfg['switch_Yini']
        self.D50 = self.cfg['D50']
        self.mf = self.cfg['mf']

        self.hb[self.hb < 0.1] = 0.1

        self.hb_s[self.hb_s < 0.1] = 0.1

        if self.switch_Yini == 0:
            self.Yini = self.Obs_splited[0]

        self.Sm = np.mean(self.Obs_splited)
        self.A = ADEAN(self.D50)


    def init_par(self, population_size: int):
        if self.switch_Yini == 0:
            lowers = np.array([np.log(self.lb[0]), np.log(self.lb[1])])
            uppers = np.array([np.log(self.ub[0]), np.log(self.ub[1])])
        elif self.switch_Yini == 1:
            lowers = np.array([np.log(self.lb[0]), np.log(self.lb[1]), 0.75*np.min(self.Obs_splited)])
            uppers = np.array([np.log(self.ub[0]), np.log(self.ub[1]), 1.25*np.max(self.Obs_splited)])
        pop = np.zeros((population_size, len(lowers)))
        for i in range(len(lowers)):
            pop[:, i] = np.random.uniform(lowers[i], uppers[i], population_size)
        return pop, lowers, uppers

    def model_sim(self, par: np.ndarray) -> np.ndarray:
        if self.switch_Yini == 0:
            kr = np.exp(par[0])
            mu = np.exp(par[1])
            Ymd = lim(self.hb_s,
                      self.dt_s,
                      self.A,
                      self.mf,
                      kr,
                      mu,
                      self.Sm,
                      self.Yini)
        elif self.switch_Yini == 1:
            kr = np.exp(par[0])
            mu = np.exp(par[1])
            Yini = par[2]
            Ymd = lim(self.hb_s,
                      self.dt_s,
                      self.A,
                      self.mf,
                      kr,
                      mu,
                      self.Sm,
                      Yini)
        return Ymd[self.idx_obs_splited]
    
    def run_model(self, par: np.ndarray) -> np.ndarray:

        if self.switch_Yini == 0:
            kr = par[0]
            mu = par[1]
            Ymd = lim(self.hb,
                      self.dt,
                      self.A,
                      self.mf,
                      kr,
                      mu,
                      self.Sm,
                      self.Yini)
        elif self.switch_Yini == 1:
            kr = par[0]
            mu = par[1]
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
        if self.switch_Yini == 0:
            self.par_names = [r'k_r', r'mu']
        elif self.switch_Yini == 1:
            self.par_names = [r'k_r', r'mu', r'Y_i']
        for idx in [0, 1]:
            self.par_values[idx] = np.exp(self.par_values[idx])
