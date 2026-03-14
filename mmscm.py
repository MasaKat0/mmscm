import numpy as np
import pandas as pd
import scipy


class MMSCM:
    def __init__(self, data, method, target_unit_var, target_unit, target_outcome_var, target_year_var, target_year, demean=False, poly=2, num_quantiles=1000, moment_weights=None, loss_type="diag", gmm_weighting="identity", gmm_ridge=1e-8, gmm_W=None):
        self.method = method
        if self.method == "Abadie":
            self.obj_func = abadie_obj_func
        elif self.method == "MMSCM":
            self.obj_func = distscm_obj_func
            self.poly = poly
        elif self.method == "DiSCo":
            self.num_quantiles = num_quantiles
            self.obj_func = abadie_obj_func

        # Optional settings for MMSCM variants used in additional experiments
        self.moment_weights = moment_weights
        self.loss_type = loss_type
        self.gmm_weighting = gmm_weighting
        self.gmm_ridge = gmm_ridge
        self.gmm_W = gmm_W

        self.data = data        
        self.target_unit_var = target_unit_var
        self.target_unit = target_unit
        self.target_outcome_var = target_outcome_var
        self.target_year_var = target_year_var
        self.target_year = target_year
        
        self.year = np.array(data[target_year_var].unique(), np.int64)

        self.demean = demean
        
        self._data_setup()
        
        
    def _data_setup(self):
        self.unit_name_list = self.data[self.target_unit_var].unique()

        sub_data = self.data.copy()

        if self.demean:
            for u in self.unit_name_list:
                temp_data = sub_data[(self.data[self.target_unit_var] == u) & (self.data[self.target_year_var] <= self.target_year)]
                target_outcome_varues = temp_data[self.target_outcome_var]
                mean_val = np.mean(target_outcome_varues)
                if u == self.target_unit:
                    self.target_mean_val = mean_val
                sub_data.loc[self.data[self.target_unit_var] == u, self.target_outcome_var] = sub_data.loc[self.data[self.target_unit_var] == u, self.target_outcome_var]  - mean_val

        temp_data = sub_data[self.data[self.target_year_var] <= self.target_year].copy()
        numeric_cols = temp_data.select_dtypes(include=[np.number])
        temp_data[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean())

        untreated_units = []

        for unit_name in self.unit_name_list:
            if unit_name == self.target_unit:
                treated_unit = temp_data[temp_data[self.target_unit_var] == self.target_unit].drop(columns=[self.target_unit_var, self.target_year_var]).values
            else:
                untreated_units.append(temp_data[temp_data[self.target_unit_var] == unit_name].drop(columns=[self.target_unit_var, self.target_year_var]).values)

        all_units = [treated_unit]

        for unit in untreated_units:
            all_units.append(unit)

        max_val = np.abs(np.concatenate(all_units, axis=0)).max(axis=0)
        

        all_units_new = []

        for unit in all_units:
            unit = (unit / max_val)
            all_units_new.append(unit)

        self.all_units = all_units_new

        self.sub_data = sub_data
        
        
    def train_param(self):
        self.treated_final, self.untreated_final_list = self.all_units[0].T, self.all_units[1:]
        
        init_beta = np.ones(len(self.untreated_final_list))
        init_beta /= len(init_beta)

        cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
        bnds = tuple((0,1) for x in init_beta)

        if self.method == "Abadie":
            init_beta2 = np.ones(len(self.treated_final.T))
            init_beta2 /= len(init_beta2)

            bnds2 = tuple((0,10) for x in init_beta2)
            
            obj = lambda beta2: abadie_obj_func2(beta2, self.treated_final, self.untreated_final_list)
            self.res2 = scipy.optimize.minimize(obj, init_beta2, method='SLSQP', bounds=bnds2)

            init_beta = np.ones(len(self.untreated_final_list))
            init_beta /= len(init_beta)

            beta2 = self.res2.x
            #beta2[0] = 1
            #beta2[1:] = 0

            obj = lambda beta: self.obj_func(beta, self.treated_final, self.untreated_final_list, beta2)
            self.res = scipy.optimize.minimize(obj, init_beta, method='SLSQP', bounds=bnds,constraints=cons)

        elif self.method == "MMSCM":
            
            for dim in range(2, self.poly):
                self.treated_final = np.concatenate([self.treated_final, [self.treated_final[0]**dim]], axis=0)
                for i in range(len(self.untreated_final_list)):
                    self.untreated_final_list[i] = np.concatenate([self.untreated_final_list[i], np.array([self.untreated_final_list[i][:,0]**dim]).T], axis=1)

            init_beta = np.ones(len(self.untreated_final_list))
            init_beta /= len(init_beta)
            
            if self.moment_weights is None:
                beta2 = np.ones(len(self.treated_final))
                beta2 /= len(beta2)
            else:
                beta2 = np.array(self.moment_weights, dtype=float)
                if beta2.ndim != 1 or len(beta2) != len(self.treated_final):
                    raise ValueError("moment_weights must be a 1D array with length equal to the number of matched moments")
                beta2 = beta2 / beta2.sum()
            
            if self.loss_type is None or self.loss_type == "diag":
                obj = lambda beta: self.obj_func(beta, self.treated_final, self.untreated_final_list, beta2)
                self.res = scipy.optimize.minimize(obj, init_beta, method='SLSQP', bounds=bnds,constraints=cons)
            elif self.loss_type == "gmm":
                K, T = self.treated_final.shape
                if self.gmm_W is not None:
                    W = np.array(self.gmm_W, dtype=float)
                elif self.gmm_weighting is None or self.gmm_weighting == "identity":
                    W = np.eye(K)
                elif self.gmm_weighting == "diag":
                    W = np.diag(beta2)
                elif self.gmm_weighting == "efficient":
                    # First-stage estimate using the diagonal loss
                    obj0 = lambda beta: self.obj_func(beta, self.treated_final, self.untreated_final_list, beta2)
                    res0 = scipy.optimize.minimize(obj0, init_beta, method='SLSQP', bounds=bnds,constraints=cons)
                    g_t = _moment_residuals_time(res0.x, self.treated_final, self.untreated_final_list)
                    S = (g_t @ g_t.T) / T
                    ridge = self.gmm_ridge
                    temp = np.array(S + ridge * np.eye(K), np.float64)
                    W = np.linalg.pinv(temp)
                else:
                    raise ValueError("Unsupported gmm_weighting: %s" % self.gmm_weighting)
                
                obj = lambda beta: distscm_obj_func_gmm(beta, self.treated_final, self.untreated_final_list, W)
                self.res = scipy.optimize.minimize(obj, init_beta, method='SLSQP', bounds=bnds,constraints=cons)
            else:
                raise ValueError("Unsupported loss_type: %s" % self.loss_type)
            
        elif self.method == "DiSCo":
            quantiles = [(i + 1)/self.num_quantiles for i in range(self.num_quantiles-1)]
                        
            self.treated_final = np.array([np.quantile(self.treated_final[0], quantiles)])
            
            untreated_final_list = []
            for i in range(len(self.untreated_final_list)):
                untreated_final_list.append(np.array([np.quantile(self.untreated_final_list[i].T[0], quantiles)]).T)
                
            self.untreated_final_list = untreated_final_list
                            
            treatment_quantile = self.treated_final

            init_beta = np.ones(len(self.untreated_final_list))
            init_beta /= len(init_beta)
            
            if self.moment_weights is None:
                beta2 = np.ones(len(self.treated_final))
                beta2 /= len(beta2)
            else:
                beta2 = np.array(self.moment_weights, dtype=float)
                if beta2.ndim != 1 or len(beta2) != len(self.treated_final):
                    raise ValueError("moment_weights must be a 1D array with length equal to the number of matched moments")
                beta2 = beta2 / beta2.sum()
            
            if self.loss_type is None or self.loss_type == "diag":
                obj = lambda beta: self.obj_func(beta, self.treated_final, self.untreated_final_list, beta2)
                self.res = scipy.optimize.minimize(obj, init_beta, method='SLSQP', bounds=bnds,constraints=cons)
            elif self.loss_type == "gmm":
                K, T = self.treated_final.shape
                if self.gmm_W is not None:
                    W = np.array(self.gmm_W, dtype=float)
                elif self.gmm_weighting is None or self.gmm_weighting == "identity":
                    W = np.eye(K)
                elif self.gmm_weighting == "diag":
                    W = np.diag(beta2)
                elif self.gmm_weighting == "efficient":
                    # First-stage estimate using the diagonal loss
                    obj0 = lambda beta: self.obj_func(beta, self.treated_final, self.untreated_final_list, beta2)
                    res0 = scipy.optimize.minimize(obj0, init_beta, method='SLSQP', bounds=bnds,constraints=cons)
                    g_t = _moment_residuals_time(res0.x, self.treated_final, self.untreated_final_list)
                    S = (g_t @ g_t.T) / T
                    ridge = float(self.gmm_ridge)
                    W = np.linalg.pinv(S + ridge * np.eye(K))
                else:
                    raise ValueError("Unsupported gmm_weighting: %s" % self.gmm_weighting)
                
                obj = lambda beta: distscm_obj_func_gmm(beta, self.treated_final, self.untreated_final_list, W)
                self.res = scipy.optimize.minimize(obj, init_beta, method='SLSQP', bounds=bnds,constraints=cons)
            else:
                raise ValueError("Unsupported loss_type: %s" % self.loss_type)
            
        
    def predict(self, bias=True, correction=0):
        if bias:
            bias = self.bias_train()
        else:
            bias = 0
        
        temp_data = self.data.copy()

        if self.demean:
            temp_data = self.sub_data

        index = 0
        counterfactual_outcome = 0
        for unit_name in self.unit_name_list:
            if unit_name == self.target_unit:
                treated_unit_outcome = temp_data[temp_data[self.target_unit_var] == self.target_unit][self.target_outcome_var].values
            else:
                counterfactual_outcome += self.res.x[index] * temp_data[temp_data[self.target_unit_var] == unit_name][self.target_outcome_var].values

                index += 1
                
        self.counterfactual_outcome = counterfactual_outcome + bias + correction
        self.treated_unit_outcome = treated_unit_outcome + correction

        if self.demean:
            self.counterfactual_outcome = self.counterfactual_outcome + self.target_mean_val
            self.treated_unit_outcome = self.treated_unit_outcome + self.target_mean_val
                
        return self.treated_unit_outcome, self.counterfactual_outcome
            

    def bias_train(self):
        temp_data = self.data.copy()

        if self.demean:
            temp_data = self.sub_data

        temp_data = temp_data[temp_data[self.target_year_var] <= self.target_year]

        index = 0
        for unit_name in self.unit_name_list:
            if unit_name == self.target_unit:
                treated_unit_val = temp_data[temp_data[self.target_unit_var] == self.target_unit][self.target_outcome_var].values
            else:
                if index == 0:
                    untreated_pred_val = self.res.x[index] * temp_data[temp_data[self.target_unit_var] == unit_name][self.target_outcome_var].values
                else:
                    untreated_pred_val += self.res.x[index] * temp_data[temp_data[self.target_unit_var] == unit_name][self.target_outcome_var].values

                index += 1

        bias = np.mean(treated_unit_val - untreated_pred_val)
                              
        return bias
    
    def treatment_effect(self, effect_year):
        self.effect_year_idx = self.year == (self.target_year + effect_year)
        
        target_treated_unit_outcome = self.treated_unit_outcome[self.effect_year_idx]
        target_counterfactual_outcome = self.counterfactual_outcome[self.effect_year_idx]

        treatment_effect = target_treated_unit_outcome - target_counterfactual_outcome
        
        return treatment_effect, target_treated_unit_outcome, target_counterfactual_outcome

    def conformal_inference(self, range_interval_list):
        post_treatment = self.year > self.target_year
        
        true_year = self.target_year 
        target_year_end = self.year[-1]
                        
        y_list = []
        for hypothesis_value in range_interval_list:
            self._data_setup_conformal(hypothesis_value)
            self.target_year = target_year_end
            self.train_param()
            treated_unit_outcome, scm_pred = self.predict()
            self.target_year = true_year

            res_sum = 0
            count = 0
            
            treated_unit_outcome[post_treatment] = treated_unit_outcome[post_treatment] - hypothesis_value

            for t in range(len(self.year)-1):
                count += 1

                post_treatment_temp = (self.year + count > self.target_year) * (self.year + count <= target_year_end)
                post_treatment_temp2 = (self.year + count - len(self.year) > self.target_year) * (self.year + count > target_year_end)
                post_treatment_temp = post_treatment_temp + post_treatment_temp2
                res0 = np.abs((treated_unit_outcome - scm_pred)[post_treatment_temp])
                res1 = np.abs((treated_unit_outcome - scm_pred)[post_treatment])
                
                if res0.sum() < res1.sum():
                    res_sum += 1

            print(1 - res_sum / count)
            if (1 - res_sum / count) > 0.1:
                y_list.append(hypothesis_value)

            print(y_list)
                
        return y_list

    def _data_setup_conformal(self, hypothesis_value):
        self.unit_name_list = self.data[self.target_unit_var].unique()

        temp_data = self.data.copy()
        temp_data[temp_data.select_dtypes(include='number').columns] = temp_data.select_dtypes(include='number').fillna(temp_data[temp_data.select_dtypes(include='number').columns].mean())

        untreated_units = []

        for unit_name in self.unit_name_list:
            if unit_name == self.target_unit:
                unit_data_temp = temp_data[temp_data[self.target_unit_var] == self.target_unit]
                temp_val = unit_data_temp[self.target_outcome_var].values
                post_treatment = unit_data_temp[self.target_year_var] > self.target_year
                temp_val[post_treatment] = temp_val[post_treatment] - hypothesis_value
                temp_data.loc[temp_data[self.target_unit_var] == self.target_unit, self.target_outcome_var] = temp_val
                treated_unit = temp_data[temp_data[self.target_unit_var] == self.target_unit].drop(columns=[self.target_unit_var, self.target_year_var]).values
            else:
                untreated_units.append(temp_data[temp_data[self.target_unit_var] == unit_name].drop(columns=[self.target_unit_var, self.target_year_var]).values)

        all_units = [treated_unit]

        for unit in untreated_units:
            all_units.append(unit)

        max_val = np.abs(np.concatenate(all_units, axis=0)).max(axis=0)

        all_units_new = []

        for unit in all_units:
            unit = unit / max_val
            all_units_new.append(unit)
            
        #self.post_treatment = post_treatment
        self.all_units = all_units_new

    def dist_infernece(self, num_resample=1000):
        temp_data = self.data.copy()

        temp_data = temp_data[temp_data[self.target_year_var] > self.target_year]

        index = 0
        untreated_pred_val_list = []
        for unit_name in self.unit_name_list:
            if unit_name == self.target_unit:
                treated_unit_val = temp_data[temp_data[self.target_unit_var] == self.target_unit][self.target_outcome_var].values
            else:
                if index == 0:
                    untreated_pred_val_list.append(temp_data[temp_data[self.target_unit_var] == unit_name][self.target_outcome_var].values)
                else:
                    untreated_pred_val_list.append(temp_data[temp_data[self.target_unit_var] == unit_name][self.target_outcome_var].values)

                index += 1

        prob = self.res.x
        counterfactual_dist = []
        for j in range(num_resample):
            choice_idx = np.random.choice([i for i in range(len(untreated_pred_val_list))], p=prob)
            counterfactual_dist.append(np.random.choice(untreated_pred_val_list[choice_idx]))
                
        self.counterfactual_outcome_dist = treated_unit_val 
        self.treated_unit_val_dist = counterfactual_dist 
                
        return self.counterfactual_outcome_dist, self.treated_unit_val_dist

    def dist_infernece_all(self, num_resample=1000):
        temp_data = self.data.copy()

        temp_data = temp_data[temp_data[self.target_year_var] > self.target_year]

        index = 0
        untreated_pred_val_list = []
        for unit_name in self.unit_name_list:
            if unit_name == self.target_unit:
                treated_unit_val = temp_data[temp_data[self.target_unit_var] == self.target_unit].values
            else:
                if index == 0:
                    untreated_pred_val_list.append(temp_data[temp_data[self.target_unit_var] == unit_name].values)
                else:
                    untreated_pred_val_list.append(temp_data[temp_data[self.target_unit_var] == unit_name].values)

                index += 1

        prob = self.res.x
        counterfactual_dist = []
        for j in range(num_resample):
            choice_idx = np.random.choice([i for i in range(len(untreated_pred_val_list))], p=prob)
            val_temp = untreated_pred_val_list[choice_idx]
            val_indx = [i for i in range(len(val_temp))]
            val_indx = np.random.choice(val_indx)
            counterfactual_dist.append(val_temp[val_indx])
                
        self.counterfactual_outcome_dist = treated_unit_val 
        self.treated_unit_val_dist = counterfactual_dist 
                
        return self.counterfactual_outcome_dist, self.treated_unit_val_dist

        

def abadie_obj_func(beta, treated, untreated, bata2):
    #treated = treated
    K, T = treated.shape
    J = len(untreated)

    untreated_weighted = 0
    
    for i in range(len(untreated)):
        untreated_weighted += (beta[i]*untreated[i].T)
        
    diff_list = []

    for i in range(K):
        diff_list.append(treated[i] - untreated_weighted[i])


    diff_list = np.array(diff_list)

    diff_list_mse = np.mean(diff_list**2, axis=1)

    loss = 0
    for i in range(K):
        loss += diff_list_mse[i] * bata2[i]

    return loss

def abadie_obj_func2(beta2, treated, untreated):
    beta2 /= np.sum(beta2)
    #beta2[0] = 1
    #beta2[1:] = 0

    init_beta = np.ones(len(untreated))
    init_beta /= len(init_beta)

    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    bnds = tuple((0,1) for x in init_beta)
    
    obj = lambda beta: abadie_obj_func(beta, treated, untreated, beta2)
    res = scipy.optimize.minimize(obj, init_beta, method='SLSQP', bounds=bnds,constraints=cons)
    
    untreated_weighted = 0

    K, T = treated.shape
    
    for i in range(len(untreated)):
        untreated_weighted += (res.x[i]*untreated[i].T)
        
    diff_list = []
    for i in range(K):
        diff_list.append(treated[i] - untreated_weighted[i])

    diff_list = np.array(diff_list)

    diff_list_mse = np.mean(diff_list**2, axis=1)


    loss = diff_list_mse[0]

    return loss


def distscm_obj_func(beta, treated, untreated, bata2):
    #treated = treated
    K, T = treated.shape
    J = len(untreated)

    untreated_weighted = 0
    
    for i in range(len(untreated)):
        untreated_weighted += (beta[i]*untreated[i].T)
        
    diff_list = []

    for i in range(K):
        diff_list.append(treated[i] - untreated_weighted[i])

    diff_list = np.array(diff_list)

    diff_list_mse = np.mean(diff_list, axis=1)**2

    loss = 0
    for i in range(K):
        loss += diff_list_mse[i] * bata2[i]

    return loss

def distscm_obj_func2(beta2, treated, untreated):
    beta2 /= np.sum(beta2)
    #beta2[0] = 1
    #beta2[1:] = 0

    init_beta = np.ones(len(untreated))
    init_beta /= len(init_beta)

    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    bnds = tuple((0,1) for x in init_beta)

    obj = lambda beta: abadie_obj_func(beta, treated, untreated, beta2)
    res = scipy.optimize.minimize(obj, init_beta, method='SLSQP', bounds=bnds,constraints=cons)
    
    untreated_weighted = 0

    K, T = treated.shape
    
    for i in range(len(untreated)):
        untreated_weighted += (res.x[i]*untreated[i].T)
        
    diff_list = []
    for i in range(K):
        diff_list.append(treated[i] - untreated_weighted[i])

    diff_list = np.array(diff_list)

    diff_list_mse = np.mean(diff_list, axis=1)**2

    loss = diff_list_mse[0]

    return loss


def disco_obj_func(beta, treated, untreated, beta2):
    #treated = treated
    K, T = treated.shape
    J = len(untreated)

    untreated_weighted = 0
    
    for i in range(len(untreated)):
        untreated_weighted += (beta[i]*untreated[i].T)
        
    diff_list = []

    for i in range(K):
        diff_list.append(treated[i] - untreated_weighted[i])

    diff_list = np.array(diff_list)

    diff_list_mse = np.mean(diff_list, axis=1)**2

    loss = 0
    for i in range(K):
        loss += diff_list_mse[i] * beta2[i]

    return loss




def _moment_residuals_time(beta, treated, untreated):
    """Compute g_t(beta) over time, used for GMM weighting.

    Returns a matrix of shape (K, T), where the k-th row is the residual
    series for the k-th matched moment.
    """
    untreated_weighted = 0
    for i in range(len(untreated)):
        untreated_weighted += (beta[i] * untreated[i].T)
    return treated - untreated_weighted


def distscm_obj_func_gmm(beta, treated, untreated, W):
    """GMM-type loss, g_mean(beta)^T W g_mean(beta)."""
    g_t = _moment_residuals_time(beta, treated, untreated)
    g_mean = np.mean(g_t, axis=1)
    return float(g_mean.T @ W @ g_mean)
