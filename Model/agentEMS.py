#!/usr/bin/env python
# coding: utf-8

from Model.FCV_Modelling_Class import FCV_model
import numpy as np


class EMS:
    """ EMS with SOH """
    def __init__(self,  soc0, abs_spd_MAX, abs_acc_MAX):
        self.time_step = 1.0
        # self.w_soc = 100
        self.done = False
        self.info = {}
        self.FCV = FCV_model()
        # self.FCHEV = FCHEV_SOH()
        # self.Battery = CellModel_2()
        self.obs_num = 3  # soc, soh-batt, soh-fcs, P_FCS, P_batt, spd, acc
        self.action_num = 1  # P_FCS
        # motor, unit in W
        # self.P_mot_max = self.FCHEV.motor_max_power  # 50 kW
        # self.P_mot_min = self.FCHEV.motor_min_power  # -50 kW
        # self.P_mot = 0
        # FCS, unit in kW
        self.h2_fcs = 0
        # self.P_FCS = 0
        # self.P_FCS_max = self.FCHEV.P_FC_max        # kW
        self.dSOH_FCS = 0
        self.SOH_FCS = 1.0
        # battery unit in W
        self.SOC_init = soc0        # 0.5
        # self.SOC_target = soc0
        # self.SOC = self.SOC_init
        # self.OCV_initial = 3.27  # soc = 0.6
        self.SOH_batt = 1.0
        # self.Tep_a = 25
        # self.P_batt = 0     # W
        # self.P_batt_max = self.Battery.batt_maxpower    # in W
        # self.SOC_delta = 0
        self.dSOH_batt = 0
        # self.I_batt = 0
        # paras_list = [SOC, SOH, Tep_c, Tep_s, Tep_a, Voc, V1, V2]
        # self.paras_list = [self.SOC, self.SOH_batt, 25, 25, self.Tep_a, self.OCV_initial, 0.001, 0.001]
        self.travle = 0
        self.car_spd = 0
        self.car_acc = 0
        self.abs_spd_MAX = abs_spd_MAX
        self.abs_acc_MAX = abs_acc_MAX
        self.P_FCS_pre = 0

    def reset_obs(self):
        self.Bat_soc = self.SOC_init
        # self.SOH_batt = 1.0
        # self.SOH_FCS = 1.0
        # self.dSOH_FCS = 0
        # self.Tep_a = 25
        self.P_dem_e = 0
        self.FCS_pwr = 0
        self.Bat_pwr = 0
        # self.paras_list = [self.SOC, self.SOH_batt, 25, 25, self.Tep_a, self.OCV_initial, 0, 0]
        self.done = False
        # self.info = {}
        self.travle = 0
        self.car_spd = 0
        self.car_acc = 0
        self.dSOH_FCS = 0
        self.dSOH_batt = 0

        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        # soc, soh-batt, soh-fcs, P_FCS, P_batt, spd, acc
        obs[0] = self.SOC_init
        obs[1] = 0        # in kW
        obs[2] = 0      # in kW

        # obs[1] = self.SOH_batt
        # obs[2] = self.SOH_FCS
        # obs[3] = self.P_batt   # in W
        # obs[1] = (self.car_spd + self.abs_spd_MAX) / (2 * self.abs_spd_MAX)
        # obs[2] = (self.car_acc + self.abs_acc_MAX) / (2 * self.abs_acc_MAX)
        return obs

    def execute(self, action, car_spd, car_acc):
        self.car_spd = car_spd
        self.car_acc = car_acc
        self.P_FCS = action * 50 * 1000   # kW

        if self.P_FCS < 500:
            self.P_FCS = 500# print(car_spd, car_acc)

        out = self.FCV.run(self.car_spd, self.car_acc, self.Bat_soc, self.P_FCS, self.P_FCS_pre)
        self.Bat_soc = out['Bat_soc']
        self.P_FCS_pre = self.P_FCS
        self.h2_fcs = out['FC_fuel']
        self.FcDegSum = out['FcDegSum']
        self.mbat_deg = out['mbat_deg']
        self.dSOH_FCS += float(out['Percentage_FC'])
        self.dSOH_batt += float(out['Percentage_Bat'])
        self.SOH_FCS = 1 - self.dSOH_FCS
        self.SOH_batt = 1 - self.dSOH_batt
        self.info.update({'FCS_SOH': self.SOH_FCS})
        self.info.update({'Batt_SOH': self.SOH_batt})
        # self.done = out['done']

        # T_axle, W_axle, P_axle = self.FCHEV.T_W_axle(self.car_spd, self.car_acc)
        # T_mot, W_mot, mot_eff, self.P_mot = self.FCHEV.run_motor(T_axle, W_axle, P_axle)  # W
        # P_dcdc, self.h2_fcs, info_fcs = self.FCHEV.run_fuel_cell(self.P_FCS)      # kW

        # self.dSOH_FCS, info_fcs_soh = self.FCHEV.run_FC_SOH(self.P_FCS)
        # self.SOH_FCS -= self.dSOH_FCS
        # self.P_batt = self.P_mot - P_dcdc*1000        # W
        # update power battery
        # self.paras_list, self.dSOH_batt, self.I_batt, self.done, info_batt = \
        #     self.Battery.run_cell(self.P_batt, self.paras_list)
        # self.SOC = self.paras_list[0]
        # self.SOH_batt = self.paras_list[1]
        # self.Tep_a = self.paras_list[4]
        self.travle += self.car_spd*self.time_step
        # self.info = {}
        self.info.update({'P_dem_m': out['P_dem_m'], 'P_dem_e': out['P_dem_e'],
                          'Mot_spd': out['Mot_spd'], 'Mot_trq': out['Mot_trq'],
                          'Mot_pwr': out['Mot_pwr'], 'Mot_eta': out['Mot_eta'],
                          'Bat_soc': out['Bat_soc'], 'Bat_vol': out['Bat_vol'],
                          'Bat_cur': out['Bat_cur'], 'Bat_pwr': out['Bat_pwr'],
                          'FCS_pwr': out['FCS_pwr'], 'FCS_eta': out['FCS_eta'],
                          'FC_fuel': out['FC_fuel'], 'FcDegSum': out['FcDegSum'],
                          'Percentage_FC': out['Percentage_FC'], 'Percentage_Bat': out['Percentage_Bat'],
                          'mbat_deg': out['mbat_deg'], 'travel': self.travle})
        # self.info.update(info_fcs)
        # self.info.update(info_batt)
        # self.info.update(info_fcs_soh)

        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        # soc, soh-batt, soh-fcs, P_FCS, P_batt, spd, acc
        obs[0] = self.Bat_soc
        obs[1] = self.P_FCS / 50 / 1000   # in kW
        obs[2] = (self.P_dem_e / 1000 + 30) / 80  # in kW
        # obs[1] = self.SOH_batt
        # obs[2] = self.SOH_FCS
        # obs[3] = self.P_batt - / 1000  # in kW
        # obs[1] = (self.car_spd + self.abs_spd_MAX) / (2 * self.abs_spd_MAX)
        # obs[2] = (self.car_acc + self.abs_acc_MAX) / (2 * self.abs_acc_MAX)
        return obs
    
    def get_reward(self, episode_step=0):
        # equivalent hydrogen consumption
        # if self.P_batt > 0:
        #     h2_batt = self.P_batt / 1000 * 0.0164      # g in one second
        #     # 取FCS效率最高点(0.5622)计算, 该系数为0.0164
        # else:
        #     h2_batt = 0
        # h2_equal = self.h2_fcs  # + h2_batt
        
        # SOC cost
        if self.Bat_soc >= 0.7 or self.Bat_soc <= 0.5:
            w_soc = 100 * 10  # 不可直接改变self.w_soc的值
        else:
            w_soc = 100
        soc_cost = w_soc * abs(self.Bat_soc - self.SOC_init)

        # money cost
        # hydrogen spent
        # h2_price = 4.4 / 1000      # ￥ per g
        # h2_cost = h2_price * self.h2_fcs

        # money spent of power battery degradation
        # batt_price = 6.5 * 178     # 108.14kWh * 1000 yuan/kWh
        # batt_money = batt_price * self.dSOH_batt

        # money spent of FCE degradation
        h2_equal = (self.h2_fcs + self.FcDegSum + self.mbat_deg)
        h2_money = (self.h2_fcs + self.FcDegSum + self.mbat_deg) * 4.4 / 1000

        # total money cost in one step # 未算在奖励函数里
        # reward = -(soc_cost + self.h2_fcs)
        reward = -(soc_cost + h2_equal)

        if self.Bat_soc < 0.3:
            reward -= 20
        # elif self.Bat_soc < 0.4 and self.Bat_soc >=0.3:
        #     reward += -3
        # elif self.Bat_soc <0.5 and self.Bat_soc >=0.4:
        #     reward += -2

        if self.Bat_soc > 0.9:
            reward -= 20
        # elif self.Bat_soc < 0.9 and self.Bat_soc >=0.8:
        #     reward += -3
        # elif self.Bat_soc <0.8 and self.Bat_soc >=0.7:
        #     reward += -2
        # if episode_step > 1780:
        #    if self.Bat_soc < 0.55 or self.Bat_soc > 0.65:
        #        reward -= 50
        #    elif self.Bat_soc < 0.58 and self.Bat_soc < 0.62:
        #       reward -= 5

        reward = float(reward)
        self.info.update({'EMS_reward': reward, 'soc_cost': soc_cost,
                          'h2_equal': h2_equal, 'h2_money': h2_money})
        # print(episode_step,{'EMS_reward': reward, 'soc_cost': soc_cost,
        #                   'h2_equal': h2_equal,  'money_cost': money_cost, 'h2_money': h2_money,
        #                   'batt_money': batt_money, 'fcs_money': fcs_money,
        #                   'money_cost_real': money_cost_real})
        return reward

    def get_info(self):
        return self.info

    def get_done(self):
        return self.done
