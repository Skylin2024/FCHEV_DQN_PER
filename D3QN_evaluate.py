#!/usr/bin/env python
# coding: utf-8

import os, shutil
import argparse
import torch

from Duel_DDQN.utils import str2bool
from datetime import datetime
from Duel_DDQN.DQN import DQN_agent
import numpy as np


from Model.RLenv import Env
from tqdm import tqdm


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
# parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=True, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=311*1801, help='which model to load')

parser.add_argument('--seed', type=int, default=1, help='random seed')
# parser.add_argument('--random_steps', type=int, default=int(3e3), help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=20, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=1, help='explore noise')
parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')
parser.add_argument('--Double', type=str2bool, default=False, help='Whether to use Double Q-learning')
parser.add_argument('--Duel', type=str2bool, default=False, help='Whether to use Duel networks')
parser.add_argument('--action_dim', type=float, default=31, help='How many action to choose')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc)  # from str to torch.device
# print(opt)

def main():

    data = scio.loadmat('./StandardCycle_kph_column/Drive_cycle_WLTP.mat')
    speed_list = np.array(data['speed_vector']).flatten()  # class <ndarray>
    speed_list = speed_list / 3.6
    soc0 = 0.6

    opt.state_dim = 3
    # opt.action_dim = 20

    env = Env(soc0, speed_list)

    # Algorithm Setting
    if opt.Duel: algo_name = 'Duel'
    else: algo_name = ''
    if opt.Double: algo_name += 'DDQN'
    else: algo_name += 'DQN'

    # Seed Everything
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Build model and replay buffer
    if not os.path.exists('Result_model'): os.makedirs('Result_model')
    # TODO 需要根据实际情况进行修改
    BriefEnvName = 'FCEV_DRL'
    model_path = 'Result_model'
    agent = DQN_agent(**vars(opt))
    episode = 300
    if opt.Loadmodel: agent.load(model_path, algo_name,BriefEnvName,episode)
    save_path = 'evaluate_DQN'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('创建了目录 {}'.format(save_path))
    print('Algorithm:', algo_name, '  Env:', BriefEnvName, '  state_dim:', opt.state_dim,
          '  action_dim:', opt.action_dim, '  Random Seed:', opt.seed, '\n')
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}-{}_S{}_'.format(algo_name,BriefEnvName[opt.EnvIdex],opt.seed) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    average_reward = []  # average_reward of each episode
    h2_100_list = []
    eq_h2_100_list = []  # equivalent hydrogen consumption per 100 km
    FCS_SoH = []
    Batt_SoH = []
    SOC = []
    action_space = np.linspace(-15, 15, opt.action_dim, dtype=np.float32)

    # training process
    for episode in tqdm(range(2)):

        state = env.reset()  # Do not use opt.seed directly, or it can overfit to opt.seed
        action = 0
        rewards = []
        info = []
        episode_info = {
            'P_dem_m': [], 'P_dem_e': [], 'Mot_spd': [], 'Mot_trq': [],
            'Mot_pwr': [], 'Mot_eta': [], 'Bat_soc': [], 'Bat_vol': [],
            'Bat_cur': [], 'Bat_pwr': [], 'FCS_pwr': [], 'FCS_eta': [],
            'FC_fuel': [], 'EMS_reward': [], 'soc_cost': [],
            'h2_equal': [], 'h2_money': [], 'Percentage_FC': [],
            'FcDegSum': [], 'mbat_deg': [], 'Percentage_Bat': [], 'FCS_SOH': [],
            'Batt_SOH': []
        }
        '''Interact & trian'''
        for episode_step in range(len(speed_list)):

            # e-greedy exploration
            action_id = agent.select_action(state, deterministic=True)
            action += (action_space[action_id] / 50)  # [float]
            action = np.clip(action, 0.01, 0.8)

            next_state, reward, done, info = env.step(action, episode_step)  # dw: dead&win; tr: truncated
            rewards.append(reward)

            agent.replay_buffer.add(state, action_id, reward, next_state, done)
            state = next_state
            for key in episode_info.keys():
                episode_info[key].append(info[key])

            # end of an episode: sava model params,
            if episode_step + 1 == 1800:
                datadir = save_path +'/data_ep%d.mat' % (episode)
                scio.savemat(datadir, mdict=episode_info)

            '''Noise decay & Record & Log'''

        '''save model'''
        agent.save(model_path, algo_name, BriefEnvName, episode)
        if agent.exp_noise > 0.02:
            agent.exp_noise *= opt.noise_decay

        # show episode data
        travel = info['travel'] / 1000  # km
        h2 = sum(episode_info['FC_fuel'])  # g
        eq_h2 = sum(episode_info['h2_equal'])  # g
        h2_100 = h2 / travel * 100
        equal_h2_100 = eq_h2 / travel * 100
        h2_100_list.append(h2_100)
        eq_h2_100_list.append(equal_h2_100)


        # print
        Bat_soc = info['Bat_soc']
        fcs_soh = info['FCS_SOH']
        bat_soh = info['Batt_SOH']
        FCS_SoH.append(fcs_soh)
        Batt_SoH.append(bat_soh)
        SOC.append(Bat_soc)

        # save loss and reward on average
        ep_r = np.mean(rewards)
        average_reward.append(ep_r)
        print('\nepi %d: SOC %.4f, H2_100km %.1f, H2_eq_100km %.1f' % (episode, Bat_soc, h2_100, equal_h2_100))
        print('\tfcs_soh %.5f, bat_soh %.5f, reward %.2f' % (fcs_soh, bat_soh, ep_r))

        scio.savemat(save_path + '/reward.mat', mdict={'reward': average_reward})
        scio.savemat(save_path + '/h2.mat', mdict={'h2': h2_100_list})
        scio.savemat(save_path + '/eq_h2.mat', mdict={'eq_h2': eq_h2_100_list})
        scio.savemat(save_path + '/FCS_SOH.mat', mdict={'FCS_SOH': FCS_SoH})
        scio.savemat(save_path + '/Batt_SOH.mat', mdict={'Batt_SOH': Batt_SoH})
        scio.savemat(save_path + '/SOC.mat', mdict={'SOC': SOC})



if __name__ == '__main__':
    main()








