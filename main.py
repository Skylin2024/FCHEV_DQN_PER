import time
import numpy as np
import torch
# import gymnasium as gym
from D3QN_PRE.PriorDQN import DQN_Agent,PrioritizedReplayBuffer,device
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
from D3QN_PRE.utils import evaluate_policy,str2bool,LinearSchedule
import scipy.io as scio
from Model.RLenv import Env
from tqdm import tqdm

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=50*1000, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(4e5), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(5e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')
parser.add_argument('--warmup', type=int, default=int(3e3), help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')
parser.add_argument('--buffer_size', type=int, default=int(1e5), help='size of replay buffer')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise_init', type=float, default=0.5, help='init explore noise')
parser.add_argument('--exp_noise_end', type=float, default=0.1, help='final explore noise')
parser.add_argument('--noise_decay_steps', type=int, default=int(1e5), help='decay steps of explore noise')
parser.add_argument('--DDQN', type=str2bool, default=True, help='True:DDQN; False:DQN')

parser.add_argument('--alpha', type=float, default=0.6, help='alpha for PER')
parser.add_argument('--beta_init', type=float, default=0.4, help='beta for PER')
parser.add_argument('--beta_gain_steps', type=int, default=int(3e5), help='steps of beta from beta_init to 1.0')
parser.add_argument('--action_dim', type=float, default=31, help='How many action to choose')

opt = parser.parse_args()
print(opt)

def main():
    BriefEnvName = 'FCEV_DRL'
    data = scio.loadmat('./StandardCycle_kph_column/Drive_cycle_WLTP.mat')
    speed_list = np.array(data['speed_vector']).flatten()  # class <ndarray>
    speed_list = speed_list / 3.6
    soc0 = 0.6

    env = Env(soc0, speed_list)
    opt.state_dim = 3
    # opt.state_dim = env.observation_space.shape[0]
    # opt.action_dim = env.action_space.n
    # opt.max_e_steps = env._max_episode_steps

    #Use DDQN or DQN
    if opt.DDQN: algo_name = 'DDQN'
    else: algo_name = 'DQN'

    #Seed everything
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    print('Algorithm:', algo_name, '  Env:', BriefEnvName, '  state_dim:', opt.state_dim,
          'action_dim:', opt.action_dim, '  Random Seed:', opt.seed, '\n')

    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/Prior{}_{}'.format(algo_name,BriefEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    #Build model and replay buffer
    if not os.path.exists('Result_model'): os.mkdir('Result_model')
    model_path = 'Result_model'
    agent = DQN_Agent(opt)
    if opt.Loadmodel: agent.load(algo_name,BriefEnvName[opt.EnvIdex],opt.ModelIdex)

    if not os.path.exists('result_DQN'):
        os.makedirs('result_DQN')
        print("创建了目录 'result_DQN'")
    save_path = 'result_DQN'
    buffer = PrioritizedReplayBuffer(opt)

    exp_noise_scheduler = LinearSchedule(opt.noise_decay_steps, opt.exp_noise_init, opt.exp_noise_end)
    beta_scheduler = LinearSchedule(opt.beta_gain_steps, opt.beta_init, 1.0)
    average_reward = []  # average_reward of each episode
    h2_100_list = []
    eq_h2_100_list = []  # equivalent hydrogen consumption per 100 km
    money_100_list = []  # money spent per 100 km
    FCS_SoH = []
    Batt_SoH = []
    SOC = []
    action_space = np.linspace(-15, 15, opt.action_dim, dtype=np.float32)

    total_steps = 0
    for episode in tqdm(range(400)):
        state = env.reset()  # Do not use opt.seed directly, or it can overfit to opt.seed
        action = 0
        done = False
        rewards = []
        loss = []
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

        for episode_step in range(len(speed_list)):
            #e-greedy exploration
            if buffer.size < opt.warmup: action_id = np.random.randint(0, 20)
            else: action_id = agent.select_action(state, deterministic=False)
            action += (action_space[action_id] / 50)  # [float]
            action = np.clip(action, 0.01, 0.8)

            next_state, reward, done, info = env.step(action, episode_step)  # dw: dead&win; tr: truncated
            rewards.append(reward)
            # if r <= -100: r = -10  # good for LunarLander
            buffer.add(state, action_id, reward, next_state, done)
            state = next_state
            for key in episode_info.keys():
                episode_info[key].append(info[key])

            agent.exp_noise = exp_noise_scheduler.value(total_steps)
            buffer.beta = beta_scheduler.value(total_steps)

            '''update if its time'''
            # train 50 times every 50 steps rather than 1 training per step. Better!
            if total_steps >= opt.warmup and total_steps % opt.update_every == 0:
                for j in range(opt.update_every):
                    agent.train(buffer)

            if episode_step + 1 == 1800:
                datadir = save_path +'/data_ep%d.mat' % (episode)
                scio.savemat(datadir, mdict=episode_info)

            '''record & log'''
            # if (total_steps) % opt.eval_interval == 0:
            #     score = evaluate_policy(eval_env, model)
            #     if opt.write:
            #         writer.add_scalar('ep_r', score, global_step=total_steps)
            #         writer.add_scalar('p_sum', buffer.sum_tree.priority_sum, global_step=total_steps)
            #         writer.add_scalar('p_max', buffer.sum_tree.priority_max, global_step=total_steps)
            #         writer.add_scalar('noise', model.exp_noise, global_step=total_steps)
            #         writer.add_scalar('beta', buffer.beta, global_step=total_steps)
            #     print('EnvName:',BriefEnvName[opt.EnvIdex],'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', int(score))

            total_steps += 1
            '''save model'''
            if (total_steps) % 1800 == 0:
                agent.save(model_path, algo_name, BriefEnvName, episode)
        # show episode data
        travel = info['travel'] / 1000  # km
        h2 = sum(episode_info['FC_fuel'])  # g
        eq_h2 = sum(episode_info['h2_equal'])  # g
        # money = sum(episode_info['money_cost_real'])  # RMB
        h2_100 = h2 / travel * 100
        equal_h2_100 = eq_h2 / travel * 100
        # m_100 = money / travel * 100
        h2_100_list.append(h2_100)
        eq_h2_100_list.append(equal_h2_100)
        # money_100_list.append(m_100)

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

        # scio.savemat(save_path + '/lr.mat', mdict={'lr': lr})
        # scio.savemat(save_path + '/epsilon_list.mat', mdict={'epsilon': epsilon_list})
        # scio.savemat(save_path + '/loss.mat', mdict={'loss': average_loss})
        scio.savemat(save_path + '/reward.mat', mdict={'reward': average_reward})
        scio.savemat(save_path + '/h2.mat', mdict={'h2': h2_100_list})
        scio.savemat(save_path + '/eq_h2.mat', mdict={'eq_h2': eq_h2_100_list})
        # scio.savemat(save_path + '/money.mat', mdict={'money': money_100_list})
        scio.savemat(save_path + '/FCS_SOH.mat', mdict={'FCS_SOH': FCS_SoH})
        scio.savemat(save_path + '/Batt_SOH.mat', mdict={'Batt_SOH': Batt_SoH})
        scio.savemat(save_path + '/SOC.mat', mdict={'SOC': SOC})

    env.close()

if __name__ == '__main__':
    main()








