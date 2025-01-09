import math
import numpy as np
import torch
import torch.nn.functional as F

class CategoricalDiffusion(object):
  def __init__(self, T, schedule):
    # Diffusion steps
    self.T = T

    # Noise schedule
    if schedule == 'linear':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0, bT, T)
    elif schedule == 'cosine':
      self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
          0)  # Generate an extra alpha for bT
      self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

    beta = self.beta.reshape((-1, 1, 1))
    eye = np.eye(2).reshape((1, 2, 2))
    ones = np.ones((2, 2)).reshape((1, 2, 2))

    self.Qs = (1 - beta) * eye + (beta / 2) * ones

    Q_bar = [np.eye(2)]
    for Q in self.Qs:
      Q_bar.append(Q_bar[-1] @ Q)
    self.Q_bar = np.stack(Q_bar, axis=0)

  def __cos_noise(self, t):
    offset = 0.008
    return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

  def get_xt(self, x0_onehot, t):
    # Select noise scales
    Q_bar = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)
    xt = torch.matmul(x0_onehot, Q_bar.reshape((Q_bar.shape[0], 1, 2, 2)))
    return torch.bernoulli(xt[..., 1].clamp(0, 1))

  def append_zero(self, x):
      return torch.cat([x, x.new_zeros([1])])

  def get_schedule_linear(self, n, s_min, s_max, device='cpu'):
      ramp = torch.linspace(s_max, s_min, n)[:-1]
      np.clip(ramp, 1, s_max)
      return self.append_zero(ramp).to(device)

  def get_schedule_cosine(self, n, s_min, s_max, device='cpu'):
      ramp = torch.linspace(0, 1, n)
      ramp = 1-np.sin(ramp * np.pi / 2)
      ramp = ramp * (s_max-s_min) + s_min
      ramp = ramp[:-1]
      ramp = np.clip(ramp, 1., s_max)
      return self.append_zero(ramp).to(device)

  def get_schedule_pow(self, n, s_min, s_max, device='cpu'):
      p = -1
      ramp = torch.linspace(0.5, 2, n)
      ramp = ramp.pow(p)
      ramp = (ramp-0.5)*(2./3)
      ramp = ramp * (s_max-s_min) + s_min
      ramp = ramp[:-1]
      ramp = np.clip(ramp, 1., s_max)
      return self.append_zero(ramp).to(device)

  def sample(self,model,xt,v,inference_step,inference_schedule,device):
      if inference_schedule =="linear":
        time_schedule = self.get_schedule_linear(inference_step,0,self.T,device)
      elif inference_schedule == "cosine":
        time_schedule = self.get_schedule_cosine(inference_step,0,self.T,device) 
      elif inference_schedule == "pow":
        time_schedule = self.get_schedule_pow(inference_step,0,self.T,device)

      xt_list = []
      
      for i in range(inference_step-1):
        t = time_schedule[i]
        target_t = time_schedule[i+1]

        x0_pred = model.forward(
            xt.float().to(device),
            t.float().to(device).unsqueeze(0),
            v.float().to(device),
        )
        x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
        
        t = t.cpu().int()
        target_t = target_t.cpu().int()
        if target_t > 0:
          Q_t = np.linalg.inv(self.Q_bar[target_t]) @ self.Q_bar[t]
          Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)
        else:
          Q_t = torch.eye(2).float().to(x0_pred_prob.device)
          
        Q_bar_t_source = torch.from_numpy(self.Q_bar[t]).float().to(x0_pred_prob.device)
        Q_bar_t_target = torch.from_numpy(self.Q_bar[target_t]).float().to(x0_pred_prob.device)

        xt = F.one_hot(xt.long(), num_classes=2).float()
        xt = xt.reshape(x0_pred_prob.shape)

        x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
        x_t_target_prob_part_2 = Q_bar_t_target[0]
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]
        x_t_target_prob_part_2_new = Q_bar_t_target[1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

        x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new

        sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

        if target_t > 0:

          xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
        else:
          xt = sum_x_t_target_prob.clamp(min=0)
        
        xt_list.append(x0_pred_prob[...,1])
      xt_list.append(xt)
      xt_list = torch.cat(xt_list, dim=0).to(device) 
      return xt_list