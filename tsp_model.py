
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info


from tsp_data import TSPDataset
from torch.utils.data import DataLoader as GraphDataLoader


from utils.lr_schedulers import get_schedule_fn
from utils.tsp_diffusion import CategoricalDiffusion
from tsp_net import TmEncoder
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours




class TSPModel(pl.LightningModule):
  def __init__(self,param_args):
    super(TSPModel, self).__init__()
    self.args = param_args
    self.diffusion_schedule = self.args.diffusion_schedule
    self.diffusion_steps = self.args.diffusion_steps

    self.train_dataset = TSPDataset(
        data_file=self.args.training_split,
    )

    self.test_dataset = TSPDataset(
        data_file=self.args.test_split,
    )

    self.validation_dataset = TSPDataset(
        data_file=self.args.validation_split,
    )

    out_channels = 2
    self.diffusion = CategoricalDiffusion(
        T=self.diffusion_steps, schedule=self.diffusion_schedule
        )

    self.model = TmEncoder(
          n_layers=self.args.n_layers,
          hidden_dim=self.args.hidden_dim,
          out_channels=out_channels,
      )
    self.num_training_steps_cached = None

  def forward(self, x, t, v):
    return self.model(x, t, v)

  def training_step(self, batch, batch_idx):
    _, points, adj_matrix, _ = batch
    t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)

    # Sample from diffusion
    adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
    xt = self.diffusion.get_xt(adj_matrix_onehot, t)
    xt = xt * 2 - 1
    xt = xt * (1.0 + 0.05 * torch.rand_like(xt))
    t = torch.from_numpy(t).float().view(adj_matrix.shape[0])

    # Denoise
    x0_pred = self.forward(
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        points.float().to(adj_matrix.device),
    )

    # Compute loss
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(x0_pred, adj_matrix.long())

    self.log("train/loss", loss)
    return loss

  def test_step(self, batch, batch_idx, split='test'):
    device = batch[-1].device
    real_batch_idx, points, adj_matrix, gt_tour = batch
    np_points = points.cpu().numpy()[0]
    np_gt_tour = gt_tour.cpu().numpy()[0]

    stacked_tours = []
    output_tours = []
    ns, merge_iterations = 0, 0
    # record time
    import time
    all_sample_time = 0
    all_search_time = 0
    all_greedy_time = 0
    all_2opt_time = 0

    if self.args.parallel_sampling > 1:
      points = points.repeat(self.args.parallel_sampling, 1, 1)
    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(adj_matrix.float())
      if self.args.parallel_sampling > 1:
        xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        xt = torch.randn_like(xt)
      xt = (xt > 0).long()

      sample_start_time = time.time()
      steps = self.args.inference_diffusion_steps+1
      xt = self.diffusion.sample(
        model = self.model,
        xt = xt,
        v = points,
        inference_step = steps,
        inference_schedule = self.args.inference_schedule,
        device=device
        )
      cnt = xt.size()[0]
      adj_mat = xt.float().cpu().detach().numpy() + 1e-6
      all_sample_time = all_sample_time + time.time()-sample_start_time

      greedy_time = time.time()
      tours, merge_iterations = merge_tours(
          adj_mat, np_points,
          parallel_sampling=cnt,
      )
      all_greedy_time = all_greedy_time + time.time()-greedy_time
      opt_time = time.time()
      solved_tours, ns = batched_two_opt_torch(
          np_points.astype("float64"), np.array(tours).astype('int64'),
          max_iterations=self.args.two_opt_iterations, device=device)
      all_2opt_time = all_2opt_time + time.time()-opt_time

      stacked_tours.append(solved_tours)
      output_tours.append(tours)
    solved_tours = np.concatenate(stacked_tours, axis=0)
    greedy_tours = np.concatenate(output_tours, axis=0)

    tsp_solver = TSPEvaluator(np_points)
    gt_cost = tsp_solver.evaluate(np_gt_tour)

    total_sampling = cnt * self.args.sequential_sampling

    all_solved_costs = [tsp_solver.evaluate(solved_tours[i]) for i in range(total_sampling)]
    best_solved_cost = np.min(all_solved_costs)
    all_greedy_costs = [tsp_solver.evaluate(greedy_tours[i]) for i in range(total_sampling)]
    best_greedy_cost = np.min(all_greedy_costs)


    metrics = {
        f"{split}/gt_cost": gt_cost,
        f"{split}/2opt_iterations": ns,
        f"{split}/merge_iterations": merge_iterations,
        f"{split}/avg_sample_time": all_sample_time,
        f"{split}/avg_greedy_time": all_greedy_time,
        f"{split}/avg_2opt_time": all_2opt_time,
        f"{split}/avg_after_time": all_greedy_time + all_2opt_time,
        f"{split}/avg_total_time": all_sample_time + all_greedy_time + all_2opt_time,
        f"{split}/greedy_cost": best_greedy_cost,
        f"{split}/greedy_drop": (best_greedy_cost-gt_cost)/gt_cost*100,
        f"{split}/solved_drop": (best_solved_cost-gt_cost)/gt_cost*100,
    }


    for k, v in metrics.items():
      self.log(k, v, on_epoch=True, sync_dist=True)
    self.log(f"{split}/all_sample_time",all_sample_time,reduce_fx=sum)
    self.log(f"{split}/all_greedy_time",all_greedy_time,reduce_fx=sum)
    self.log(f"{split}/all_2opt_time",all_2opt_time,reduce_fx=sum)
    self.log(f"{split}/all_total_time",all_sample_time+all_greedy_time+all_2opt_time,reduce_fx=sum)
    self.log(f"{split}/solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)
    return metrics



  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')

  def configure_optimizers(self):
    rank_zero_info('Parameters: %d' % sum([p.numel() for p in self.model.parameters()]))
    rank_zero_info('Training steps: %d' % self.get_total_num_training_steps())

    if self.args.lr_scheduler == "constant":
      return torch.optim.AdamW(
          self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    else:
      optimizer = torch.optim.AdamW(
          self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
      scheduler = get_schedule_fn(self.args.lr_scheduler, self.get_total_num_training_steps())(optimizer)

      return {
          "optimizer": optimizer,
          "lr_scheduler": {
              "scheduler": scheduler,
              "interval": "step",
          },
      }

  def train_dataloader(self):
    batch_size = self.args.batch_size
    train_dataloader = GraphDataLoader(
        self.train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=self.args.num_workers, pin_memory=True,
        persistent_workers=True, drop_last=True)
    return train_dataloader

  def test_dataloader(self):
    batch_size = 1
    print("Test dataset size:", len(self.test_dataset))
    test_dataloader = GraphDataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=self.args.num_workers, persistent_workers=True,
    )
    return test_dataloader

  def val_dataloader(self):
    batch_size = 1
    val_dataset = torch.utils.data.Subset(self.validation_dataset, range(self.args.validation_examples))
    print("Validation dataset size:", len(val_dataset))
    val_dataloader = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=self.args.num_workers, persistent_workers=True,
    )
    return val_dataloader


  def get_total_num_training_steps(self) -> int:
    """Total training steps inferred from datamodule and devices."""
    if self.num_training_steps_cached is not None:
      return self.num_training_steps_cached
    dataset = self.train_dataloader()
    if self.trainer.max_steps and self.trainer.max_steps > 0:
      return self.trainer.max_steps

    dataset_size = (
        self.trainer.limit_train_batches * len(dataset)
        if self.trainer.limit_train_batches != 0
        else len(dataset)
    )

    num_devices = max(1, self.trainer.num_devices)
    effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
    self.num_training_steps_cached = (dataset_size // effective_batch_size) * self.trainer.max_epochs
    return self.num_training_steps_cached
