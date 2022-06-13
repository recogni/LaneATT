import pickle
import random
import logging
from typing import List
import cv2
import torch
import numpy as np
from tqdm import tqdm, trange
from lib.models.matching import match_proposals_with_targets

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

class Runner:
    def __init__(self, cfg, exp, device, resume=False, view=None, deterministic=False):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.logger = logging.getLogger(__name__)

        # Fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self):
        self.exp.train_start_callback(self.cfg)
        starting_epoch = 0
        model = self.cfg.get_model()
        model = model.to(self.device)
        optimizer = self.cfg.get_optimizer(model.parameters())
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(model, optimizer, scheduler)
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg['epochs']
        train_loader = self.get_train_dataloader()
        loss_parameters = self.cfg.get_loss_parameters()
        for epoch in trange(starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs):
            self.exp.epoch_start_callback(epoch, max_epochs)
            model.train()
            pbar = tqdm(train_loader)
            for i, (images, labels, _) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs, output_map = model(images, **self.cfg.get_train_parameters())
                loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Scheduler step (iteration based)
                scheduler.step()

                # Log
                postfix_dict = {key: float(value) for key, value in loss_dict_i.items()}
                postfix_dict['lr'] = optimizer.param_groups[0]["lr"]

                scores = torch.nn.functional.softmax(output_map[:, :, :2, ...], dim=2)[:, :, 1:2, ...]
                scores = torch.max(scores, dim=1)[0]

                scores = torch.cat([scores] * 3, dim=1)
                scores_th = (scores >= 0.8)

                l_cpu_np = labels.cpu().numpy()
                l_cpu_np = l_cpu_np[0, l_cpu_np[0, :, 1] >= 1.0, :]  # num_lanes, properties
                input_scores_map = np.zeros(scores[0].shape) # 3, H, W
                x, y = (model.fmap_w - 1)*l_cpu_np[:, 3] / (model.img_w - 1), l_cpu_np[:, 2] * (model.fmap_h - 1)
                input_scores_map[:, y.round().astype(np.int), x.round().astype(np.int)] = 1.0
                input_scores_map = input_scores_map[np.newaxis, ...]

                plt.figure()
                fig = plt.Figure(figsize=(model.img_w // 100, model.img_h // 100), dpi=100)
                canvas = FigureCanvasAgg(fig)
                ax = fig.add_subplot()
                ax.imshow(images[0].permute((1, 2, 0)).cpu().numpy())
                for lane_props in l_cpu_np:
                    mask = np.logical_and(lane_props[5:] >= 0.0, lane_props[5:] < model.img_w)
                    ax.plot(lane_props[5:][mask], (model.anchor_ys.cpu().numpy()*(model.img_h - 1))[mask])
                canvas.draw()
                buf = canvas.buffer_rgba()
                gt_lines = np.asarray(buf)[..., :3].transpose((2, 0, 1))[None, ...]
                gt_lines = gt_lines.astype(np.float32) / 255.0
                fig.clear()
                plt.close('all')

                positives_mask, invalid_offsets_mask, negatives_mask, target_positives_indices, distances = match_proposals_with_targets(
                    model, outputs[0][1], labels[0][labels[0][:, 1] == 1])
                l_cpu_np = outputs[0][1][positives_mask].cpu().numpy()  # num_lanes, properties
                matched_scores_map = np.zeros(scores[0].shape)  # 3, H, W
                x, y = (model.fmap_w - 1) * l_cpu_np[:, 3] / (model.img_w - 1), l_cpu_np[:, 2] * (model.fmap_h - 1)
                matched_scores_map[:, y.round().astype(np.int), x.round().astype(np.int)] = 1.0
                matched_scores_map = matched_scores_map[np.newaxis, ...]

                # plotting predictions
                plt.figure()
                fig = plt.Figure(figsize=(model.img_w // 100, model.img_h // 100), dpi=100)
                canvas = FigureCanvasAgg(fig)
                ax = fig.add_subplot()
                ax.imshow(images[0].permute((1, 2, 0)).cpu().numpy())
                positives_mask = torch.nn.functional.softmax(outputs[0][0][:, :2], dim=1)[:, 1] > 0.80
                l_cpu_np = outputs[0][0][positives_mask].detach().cpu().numpy()  # num_lanes, properties
                for lane_props in l_cpu_np:
                    start = int(round(lane_props[2].item() * model.n_strips))
                    length = int(round(lane_props[4].item()))
                    if length <= 1:
                        continue
                    end = start - (length - 1)
                    end = max(end, 0)
                    start = model.n_strips - start
                    end = model.n_strips - end
                    ax.plot(lane_props[5 + start: 5 + end + 1], model.anchor_ys.cpu().numpy()[start:end + 1]*(model.img_h - 1))

                canvas.draw()
                buf = canvas.buffer_rgba()
                pred_lines = np.asarray(buf)[..., :3].transpose((2, 0, 1))[None, ...]
                pred_lines = pred_lines.astype(np.float32) / 255.0
                fig.clear()
                plt.close('all')

                # plotting matched anchors
                plt.figure()
                fig = plt.Figure(figsize=(model.img_w // 100, model.img_h // 100), dpi=100)
                canvas = FigureCanvasAgg(fig)
                ax = fig.add_subplot()
                ax.imshow(images[0].permute((1, 2, 0)).cpu().numpy())
                positives_mask = torch.nn.functional.softmax(outputs[0][0][:, :2], dim=1)[:, 1] > 0.80
                l_cpu_np = outputs[0][1][positives_mask].detach().cpu().numpy()  # num_lanes, properties
                for lane_props in l_cpu_np:
                    mask = np.logical_and(lane_props[5:] >= 0.0, lane_props[5:] < model.img_w)
                    ax.plot(lane_props[5:][mask], (model.anchor_ys.cpu().numpy() * (model.img_h - 1))[mask])
                canvas.draw()
                buf = canvas.buffer_rgba()
                anchor_lines = np.asarray(buf)[..., :3].transpose((2, 0, 1))[None, ...]
                anchor_lines = anchor_lines.astype(np.float32) / 255.0
                fig.clear()
                plt.close('all')

                # create gt lines and anchor lines
                self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), postfix_dict,
                                           model_inputs=images, score_outputs=scores, score_outputs_th=scores_th,
                                           score_inputs=input_scores_map, matched_anchors=matched_scores_map,
                                           gt_lines=gt_lines, pred_lines=pred_lines, anchor_lines=anchor_lines)
                postfix_dict['loss'] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)
            self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler)

            # Validate
            if (epoch + 1) % self.cfg['val_every'] == 0:
                self.eval(epoch, on_val=True)
        self.exp.train_end_callback()

    def eval(self, epoch, on_val=False, save_predictions=False):
        model = self.cfg.get_model()

        if epoch >= 0:
            model_path = self.exp.get_checkpoint_path(epoch)
            self.logger.info('Loading model %s', model_path)
            model.load_state_dict(self.exp.get_epoch_model(epoch))

        model = model.to(self.device)
        model.eval()
        if on_val:
            dataloader = self.get_val_dataloader()
        else:
            dataloader = self.get_test_dataloader()
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        self.exp.eval_start_callback(self.cfg)
        with torch.no_grad():
            for idx, (images, labels, _) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                output, output_map = model(images, **test_parameters)
                prediction = model.decode(output, as_lanes=True)
                predictions.extend(prediction)

                scores = torch.nn.functional.softmax(output_map[:, :, :2, ...], dim=2)[:, :, 1:2, ...]
                scores = torch.max(scores, dim=1)[0]
                scores = torch.cat([scores] * 3, dim=1)
                scores_th = (scores >= 0.8)

                l_cpu_np = labels.cpu().numpy()
                l_cpu_np = l_cpu_np[0, l_cpu_np[0, :, 1] >= 1.0, :]  # num_lanes, properties
                input_scores_map = np.zeros(scores[0].shape)  # 3, H, W
                x, y = (model.fmap_w - 1) * l_cpu_np[:, 3] / (model.img_w - 1), l_cpu_np[:, 2] * (model.fmap_h - 1)
                input_scores_map[:, y.round().astype(np.int), x.round().astype(np.int)] = 1.0
                input_scores_map = input_scores_map[np.newaxis, ...]

                self.exp.tensorboard_writer.add_image('eval/output_scores_th', scores_th[0], idx)
                self.exp.tensorboard_writer.add_image('eval/output_scores', scores[0], idx)
                self.exp.tensorboard_writer.add_image('eval/input_image', images[0], idx)
                self.exp.tensorboard_writer.add_image('eval/input_scores', input_scores_map[0], idx)

                if self.view:
                    img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0])
                    if self.view == 'mistakes' and fp == 0 and fn == 0:
                        continue
                    cv2.imshow('pred', img)
                    cv2.waitKey(0)

        if save_predictions:
            with open('predictions.pkl', 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)

    def get_train_dataloader(self):
        train_dataset = self.cfg.get_dataset('train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.cfg['batch_size'],
                                                   shuffle=True,
                                                   num_workers=8,
                                                   worker_init_fn=self._worker_init_fn_)
        return train_loader

    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset('val')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.cfg['batch_size'],
                                                 shuffle=False,
                                                 num_workers=8,
                                                 worker_init_fn=self._worker_init_fn_)
        return val_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
