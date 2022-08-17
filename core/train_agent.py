from core.datasets.dataset import Dataset
from core.models.transfusion import TransFusion, SetCriterion

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import time
from tqdm import tqdm

class TrainAgent():
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.prev_val_loss = 1e6



    def prepare_loaders(self):
        config = self.config["data"]
        aug_config = self.config["augmentation"]
        train_dataset = Dataset(self.config["train"]["data"], config, aug_config)
        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.config["train"]["batch_size"], collate_fn = train_dataset.collate_fn)
        
        val_dataset = Dataset(self.config["val"]["data"], config, aug_config, "sth")
        self.val_loader = DataLoader(val_dataset, shuffle=True, batch_size=self.config["val"]["batch_size"], collate_fn = val_dataset.collate_fn)
        
    def build_model(self):
        learning_rate = self.config["train"]["learning_rate"]
        weight_decay = self.config["train"]["weight_decay"]
        lr_decay_at = self.config["train"]["lr_decay_at"]
        
        self.model = TransFusion(self.config["data"])

        self.model.to(self.device)
        self.criterion = SetCriterion(self.config["data"])

        if self.config["train"]["use_differential_learning"]:
            dif_learning_rate = self.config["train"]["differential_learning_rate"]
            self.optimizer = torch.optim.Adam([{'params': self.model.backbone.parameters(), 'lr': dif_learning_rate[1]},
                                             {'params': self.model.header.parameters(), 'lr': dif_learning_rate[0]}], 
                                             lr=learning_rate, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_decay_at, gamma=0.1)

    def train_one_epoch(self, epoch):
        total_loss = {}
        start_time = time.time()
        self.model.train()
        for data in tqdm(self.train_loader):
            voxel = data["voxel"].to(self.device)
            boxes = data["boxes"].to(self.device)
            data_type = data["data_type"].to(self.device)
            heatmap = data["heatmap"].to(self.device)
            
            self.optimizer.zero_grad()

            pred = self.model(voxel)
            loss_dict = self.criterion(pred, boxes, heatmap, data_type)
            loss_dict["loss"].backward()
            self.optimizer.step()

            loss_dict["loss"] = loss_dict["loss"].item()
            for key in loss_dict.keys():
                if key in total_loss:
                    total_loss[key] += loss_dict[key]
                else:
                    total_loss[key] = loss_dict[key]


            
        for key in total_loss.keys():
            self.writer.add_scalar(key, total_loss[key] / len(self.train_loader), epoch)

        
        print("Epoch {}|Time {}|Training Loss: {:.5f}".format(
            epoch, time.time() - start_time, total_loss["loss"] / len(self.train_loader)))

            


    def train(self):
        self.prepare_loaders()
        self.build_model()
        self.make_experiments_dirs()
        self.writer = SummaryWriter(log_dir = self.runs_dir)

        start_epoch = 0
        if self.config["resume_training"]:
            model_path = os.path.join(self.checkpoints_dir, str(self.config["resume_from"]) + "epoch")
            self.model.load_state_dict(torch.load(model_path, map_location=self.config["device"]))
            start_epoch = self.config["resume_from"]
            print("successfully loaded model starting from " + str(self.config["resume_from"]) + " epoch") 
        
        for epoch in range(start_epoch + 1, self.config["train"]["epochs"]):
            self.train_one_epoch(epoch)

            if epoch % self.config["train"]["save_every"] == 0:
                path = os.path.join(self.checkpoints_dir, str(epoch) + "epoch")
                torch.save(self.model.state_dict(), path)

            if (epoch + 1) % self.config["val"]["val_every"] == 0:
                self.validate(epoch)

            self.scheduler.step()


    def validate(self, epoch):
        total_loss = {}
        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for data in self.val_loader:
                voxel = data["voxel"].to(self.device)
                boxes = data["boxes"].to(self.device)
                data_type = data["data_type"].to(self.device)
                heatmap = data["heatmap"].to(self.device)

                pred = self.model(voxel)
                loss_dict = self.criterion(pred, boxes, heatmap, data_type)

                loss_dict["loss"] = loss_dict["loss"].item()

                for key in loss_dict.keys():
                    if key in total_loss:
                        total_loss[key] += loss_dict[key]
                    else:
                        total_loss[key] = loss_dict[key]

            

        self.model.train()

        for key in total_loss.keys():
            self.writer.add_scalar(key, total_loss[key] / len(self.train_loader), epoch)

        print("Epoch {}|Time {}|Validation Loss: {:.5f}".format(
            epoch, time.time() - start_time, total_loss["loss"] / len(self.val_loader)))

        if total_loss["loss"] / len(self.val_loader) < self.prev_val_loss:
            self.prev_val_loss = total_loss["loss"] / len(self.val_loader)
            path = os.path.join(self.best_checkpoints_dir, str(epoch) + "epoch")
            torch.save(self.model.state_dict(), path)
            


    def make_experiments_dirs(self):
        base = self.config["model"] + "_" + self.config["note"] + "_" + self.config["date"] + "_" + str(self.config["ver"])
        path = os.path.join(self.config["experiments"], base)
        if not os.path.exists(path):
            os.mkdir(path)
        self.checkpoints_dir = os.path.join(path, "checkpoints")
        self.best_checkpoints_dir = os.path.join(path, "best_checkpoints")
        self.runs_dir = os.path.join(path, "runs")

        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)

        if not os.path.exists(self.best_checkpoints_dir):
            os.mkdir(self.best_checkpoints_dir)

        if not os.path.exists(self.runs_dir):
            os.mkdir(self.runs_dir)