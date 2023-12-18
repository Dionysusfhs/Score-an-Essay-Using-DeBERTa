import torch
import torch.nn as nn

from tqdm import tqdm  # progress meter
import gc  # Garbage Collection
import wandb
import numpy as np

# funcs to train & val & test etc.
class Trainer:
    def __init__(self, model, loaders, config):
        self.model = model
        self.train_loader, self.val_loader = loaders
        self.config = config
        self.input_keys = ['input_ids', 'token_type_ids', 'attention_mask']

        self.criterion_options = {
            'SmoothL1Loss': nn.SmoothL1Loss(),
            'L1Loss': nn.L1Loss(),
            'MSELoss': nn.MSELoss()
        }
        self.criterion = self.criterion_options[self.config.criterion]

        self.optimizer = self._get_optim()

        self.scheduler_options = {
            'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
            (self.optimizer, T_0=self.config.T_0, eta_min=self.config.eta_min),

            'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=6),
            'StepLR': torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2),
        }
        self.scheduler = self.scheduler_options[self.config.scheduler]

        # histories
        self.train_losses = []
        self.valid_losses = []

        self.valid_mcrmse = []

    def _get_optim(self, weight_decay=0.01):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0},


        ]
        optimizer = torch.optim.AdamW(optimizer_parameters,
                                      lr=self.config.lr, eps=self.config.eps, betas=(self.config.b1, self.config.b2))

        return optimizer

    def mcrmse(self, outputs, targets):
        colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
        loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
        return loss

    def train_one_epoch(self, epoch):
        running_loss = 0.

        progress = tqdm(self.train_loader, total=len(self.train_loader))
        for i, (inputs, targets) in enumerate(progress):
            # for i, (inputs, targets) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            inputs = {k: inputs[k].to(device=self.config.device)
                      for k in inputs.keys()}
            targets = targets['labels'].to(device=self.config.device)

            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)

            running_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            if self.config.scheduler == 'CosineAnnealingWarmRestarts':
                self.scheduler.step(epoch-1+i/len(self.train_loader))

            # wandb
            if self.config.use_wandb:
                wandb.log({"loss": loss,
                           "lr": self.scheduler.get_last_lr()[0]})

            del inputs, targets, outputs, loss

        if self.config.scheduler == 'StepLR':
            self.scheduler.step()

        train_loss = running_loss/len(self.train_loader)
        self.train_losses.append(train_loss)

    @torch.no_grad()
    def valid_one_epoch(self, epoch):
        running_loss = 0.
        running_mcrmse = 0.

        progress = tqdm(self.val_loader, total=len(self.val_loader))
        for (inputs, targets) in progress:
            # for (inputs, targets) in self.val_loader:

            inputs = {k: inputs[k].to(device=self.config.device)
                      for k in inputs.keys()}
            targets = targets['labels'].to(device=self.config.device)

            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)
            running_loss += loss.item()

            running_mcrmse += self.mcrmse(outputs, targets).item()

            del inputs, targets, outputs, loss

        val_loss = running_loss/len(self.val_loader)
        self.valid_losses.append(val_loss)

        self.valid_mcrmse.append(running_mcrmse/len(self.val_loader))
        del running_mcrmse

        if self.config.scheduler == 'ReduceLROnPlateau':
            self.scheduler.step(val_loss)

    def test(self, test_loader):

        preds = []
        self.model.eval()
        # iterate test_loader
        for (inputs) in test_loader:
            inputs = {k: inputs[k].to(device=self.config.device)
                      for k in inputs.keys()}

            outputs = self.model(inputs)
            preds.append(outputs.detach().cpu())

        preds = torch.concat(preds)
        return preds

    def fit(self):

        progress = tqdm(range(self.config.epoch+1, self.config.epoch +
                        self.config.n_epochs + 1), total=self.config.n_epochs)
        for epoch in progress:
            # for epoch in range(self.config.epoch+1, self.config.epoch+self.config.n_epochs + 1):
            current_lr = self.optimizer.param_groups[0]['lr']

            self.model.train()
            self.train_one_epoch(epoch)

            self.model.eval()
            self.valid_one_epoch(epoch)

            print(
                f"{'-' * 30} EPOCH {epoch} / {self.config.n_epochs+self.config.epoch} {'-' * 30}")
            print('current lr: {:.7f}'.format(current_lr))

            print("train Loss: %.7f" % self.train_losses[-1], end=',\t\t')
            print("valid Loss: %.7f" % self.valid_losses[-1])
            print("valid MCRMSE: %.7f" % self.valid_mcrmse[-1])

            # save model
            if self.valid_mcrmse[-1] <= min(self.valid_mcrmse):
                torch.save(self.model.state_dict(),
                           self.config.checkpoint_path)

            # wandb
            if self.config.use_wandb:
                wandb.log({
                    "train_loss": self.train_losses[-1],
                    "val_loss": self.valid_losses[-1],
                    "val_mcrmse": self.valid_mcrmse[-1], })

        print(f"{'-' * 30} FINISH {'-' * 30}")
        print(f"MIN valid Loss: {min(self.valid_losses)}")
        print(f"MIN valid MCRMSE: {min(self.valid_mcrmse)}")

    def clear(self):
        gc.collect()
        torch.cuda.empty_cache()

# create dataset
class EssayDataset:
    def __init__(self, df, config, tokenizer=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.classes = ['cohesion', 'syntax', 'vocabulary',
                        'phraseology', 'grammar', 'conventions']
        self.max_len = config.max_length
        self.tokenizer = tokenizer
        self.config = config
        self.is_test = is_test

    def __getitem__(self, idx):
        sample = self.df['full_text'][idx]

        tokenized = self.tokenizer.encode_plus(sample,
                                               None,
                                               add_special_tokens=True,
                                               max_length=self.max_len,
                                               truncation=True,
                                               padding='max_length'
                                               )
        inputs = {
            "input_ids": torch.tensor(tokenized['input_ids'], dtype=torch.long),
            "token_type_ids": torch.tensor(tokenized['token_type_ids'], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized['attention_mask'], dtype=torch.long)
        }

        if self.is_test == True:
            return inputs

        label = self.df.loc[idx, self.classes].to_list()
        targets = {
            "labels": torch.tensor(label, dtype=torch.float32),
        }

        return inputs, targets

    def __len__(self):
        return len(self.df)
