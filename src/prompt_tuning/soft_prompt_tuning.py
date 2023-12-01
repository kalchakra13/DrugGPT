import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class SoftPromptTuner:
    """
        A class dedicated to fine-tuning the soft prompts for knowledge acquisition tasks.

        This class facilitates the training and validation processes of the soft prompts, which are learnable
        parameters used to adapt the LLaMA model for specific tasks without altering the original pre-trained model weights.

        Attributes:
            llama_utils (LLaMAUtils): An instance of the LLaMAUtils class for LLaMA model operations.
            config (dict): A configuration dictionary containing training parameters.
            criterion (torch.nn.Module): The loss function used for training.
            writer (SummaryWriter): Tensorboard writer for logging training metrics.
            best_val_loss (float): The best validation loss achieved during training.

        Methods:
            get_optimizer(): Initializes and returns the optimizer based on configuration.
            get_lr_scheduler(optimizer, num_training_steps, num_warmup_steps): Creates a learning rate scheduler.
            train_epoch(train_loader, optimizer, lr_scheduler, epoch): Conducts training for one epoch.
            validate_epoch(val_loader, epoch): Conducts validation for one epoch.
            save_checkpoint(filename="soft_prompt_checkpoint.pth"): Saves the current best model checkpoint.
            load_checkpoint(filename): Loads a model checkpoint.
            train(train_loader, val_loader): Initiates the training and validation process over multiple epochs.
        """

    def __init__(self, llama_utils, config):
        self.llama_utils = llama_utils
        self.config = config
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.writer = SummaryWriter()
        self.best_val_loss = float('inf')

    def get_optimizer(self):
        if self.config['optimizer'] == 'Adam':
            return optim.Adam(self.llama_utils.soft_embedding.parameters(),
                              lr=self.config['learning_rate'],
                              weight_decay=self.config['weight_decay'])
        else:
            raise ValueError(f"Invalid optimizer name: {self.config['optimizer']}")

    def get_lr_scheduler(self, optimizer, num_training_steps, num_warmup_steps):
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def train_epoch(self, train_loader, optimizer, lr_scheduler, epoch):
        self.llama_utils.model.train()
        epoch_loss = 0
        for batch_idx, (input_ids, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            input_ids, labels = input_ids.to(self.llama_utils.device), labels.to(self.llama_utils.device)
            optimizer.zero_grad()
            logits = self.llama_utils.llama_training_forward(input_ids)

            # Adjust labels to match the logits shape
            labels = labels[:, :logits.shape[1]]
            labels[labels[:, :logits.shape[1]].shape[1]:] = -100

            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()

            if self.config['gradient_clip']:
                torch.nn.utils.clip_grad_norm_(self.llama_utils.soft_embedding.parameters(),
                                               self.config['gradient_clip'])

            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            self.writer.add_scalar("Training Loss", loss.item(), epoch * len(train_loader) + batch_idx)
            self.writer.add_scalar("Learning Rate", lr_scheduler.get_last_lr()[0],
                                   epoch * len(train_loader) + batch_idx)

        return epoch_loss / len(train_loader)

    def validate_epoch(self, val_loader, epoch):
        self.llama_utils.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (input_ids, labels) in enumerate(val_loader):
                input_ids, labels = input_ids.to(self.llama_utils.device), labels.to(self.llama_utils.device)
                logits = self.llama_utils.llama_training_forward(input_ids)

                labels = labels[:, :logits.shape[1]]
                labels[labels[:, :logits.shape[1]].shape[1]:] = -100

                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        self.writer.add_scalar("Validation Loss", val_loss, epoch)

        return val_loss

    def save_checkpoint(self, filename="soft_prompt_checkpoint.pth"):
        checkpoint = {'soft_prompt_params': self.soft_embedding.state_dict()}
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved with validation loss: {self.best_val_loss:.4f}")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.soft_embedding.load_state_dict(checkpoint['soft_prompt_params'])
        print("Checkpoint loaded.")

    def train(self, train_loader, val_loader):
        optimizer = self.get_optimizer()

        num_training_steps = len(train_loader) * self.config['epochs']
        num_warmup_steps = int(num_training_steps * self.config['warmup_ratio'])
        lr_scheduler = self.get_lr_scheduler(optimizer, num_training_steps, num_warmup_steps)

        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(train_loader, optimizer, lr_scheduler, epoch)
            print(f"Epoch [{epoch + 1}/{self.config['epochs']}], Train Loss: {train_loss:.4f}")

            val_loss = self.validate_epoch(val_loader, epoch)
            print(f"Validation Loss: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint()

        self.writer.close()

