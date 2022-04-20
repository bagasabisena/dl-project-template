import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.aggregation import MeanMetric

END_EPOCH_LOG = '{} for epoch {} is finished. Loss: {}. Acc: {}'


class MyModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, learning_rate) -> None:
        super().__init__()
        # all arguments will be saved by pytorch lightning
        # and can be accessed through self.hparams
        # this has the benefit of easier parameter loading from the checkpoint
        # dont sent to logger, so we do it manually
        # this benefit tensorboard logger where we can push metrics in the hparams
        self.save_hyperparameters(logger=False)
        self.layer = nn.Linear(self.hparams.input_dim, self.hparams.output_dim)

        # torchmetrics for easier metrics calculation and aggregation
        # of course for task other than classification you can use other metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, inputs):
        outputs = self.layer(inputs)
        return outputs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def on_fit_start(self) -> None:
        # this enables pushing hparams to tensorboard, so we can track metrics
        # and the corresponding hyperparameters easily on tensorboard
        # setup initial metrics here for hparams
        metrics = {
            'epoch_metrics/val_loss': 0,
            'epoch_metrics/val_acc': 10
        }
        self.logger.log_hyperparams(self.hparams, metrics=metrics)

    def training_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)

        loss = F.cross_entropy(outputs, y)
        # this will update the state so we can agg later
        # no loss is computed since this is just a avgmetrics
        loss_val = loss.detach()
        self.train_loss.update(loss_val)
        self.log("step_metrics/train_loss", loss_val)

        # this will calculate the accuracy and also update the state so we can agg later
        acc_val = self.train_accuracy(outputs.detach(), y)
        self.log("step_metrics/train_acc", acc_val)
        return loss

    def training_epoch_end(self, step_outputs) -> None:
        # compute metrics aggregate across steps
        loss_val = self.train_loss.compute()
        acc_val = self.train_accuracy.compute()

        # compute aggregate the loss across steps, which avg it (AvgMetric)
        self.log("epoch_metrics/train_loss", loss_val)
        self.log("epoch_metrics/train_acc", acc_val)
        print(END_EPOCH_LOG.format("Training", self.current_epoch, loss_val, acc_val))

        # reset the accuracy so it wont accumulate across epoch
        self.train_loss.reset()
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)

        loss = F.cross_entropy(outputs, y)
        loss_val = loss.detach()
        self.val_loss.update(loss_val)
        self.log("step_metrics/val_loss", loss_val)

        acc_val = self.val_accuracy(outputs.detach(), y)
        self.log("step_metrics/val_acc", acc_val)
        return loss

    def validation_epoch_end(self, step_outputs) -> None:
        loss_val = self.val_loss.compute()
        acc_val = self.val_accuracy.compute()
        print(END_EPOCH_LOG.format("Validation", self.current_epoch, loss_val, acc_val))
        self.log("epoch_metrics/val_loss", loss_val)
        self.log("epoch_metrics/val_acc", acc_val)

        self.val_accuracy.reset()
        self.val_loss.reset()

    def test_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        self.test_accuracy.update(outputs.detach(), y)

    def test_epoch_end(self, step_outputs) -> None:
        acc_val = self.test_accuracy.compute()
        self.log("epoch_metrics/test_acc", acc_val)

        self.test_accuracy.reset()
