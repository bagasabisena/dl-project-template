import datetime
from pathlib import Path
import subprocess
import sys

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.neptune import NeptuneLogger
import hydra

OmegaConf.register_new_resolver("dateid", lambda: datetime.datetime.now().strftime("%Y%m%d%H%M%S"))


def modify_argv():
    # a hack to remove --job-dir that is passed by the google ai platform
    # so it wont interfere with Hydra interface
    print('default argv')
    print(sys.argv)
    argv = [arg for arg in sys.argv if not arg.startswith('--job-dir')]
    # now for the rest args with --, we remove it. it might be params for tuning
    # removing -- makes it compatible with hydra
    argv = [arg.replace('--', '') for arg in argv]
    sys.argv = argv
    print('resulting argv')
    print(sys.argv)


@hydra.main(config_path='conf/', config_name='config')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # insttiate the model using hydra initiate
    model = hydra.utils.instantiate(cfg.model)

    # initiate datamodule
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # initiate loggers
    loggers = []
    if "logger" in cfg:
        for k in cfg.logger:
            loggers.append(hydra.utils.instantiate(cfg['logger'][k]))
    
    for logger in loggers:
        if isinstance(logger, NeptuneLogger):
            logger.experiment["params"] = cfg
    
    # initiate callbacks
    callbacks = []
    if "callback" in cfg:
        for k in cfg.callback:
            callbacks.append(hydra.utils.instantiate(cfg['callback'][k]))
    
    trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)

    # run validate separately to run validation eval on the optimized model
    trainer.validate(datamodule=datamodule)

    # uncomment if you want to run the test
    # trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    # uncomment this line if you use the Google AI Platform Training for training
    # modify_argv()
    main()
