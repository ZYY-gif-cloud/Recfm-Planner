import os
from flow_matching_policy.workspace.base_workspace import BaseWorkspace
import pathlib
from omegaconf import OmegaConf
import hydra
from termcolor import cprint
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
os.environ['WANDB_SILENT'] = "True"
os.environ['NUMEXPR_MAX_THREADS'] = '16'
OmegaConf.register_new_resolver("eval", eval, replace=True)
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('config')),
    config_name="Train_CFM"
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()
if __name__ == "__main__":
    main()
