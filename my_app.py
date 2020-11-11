from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(config_name="config")
def my_app(cfg: DictConfig) -> None:
    assert cfg.node.loompa == 10
    assert cfg["node"]["loompa"] == 10

    assert cfg.node.zippity == 10
    assert isinstance(cfg.node.zippity, int)
    assert cfg.node.do == "oompa 10"

    cfg.node.waldo


if __name__ == "__main__":
    my_app()
