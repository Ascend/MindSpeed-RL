import hydra

from mindspeed_rl import MegatronConfig


@hydra.main(config_path='./configs', config_name='grpo_trainer_llama32_1b')
def main(config):
    print(config)
    actor_config = MegatronConfig(config.get("actor_config"), config.get('model'))
    print(actor_config.items())


main()