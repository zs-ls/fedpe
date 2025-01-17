import yaml
def get_config(path):
    with open(path, 'r') as file:
        config_data = yaml.safe_load(file)
        return Config(config_data)


class Config:
    def __init__(self, args):
        # 将传入的数据转换为属性
        if isinstance(args, dict):
            for key, value in args.items():
                # 遍历字典项并递归转换
                setattr(self, key, Config(value) if isinstance(value, dict) else value)
        else:
            self.args = args

    def __repr__(self):
        return f"Config({self.__dict__})"