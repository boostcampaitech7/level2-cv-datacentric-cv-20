import os, sys
from argparse import ArgumentParser
from torch import cuda
from data_loader.transform import get_train_transform
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config 

def parse_args(type):
    parser_path = ArgumentParser()
    parser_path.add_argument('--configs', type=str, default="./configs/default.yaml")
    config_args, _ = parser_path.parse_known_args()

    arge = load_config(config_args.configs)
    argWandb = arge['wandb']
    parser = ArgumentParser()

    if type == 'train':
        arge = arge[type]       
        
        # Conventional args
        parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', arge['data_dir']))
        parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', arge['model_dir']))
        parser.add_argument('--device', default=arge['device'] if cuda.is_available() else 'cpu')
        parser.add_argument('--num_workers', type=int, default=arge['num_workers'])
        parser.add_argument('--image_size', type=int, default=arge['image_size'])
        parser.add_argument('--input_size', type=int, default=arge['input_size'])
        parser.add_argument('--batch_size', type=int, default=arge['batch_size'])
        parser.add_argument('--learning_rate', type=float, default=arge['learning_rate'])
        parser.add_argument('--max_epoch', type=int, default=arge['max_epoch'])
        # parser.add_argument('--save_interval', type=int, default=arge['save_interval'])
        parser.add_argument('--project_name', type=str, default=argWandb['project_name'])
        parser.add_argument('--model_name', type=str, default=argWandb['model_name']) 
        parser.add_argument('--entity', type=str, default=argWandb['entity']) 

        parser.add_argument('--dataset', type=str, default=arge['dataset'])
        parser.add_argument('--valid', type=str, default=arge['valid'])
        parser.add_argument('--resume', type=str, default=arge['resume'])

    elif type == 'inference':
        arge = arge[type]
        parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', arge['data_dir']))
        parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', arge['model_dir']))
        parser.add_argument('--device', default=arge['device'] if cuda.is_available() else 'cpu')
        parser.add_argument('--input_size', type=int, default=arge['input_size'])
        parser.add_argument('--batch_size', type=int, default=arge['batch_size'])
        parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', arge['output_dir']))

    elif type == 'valid':
        arge = arge[type]
        parser.add_argument('--num_workers', type=int, default=arge['num_workers'])
        parser.add_argument('--input_size', type=int, default=arge['input_size'])
        parser.add_argument('--batch_size', type=int, default=arge['batch_size'])
        parser.add_argument('--project_name', type=str, default=argWandb['project_name'])
        parser.add_argument('--model_name', type=str, default=argWandb['model_name']) 
  

    args = parser.parse_args()
    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')
    print(arge)
    return args

