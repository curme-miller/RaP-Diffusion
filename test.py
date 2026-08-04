import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf
from utils import util_image
from train import Trainer
import torch
import numpy as np
from model.datasets import create_dataset
import torch.utils.data as udata


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--cfg_path", type=str, default="./config/config.yaml", help="Configs of yaml file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_path", type=str, default="./validation_images", help="Output path for test results")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs to use")
    args = parser.parse_args()
    return args

def setup_environment(args):
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        print(f"Using GPUs: {args.gpu_ids}")

class TestTrainer:
    def __init__(self, trainer, output_path):
        self.trainer = trainer
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def test(self):
        print("Starting testing...")
        if hasattr(self.trainer, 'ema_model') and self.trainer.configs.train.get('use_ema_val', True):
            self.trainer.reload_ema_model()
            model = self.trainer.ema_model
            print("Using EMA model for testing")
        else:
            model = self.trainer.model
            print("Using main model for testing")
        model.eval()
        
        if 'val' in self.trainer.dataloaders:
            dataloader = self.trainer.dataloaders['val']
            print(f"Testing on validation dataset with {len(self.trainer.datasets['val'])} images")
        else:
            print("No validation dataset found, creating test dataloader")
            test_dataset = create_dataset(self.trainer.configs.data.test)
            dataloader = udata.DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )
        num_processed = 0
        for i, data in enumerate(dataloader):
            data = self.trainer.prepare_data(data, phase='val')
            
            if 'gt' in data:
                im_x0, im_gt = data['x0'], data['gt']
                has_gt = True
            else:
                im_x0 = data['x0']
                has_gt = False
            img_name = data['img_name'][0]
            if self.trainer.configs.model.params.cond_x0:
                model_kwargs = {'x0': data['x0']}
                if 'mask' in data:
                    model_kwargs['mask'] = data['mask']
            else:
                model_kwargs = None
            result = None
            for sample in self.trainer.base_diffusion.p_sample_loop_progressive(
                y=im_x0,
                model=model,
                first_stage_model=self.trainer.autoencoder,
                noise=None,
                clip_denoised=True if self.trainer.autoencoder is None else False,
                model_kwargs=model_kwargs,
                device=f"cuda:{self.trainer.rank}",
                progress=False,
            ):
                result = sample
            if result is not None:
                final_result = self.trainer.base_diffusion.decode_first_stage(
                    result['sample'], self.trainer.autoencoder
                ).clamp(-1.0, 1.0)
                for j in range(final_result.shape[0]):
                    result_img = (final_result[j] * 0.5 + 0.5).clamp(0, 1)
                    result_np = result_img.cpu().permute(1, 2, 0).numpy()
                    result_filename = self.output_path / f"{img_name}.png"
                    util_image.imwrite(result_np, str(result_filename))
                    if has_gt:
                        x0_img = (im_x0[j] * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                        gt_img = (im_gt[j] * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                        comparison = np.concatenate([x0_img, result_np, gt_img], axis=1)
                        comparison_filename = self.output_path / f"comparison_{img_name}.png"
                        util_image.imwrite(comparison, str(comparison_filename))
                    num_processed += 1
                    print(f"Processed image {num_processed}")
        print(f"Testing completed! Processed {num_processed} images")
        print(f"Results saved to: {self.output_path}")

def main():
    print("=== Test Script ===")
    print("Testing trained model")
    args = get_parser()
    setup_environment(args)
    print(f"Loading config from: {args.cfg_path}")
    configs = OmegaConf.load(args.cfg_path)
    configs.model.ckpt_path = args.checkpoint_path
    configs.train.use_swanlab = False
    configs.resume = ""
    configs.save_dir = "./experiments/test"
    configs.cfg_path = args.cfg_path
    configs.data.train.params.dir_path = args.test_data_path + '/A'
    configs.data.train.params.dir_path_extra = args.test_data_path + '/B'
    configs.data.val.params.dir_path = args.test_data_path + '/A'
    configs.data.val.params.dir_path_extra = args.test_data_path + '/B'
    configs.data.val.params.length = None
    configs.data.test = {
        'type': 'paired',
        'params': {
            'dir_path': args.test_data_path + '/A',
            'dir_path_extra': args.test_data_path + '/B',
            'transform_type': 'default',
            'transform_kwargs': {
                'pass_aug': True,
                'pch_size': 256,
                'mean': 0.5,
                'std': 0.5,
                'max_value': 255
            },
            'im_exts': 'png',
            'length': None,
            'recursive': False
        }
    }
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(configs)
    test_trainer = TestTrainer(trainer, args.output_path)
    test_trainer.test()

if __name__ == "__main__":
    main()