import os, math, random, datetime, functools, argparse
import lpips
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from contextlib import nullcontext
import swanlab
import torch
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from model.datasets import create_dataset
from utils import util_net, util_common, util_image
import warnings
warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated. Please install nvidia-ml-py instead",
    category=FutureWarning,
    module="torch.cuda"
)
warnings.filterwarnings(
    "ignore",
    message="The parameter 'pretrained' is deprecated since 0.13",
    category=UserWarning,
    module="torchvision.models._utils"
)
warnings.filterwarnings(
    "ignore",
    message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13",
    category=UserWarning,
    module="torchvision.models._utils"
)

console = Console()

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--save_dir",type=str,default="./experiments/ctm_training",help="Folder to save the checkpoints and training log",)
    parser.add_argument("--resume",type=str,const=True,default="",nargs="?",help="resume from the save_dir or checkpoint",)
    parser.add_argument("--cfg_path",type=str,default="./config/config.yaml",help="Configs of yaml file",)
    parser.add_argument("--data_root",type=str,default="/mnt/zlx/clash/data/StM",help="Root directory of the dataset",)
    parser.add_argument("--gpu_ids",type=str,default="0",help="GPU IDs to use (e.g., '0' or '0,1,2,3')",)
    parser.add_argument("--batch_size",type=int,default=4,help="Training batch size per GPU",)
    parser.add_argument("--learning_rate",type=float,default=5e-5,help="Learning rate",)
    parser.add_argument("--iterations",type=int,default=100000,help="Total training iterations",)
    args = parser.parse_args()
    return args

def extract_dataset_name(data_path):
    path = Path(data_path)
    if 'train' in path.parts:
        train_idx = path.parts.index('train')
        if train_idx > 0:
            return path.parts[train_idx - 1]
    return path.parent.parent.name

def save_validation_images(x0, gt, pred, save_path, tag, step):
    x0 = (x0 * 0.5 + 0.5).clamp(0, 1)
    gt = (gt * 0.5 + 0.5).clamp(0, 1) 
    pred = (pred * 0.5 + 0.5).clamp(0, 1)

    x0_np = x0.cpu().permute(1, 2, 0).numpy()
    gt_np = gt.cpu().permute(1, 2, 0).numpy()
    pred_np = pred.cpu().permute(1, 2, 0).numpy()
    
    combined = np.concatenate([x0_np, gt_np, pred_np], axis=1)
    
    filename = f"{tag}_step_{step:06d}.png"
    full_path = save_path / filename
    util_image.imwrite(combined, str(full_path))

def upload_random_validation_images(val_images_dir, num_images=10):
    try:
        image_files = list(val_images_dir.glob("*.png")) + list(val_images_dir.glob("*.jpg"))
        if len(image_files) == 0:
            return []
        num_to_select = min(num_images, len(image_files))
        selected_files = random.sample(image_files, num_to_select)
        swanlab_images = []
        for img_path in selected_files:
            swanlab_images.append(swanlab.Image(str(img_path), caption=img_path.name))
        return swanlab_images
    except Exception as e:
        print(f"Error uploading validation images: {e}")
        return []

class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.dataset_name = extract_dataset_name(configs.data.train.params.dir_path)
        self.setup_dist()
        self.setup_seed()
        self.setup_paths()
        self.init_logger()
        self.init_swanlab()
        self.build_model()
        self.setup_optimization()
        self.resume_from_ckpt()
        self.build_dataloader()

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()
        
        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                timeout=datetime.timedelta(seconds=3600),
                backend='nccl',
                init_method='env://',
            )
        
        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.global_seeding
            assert isinstance(global_seeding, bool)
        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def setup_paths(self):
        self.base_dir = Path("./")
        self.ckpt_dir = self.base_dir / "checkpoints" / self.dataset_name
        if self.rank == 0 and not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True)
        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate
            self.ema_ckpt_dir = self.ckpt_dir / "ema"
            if self.rank == 0 and not self.ema_ckpt_dir.exists():
                self.ema_ckpt_dir.mkdir(parents=True)
        self.val_images_dir = self.base_dir / "validation_images" / self.dataset_name
        if self.rank == 0 and not self.val_images_dir.exists():
            self.val_images_dir.mkdir(parents=True)
        self.best_lpips = float('inf')
        self.max_checkpoints = 20

    def init_logger(self):
        if self.rank == 0:
            log_path = Path("training.log")
            self.logger = logger
            self.logger.remove()
            self.logger.add(
                log_path, 
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                mode='a', 
                level='INFO'
            )
            self.logger.info("=" * 80)
            self.logger.info(f"Training started for dataset: {self.dataset_name}")
            self.logger.info(f"Checkpoint directory: {self.ckpt_dir}")
            self.logger.info(f"Validation images directory: {self.val_images_dir}")
            self.logger.info("=" * 80)

    def init_swanlab(self):
        """Initialize SwanLab for experiment tracking"""
        if self.rank == 0:
            use_swanlab = self.configs.train.get('use_swanlab', True)
            if use_swanlab:
                try:
                    # Initialize SwanLab
                    swanlab.init(
                        project=f"Rap-Diffusion",
                        name=f"{self.configs.train.swanlab_name}",
                        config=OmegaConf.to_container(self.configs, resolve=True),
                        tags=[self.dataset_name, "diffusion"]
                    )
                    self.use_swanlab = True
                    self.logger.info("✅ SwanLab initialized successfully")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to initialize SwanLab: {e}")
                    self.use_swanlab = False
            else:
                self.use_swanlab = False
                self.logger.info("SwanLab disabled in config")
        else:
            self.use_swanlab = False

    def build_model(self):
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        model.cuda()
        
        if self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(model, ckpt)
        
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling model...")
            model = torch.compile(model, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        
        if self.num_gpus > 1:
            self.model = DDP(model, device_ids=[self.rank,], static_graph=False)
        else:
            self.model = model

        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_model = deepcopy(model).cuda()
            self.ema_state = OrderedDict(
                {key:deepcopy(value.data) for key, value in self.model.state_dict().items()}
            )
            self.ema_ignore_keys = [x for x in self.ema_state.keys() if ('running_' in x or 'num_batches_tracked' in x)]
            self.ema_ignore_keys.extend([x for x in self.ema_state.keys() if 'relative_position_index' in x])

        if self.configs.autoencoder is not None:
            ckpt = torch.load(self.configs.autoencoder.ckpt_path, map_location=f"cuda:{self.rank}")
            if self.rank == 0:
                self.logger.info(f"Restoring autoencoder from {self.configs.autoencoder.ckpt_path}")
            params = self.configs.autoencoder.get('params', dict)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder.cuda()
            self.load_model(autoencoder, self.configs.autoencoder.ckpt_path, tag='autoencoder', strict=True)
            
            if not self.configs.autoencoder.tune_decoder:
                self.freeze_model(autoencoder)
                autoencoder.eval()
            
            if self.configs.train.compile.flag:
                if self.rank == 0:
                    self.logger.info("Begin compiling autoencoder model...")
                autoencoder = torch.compile(autoencoder, mode=self.configs.train.compile.mode)
                if self.rank == 0:
                    self.logger.info("Compiling Done")
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

        # Build LPIPS metric
        if hasattr(self.configs, 'lpips'):
            lpips_net = self.configs.lpips.net
        else:
            lpips_net = 'vgg'
        if self.rank == 0:
            self.logger.info(f"Loading LPIPS Metric: {lpips_net}...")
        lpips_loss = lpips.LPIPS(net=lpips_net).to(f"cuda:{self.rank}")
        for params in lpips_loss.parameters():
            params.requires_grad_(False)
        lpips_loss.eval()
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling LPIPS Metric...")
            lpips_loss = torch.compile(lpips_loss, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        self.lpips_loss = lpips_loss

        # Build diffusion
        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)

        # Model information
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            self.logger.info(f"Number of parameters: {num_params:.4f}M")

    def setup_optimization(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.configs.train.lr,
            weight_decay=self.configs.train.weight_decay
        )

        if self.configs.train.lr_schedule == 'cosin':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                eta_min=self.configs.train.lr_min,
            )
        self.amp_scaler = torch.amp.GradScaler('cuda' ) if self.configs.train.use_amp else None

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: 
                yield from loader
        datasets = {'train': create_dataset(self.configs.data.get('train', dict)), }
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.get('val', dict))
        if self.rank == 0:
            for phase in datasets.keys():
                length = len(datasets[phase])
                self.logger.info(f'Number of images in {phase} data set: {length}')
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                datasets['train'],
                num_replicas=self.num_gpus,
                rank=self.rank,
            )
        else:
            sampler = None
        dataloaders = {'train': _wrap_loader(udata.DataLoader(
                    datasets['train'],
                    batch_size=self.configs.train.batch[0] // self.num_gpus,
                    shuffle=False if self.num_gpus > 1 else True,
                    drop_last=True,
                    num_workers=min(self.configs.train.num_workers, 4),
                    pin_memory=True,
                    prefetch_factor=self.configs.train.get('prefetch_factor', 2),
                    worker_init_fn=my_worker_init_fn,
                    sampler=sampler,
                ))}
        
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(
                datasets['val'],
                batch_size=self.configs.train.batch[1],
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

    def load_model(self, model, ckpt_path=None, tag='model', strict=True):
        if self.rank == 0:
            self.logger.info(f'Loading {tag} from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        if strict:
            util_net.reload_model(model, ckpt)
        else:
            model.load_state_dict(ckpt, strict=False)
        if self.rank == 0:
            self.logger.info('Loaded Done')

    def prepare_data(self, data, dtype=torch.float32, phase='train'):
        if phase == 'val':
            offset = self.configs.train.get('val_resolution', 256)
            data2 = data['img_name']
            del data['img_name']
            for key, value in data.items():
                h, w = value.shape[2:]
                if h > offset and w > offset:
                    h_end = int((h // offset) * offset)
                    w_end = int((w // offset) * offset)
                    data[key] = value[:, :, :h_end, :w_end]
                else:
                    h_pad = math.ceil(h / offset) * offset - h
                    w_pad = math.ceil(w / offset) * offset - w
                    padding_mode = self.configs.train.get('val_padding_mode', 'reflect')
                    data[key] = F.pad(value, pad=(0, w_pad, 0, h_pad), mode=padding_mode)
                    
            data = {key:value.cuda().to(dtype=dtype) for key, value in data.items()}  
            data['img_name'] = data2
            return data
        else:
            data2 = data['img_name']
            del data['img_name']
            data = {key:value.cuda().to(dtype=dtype) for key, value in data.items()}  
            data['img_name'] = data2
            return data

    def save_ckpt(self, is_best=False):
        if self.rank == 0:
            ckpt = {
                'iters_start': self.current_iters,
                'state_dict': self.model.state_dict(),
            }
            if self.amp_scaler is not None:
                ckpt['amp_scaler'] = self.amp_scaler.state_dict()
            
            if is_best:
                best_ckpt_path = self.ckpt_dir / 'best_model.pth'
                torch.save(ckpt, best_ckpt_path)
                self.logger.info(f"Best model saved: {best_ckpt_path}")
                if hasattr(self, 'ema_rate'):
                    best_ema_path = self.ema_ckpt_dir / 'best_ema_model.pth'
                    torch.save(self.ema_state, best_ema_path)
                    self.logger.info(f"Best EMA model saved: {best_ema_path}")
            ckpt_path = self.ckpt_dir / f'model_{self.current_iters:06d}.pth'
            torch.save(ckpt, ckpt_path)
            if hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / f'ema_model_{self.current_iters:06d}.pth'
                torch.save(self.ema_state, ema_ckpt_path)
            self.logger.info(f"Checkpoint saved: {ckpt_path}")
            self.cleanup_old_checkpoints()
    
    def cleanup_old_checkpoints(self):
        if self.rank == 0:
            model_files = list(self.ckpt_dir.glob('model_*.pth'))
            model_files = [f for f in model_files if f.name != 'best_model.pth']
            model_files.sort(key=lambda x: x.stat().st_mtime)
            
            if len(model_files) > self.max_checkpoints:
                files_to_remove = model_files[:-self.max_checkpoints]
                for file_path in files_to_remove:
                    try:
                        file_path.unlink()
                        self.logger.info(f"Removed old checkpoint: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove {file_path}: {e}")
            
            if hasattr(self, 'ema_rate'):
                ema_files = list(self.ema_ckpt_dir.glob('ema_model_*.pth'))
                ema_files = [f for f in ema_files if f.name != 'best_ema_model.pth']
                ema_files.sort(key=lambda x: x.stat().st_mtime)
                
                if len(ema_files) > self.max_checkpoints:
                    files_to_remove = ema_files[:-self.max_checkpoints]
                    for file_path in files_to_remove:
                        try:
                            file_path.unlink()
                            self.logger.info(f"Removed old EMA checkpoint: {file_path}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove {file_path}: {e}")

    def resume_from_ckpt(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth") and os.path.isfile(self.configs.resume)
            if self.rank == 0:
                self.logger.info(f"=> Loaded checkpoint from {self.configs.resume}")
            ckpt = torch.load(self.configs.resume, map_location=f"cuda:{self.rank}")
            util_net.reload_model(self.model, ckpt['state_dict'])
            torch.cuda.empty_cache()
            self.iters_start = ckpt['iters_start']
            for ii in range(1, self.iters_start+1):
                self.adjust_lr(ii)
            if self.rank == 0 and hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / f"ema_model_{self.iters_start:06d}.pth"
                if ema_ckpt_path.exists():
                    self.logger.info(f"=> Loaded EMA checkpoint from {str(ema_ckpt_path)}")
                    ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                    for key in self.ema_state.keys():
                        if key in ema_ckpt:
                            self.ema_state[key] = deepcopy(ema_ckpt[key].detach().data)
            torch.cuda.empty_cache()
            if self.amp_scaler is not None:
                if "amp_scaler" in ckpt:
                    self.amp_scaler.load_state_dict(ckpt["amp_scaler"])
                    if self.rank == 0:
                        self.logger.info("Loading scaler from resumed state...")
            self.setup_seed(seed=self.iters_start)
        else:
            self.iters_start = 0

    @torch.no_grad()
    def update_ema_model(self):
        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            source_state = self.model.state_dict()
            rate = self.ema_rate
            for key, value in self.ema_state.items():
                if key in self.ema_ignore_keys:
                    self.ema_state[key] = source_state[key]
                else:
                    self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1-rate)

    def reload_ema_model(self):
        if self.rank == 0:
            if self.num_gpus > 1:
                model_state = {key[7:]:value for key, value in self.ema_state.items()}
            else:
                model_state = self.ema_state
            self.ema_model.load_state_dict(model_state)

    def adjust_lr(self, current_iters=None):
        base_lr = self.configs.train.lr
        warmup_steps = self.configs.train.warmup_iterations
        current_iters = self.current_iters if current_iters is None else current_iters
        if current_iters <= warmup_steps:
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = (current_iters / warmup_steps) * base_lr
        else:
            if hasattr(self, 'lr_scheduler'):
                self.lr_scheduler.step()

    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        loss_coef = self.configs.train.get('loss_coef')
        from functools import partial
        context = (partial(torch.amp.autocast, device_type='cuda') 
                if self.configs.train.use_amp else nullcontext)
        with context():
            losses, z_t, z0_pred = dif_loss_wrapper()
            x0_pred = self.base_diffusion.decode_first_stage(z0_pred, self.autoencoder)
            self.current_x0_pred = x0_pred.detach()

            losses["lpips"] = self.lpips_loss(x0_pred, micro_data['gt']).to(z0_pred.dtype).view(-1)
            flag_nan = torch.any(torch.isnan(losses["lpips"]))
            if flag_nan:
                losses["lpips"] = torch.nan_to_num(losses["lpips"], nan=0.0)
            losses["lpips"] *= loss_coef[1]

            if loss_coef[0] > 0:
                losses["mse"] *= loss_coef[0]
            else:
                assert loss_coef[2] > 0
                losses["mse"] = mean_flat((x0_pred - micro_data['gt']) ** 2)
                losses["mse"] *= loss_coef[2]

            assert losses["mse"].shape == losses["lpips"].shape
            if flag_nan:
                losses["loss"] = losses["mse"]
            else:
                losses["loss"] = losses["mse"] + losses["lpips"]
            loss = losses['loss'].mean() / num_grad_accumulate
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()
        return losses, z0_pred, z_t

    def training_step(self, data):
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)
        data2 = data['img_name']
        del data['img_name']
        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)
            tt = torch.randint(
                0, self.base_diffusion.num_timesteps,
                size=(micro_data['gt'].shape[0],),
                device=f"cuda:{self.rank}",
            )
            latent_downsamping_sf = 2**(len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
            latent_resolution = micro_data['gt'].shape[-1] // latent_downsamping_sf
            if 'autoencoder' in self.configs:
                noise_chn = self.configs.autoencoder.params.embed_dim
            else:
                noise_chn = micro_data['gt'].shape[1]
            noise = torch.randn(
                size= (micro_data['gt'].shape[0], noise_chn,) + (latent_resolution, ) * 2,
                device=micro_data['gt'].device,
            )
            if self.configs.model.params.cond_x0:
                model_kwargs = {'x0':micro_data['x0'],}
                if 'mask' in micro_data:
                    model_kwargs['mask'] = micro_data['mask']
            else:
                model_kwargs = None
            
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['gt'],
                micro_data['x0'],
                tt,
                first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs,
                noise=noise,
            )
            if last_batch or self.num_gpus <= 1:
                losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)
            else:
                with self.model.no_sync():
                    losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)
            if last_batch:
                self.log_step_train(losses, tt, micro_data, z_t, z0_pred.detach())
        data['img_name'] = data2
        if self.configs.train.use_amp:
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            self.optimizer.step()
        self.model.zero_grad()
        if hasattr(self.configs.train, 'ema_rate'):
            self.update_ema_model()

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, phase='train'):
        if self.rank == 0:
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    assert value.shape == mask.shape
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()
            if self.current_iters % self.configs.train.log_freq[0] == 0:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                log_str = f'Train: {self.current_iters:06d}/{self.configs.train.iterations:06d}, MSE/LPIPS: '
                for jj, current_record in enumerate(record_steps):
                    log_str += f't({current_record:d}):{self.loss_mean["mse"][jj].item():.1e}/{self.loss_mean["lpips"][jj].item():.1e}, '
                log_str += f'lr:{self.optimizer.param_groups[0]["lr"]:.2e}'
                self.logger.info(log_str)
                
                if self.use_swanlab:
                    try:
                        swanlab_metrics = {
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                            "iteration": self.current_iters
                        }
                        for jj, current_record in enumerate(record_steps):
                            swanlab_metrics[f"train/MSE_t{current_record}"] = self.loss_mean["mse"][jj].item()
                            swanlab_metrics[f"train/LPIPS_t{current_record}"] = self.loss_mean["lpips"][jj].item()
                            swanlab_metrics[f"train/Loss_t{current_record}"] = self.loss_mean["loss"][jj].item()
                        
                        swanlab.log(swanlab_metrics)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to upload training metrics to SwanLab: {e}")

    def validation(self, phase='val'):
        if self.rank == 0:
            if self.configs.train.use_ema_val:
                self.reload_ema_model()
                self.ema_model.eval()
            else:
                self.model.eval()

            indices = np.linspace(
                0,
                self.base_diffusion.num_timesteps,
                self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 4,
                endpoint=False,
                dtype=np.int64,
            ).tolist()
            if not (self.base_diffusion.num_timesteps-1) in indices:
                indices.append(self.base_diffusion.num_timesteps-1)
            
            batch_size = self.configs.train.batch[1]
            # num_iters_epoch = math.ceil(len(self.datasets[phase]) / batch_size)
            mean_psnr = mean_lpips = 0
            
            for ii, data in enumerate(self.dataloaders[phase]):
                data = self.prepare_data(data, phase='val')
                if 'gt' in data:
                    im_x0, im_gt = data['x0'], data['gt']
                else:
                    im_x0 = data['x0']
                
                num_iters = 0
                if self.configs.model.params.cond_x0:
                    model_kwargs = {'x0':data['x0'],}
                    if 'mask' in data:
                        model_kwargs['mask'] = data['mask']
                else:
                    model_kwargs = None
                
                tt = torch.tensor(
                    [self.base_diffusion.num_timesteps, ]*im_x0.shape[0],
                    dtype=torch.int64,
                ).cuda()
                
                for sample in self.base_diffusion.p_sample_loop_progressive(
                    y=im_x0,
                    model=self.ema_model if self.configs.train.use_ema_val else self.model,
                    first_stage_model=self.autoencoder,
                    noise=None,
                    clip_denoised=True if self.autoencoder is None else False,
                    model_kwargs=model_kwargs,
                    device=f"cuda:{self.rank}",
                    progress=False,
                ):
                    sample_decode = {}
                    if num_iters in indices:
                        for key, value in sample.items():
                            if key in ['sample', ]:
                                sample_decode[key] = self.base_diffusion.decode_first_stage(
                                    value,
                                    self.autoencoder,
                                ).clamp(-1.0, 1.0)
                        im_sr_progress = sample_decode['sample']
                        if num_iters + 1 == 1:
                            im_sr_all = im_sr_progress
                        else:
                            im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                    num_iters += 1
                    tt -= 1

                if 'gt' in data:
                    mean_psnr += util_image.batch_PSNR(
                        sample_decode['sample'] * 0.5 + 0.5,
                        im_gt * 0.5 + 0.5,
                        ycbcr=self.configs.train.val_y_channel,
                    )
                    mean_lpips += self.lpips_loss(
                        sample_decode['sample'],
                        im_gt,
                    ).sum().item()
                if (ii + 1) % self.configs.train.log_freq[2] == 0:
                    if 'gt' in data:
                        save_validation_images(
                            im_x0[0], im_gt[0], sample_decode['sample'][0],
                            self.val_images_dir, f"val_sample_{ii+1}", self.current_iters
                        )
            if 'gt' in data:
                mean_psnr /= len(self.datasets[phase])
                mean_lpips /= len(self.datasets[phase])
                self.logger.info(f'Validation Metric: PSNR={mean_psnr:5.2f}, LPIPS={mean_lpips:6.4f}')
                is_best = mean_lpips < self.best_lpips
                if is_best:
                    previous_best = self.best_lpips
                    self.best_lpips = mean_lpips
                    self.logger.info(f'🎉 New best LPIPS: {mean_lpips:6.4f} (Previous best: {previous_best:6.4f})')
                    self.save_ckpt(is_best=True)
                
                if self.use_swanlab:
                    try:
                        swanlab.log({
                            "val/PSNR": mean_psnr,
                            "val/LPIPS": mean_lpips,
                            "iteration": self.current_iters
                        })
                        val_images = upload_random_validation_images(self.val_images_dir, num_images=10)
                        if val_images:
                            swanlab.log({
                                "val/images": val_images,
                                "iteration": self.current_iters
                            })
                            self.logger.info(f"📊 Uploaded {len(val_images)} validation images to SwanLab")
                        self.logger.info("📊 Validation metrics uploaded to SwanLab")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to upload to SwanLab: {e}")
            self.logger.info("=" * 80)
            if not (self.configs.train.use_ema_val and hasattr(self.configs.train, 'ema_rate')):
                self.model.train()

    def train(self):
        console.print(f"Dataset: [green]{self.dataset_name}[/green]")
        console.print(f"Total iterations: [green]{self.configs.train.iterations}[/green]")
        console.print(f"Checkpoint directory: [green]{self.ckpt_dir}[/green]")
        console.print(f"Validation images directory: [green]{self.val_images_dir}[/green]")
        self.model.train()
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=1
        ) as progress:
            task = progress.add_task(
                f"[cyan]Training {self.dataset_name}",
                total=self.configs.train.iterations - self.iters_start
            )
            for ii in range(self.iters_start, self.configs.train.iterations):
                self.current_iters = ii + 1
                data = self.prepare_data(next(self.dataloaders['train']))
                self.training_step(data)
                if 'val' in self.dataloaders and (ii+1) % self.configs.train.get('val_freq', 10000) == 0:
                    self.validation()
                self.adjust_lr()
                if (ii+1) % self.configs.train.save_freq == 0:
                    self.save_ckpt()
                if (ii+1) % num_iters_epoch == 0 and self.sampler is not None:
                    self.sampler.set_epoch(ii+1)
                progress.update(task, advance=1)
                if ii % self.configs.train.log_freq[0] == 0 and hasattr(self, 'loss_mean'):
                    desc = f"[cyan]Training {self.dataset_name} - iter {ii+1}/{self.configs.train.iterations}"
                    progress.update(task, description=desc)
        console.print("[bold green]Training completed![/bold green]")

def update_config_with_args(configs, args):
    for key in vars(args):
        if key in ['cfg_path', 'save_dir', 'resume']:
            configs[key] = getattr(args, key)
    if args.data_root:
        configs.data.train.params.dir_path = f"{args.data_root}/train/A"
        configs.data.train.params.dir_path_extra = f"{args.data_root}/train/B"
        configs.data.val.params.dir_path = f"{args.data_root}/val/A"
        configs.data.val.params.dir_path_extra = f"{args.data_root}/val/B"
    if args.batch_size:
        configs.train.batch[1] = args.batch_size
    if args.learning_rate:
        configs.train.lr = args.learning_rate
    if args.iterations:
        configs.train.iterations = args.iterations
    return configs

def setup_environment(args):
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        console.print(f"GPUs: [green]{args.gpu_ids}[/green]")

def main():
    console.print("[bold blue]🚀 Rap-Diffusion Training 🚀[/bold blue]")
    args = get_parser()
    setup_environment(args)
    console.print(f"Config file: [green]{args.cfg_path}[/green]")
    configs = OmegaConf.load(args.cfg_path)
    configs = update_config_with_args(configs, args)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config_save_path = save_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        OmegaConf.save(configs, f)
    trainer = Trainer(configs)
    trainer.train()

if __name__ == "__main__":
    main()