import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import yaml
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import nibabel as nib
from PIL import Image

def make_batch_nii(nii_file, device):
    img = nib.load(nii_file)
    img = np.array(img.get_fdata()) 
    img = img / 4096 * 2 -1 # make the scale to [-1, 1] # specifically for the M0 case    
    img = np.transpose(img, (2,0,1)) # shape of [Nx Ny Nslice] -> [Nslice Nx Ny]
    img = np.expand_dims(img,axis=1) # shape of [Nslice 1 Nx Ny]
    img = np.concatenate((img,img,img),axis=1) # shape of [Nslice 3 Nx Ny]
    img = torch.from_numpy(img).to(device = device, dtype=torch.float)

    return img

def make_gif_for_intermediates(intermediates):
    gif_list = []
    for i in range(len(intermediates)):
        tmp = intermediates[i].cpu().numpy()
        slice,_,_,_ = tmp.shape
        center_slice = np.mean(tmp[int(slice/2),:,:,:], axis=0)
        center_slice = Image.fromarray(np.uint8(center_slice*255))
        gif_list.append(center_slice)
    
    return gif_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing the control image to be used to generate the M0 image",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--model_name",
        type = str,
        nargs = "?",
        help = "to load the trained model",
    )
    parser.add_argument(
        "--GPU",
        type = int,
        default = 0,
        help = "specify the number of GPU to use",
    )

    parser.add_argument(
        "--num_averages",
        type = int,
        default = 1,
        help = "number of generated samples for averaging"
    )

    opt = parser.parse_args()

    # data path
    controls = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(opt.indir,"*")))]
    print(f"Found {len(controls)} inputs.")
    
    # model path and config
    model_base = 'trained_models'
    model_name = opt.model_name
    time_start = model_name[0:19] # 'YYYY-MM-DDTHH-MM-SS'
    config = OmegaConf.load(os.path.join(model_base, model_name, "configs", time_start + "-project.yaml"))    
    model = instantiate_from_config(config.model)

    model_checkpoint_path = os.path.join(model_base, model_name, "checkpoints", "last.ckpt")
    model.load_state_dict(torch.load(model_checkpoint_path)["state_dict"],
                          strict=False)

    GPU_num = opt.GPU
    device = torch.device("cuda:"+str(GPU_num)) if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    num_avg = opt.num_averages
    
    output_dir = os.path.join(opt.outdir,model_name) 
    os.makedirs(output_dir, exist_ok=True)
    
    # start sampling
    with torch.no_grad():
        with model.ema_scope():
            for sub in tqdm(controls):
                avg_sample = 0
                for i in tqdm(range(num_avg)):
                    outpath = os.path.join(output_dir, sub)
                    os.makedirs(outpath, exist_ok=True)
                    control_nii = os.path.join(opt.indir, sub, 'PASL_processed', 'normGW_combined_mean_control.nii')
                    batch = make_batch_nii(control_nii, device=device)
                    c = model.cond_stage_model.encode(batch)  # since currently only 1 input condition

                    shape = (c.shape[1:])
                    samples_ddim, intermediates = sampler.sample(S=opt.steps,
                                                    conditioning=c,
                                                    batch_size=c.shape[0],
                                                    shape=shape,
                                                    log_every_t=1,
                                                    verbose=False)
                    
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    avg_sample += x_samples_ddim

                avg_sample /= num_avg

                # rescale the image, the orginal scale is [0, 4096], hard code, need to be consistent with exact images
                sampled_M0_img = torch.clamp((avg_sample + 1.0) / 2.0, min=0.0, max=1.0) * 4096 
                sampled_M0_img = np.mean(sampled_M0_img.cpu().numpy(), axis=1)
                sampled_M0_img = np.transpose(sampled_M0_img, (1, 2, 0))
                sampled_M0_nii = nib.Nifti1Image(sampled_M0_img, affine=nib.load(control_nii).affine)

                nib.save(sampled_M0_nii, os.path.join(outpath, 'generated_M0_avg' + str(num_avg) + '.nii'))
                nib.save(nib.load(control_nii), os.path.join(outpath, 'control.nii'))

    hyperparam_dict = {
        "model path": model_base + '/' + model_name,
        "diffusion steps": opt.steps,
        "num of samples": num_avg,
    }

    # save hyperparameters
    with open(os.path.join(output_dir, "sampling_configs.yaml"), "w") as file:
        yaml.dump(hyperparam_dict, file)        
