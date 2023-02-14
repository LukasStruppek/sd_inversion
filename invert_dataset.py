from inversion.null_text_inversion import NullInversion
from inversion.inference import run_and_display
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import torchvision
from rtpt import RTPT
from PIL import Image
import os
from tqdm import tqdm


GUIDANCE_SCALE = 7.5
NUM_DDIM_STEPS = 50
PROMPT = 'a photo of a person'
IMAGE_FOLDER = 'data/ffhq'
MODEL_PATH = 'runwayml/stable-diffusion-v1-5'
OUTPUT_FOLDER = 'data/ffhq_latents'

def main():

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device(
        'cpu')

    image_paths = sorted(os.listdir(IMAGE_FOLDER))
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    rtpt = RTPT('LS', 'Null-Text Inversion', len(image_paths))
    rtpt.start()

    ldm_model = load_ldm_model(device)

    null_inversion = NullInversion(ldm_model,
                                num_ddim_steps=NUM_DDIM_STEPS,
                                guidance_scale=GUIDANCE_SCALE)

    for input_file in tqdm(image_paths):
        _, latent, uncond_embeddings = null_inversion.invert(os.path.join(IMAGE_FOLDER, input_file),
                                                            PROMPT,
                                                            verbose=False)
        output_file = input_file.split('.')[0] + '.pt'
        torch.save(
            {
                'prompt': PROMPT,
                'guidance_scale': GUIDANCE_SCALE,
                'num_inference_steps': NUM_DDIM_STEPS,
                'latents': latent.cpu(),
                'uncond_embeddings': uncond_embeddings.cpu()
            }, os.path.join(OUTPUT_FOLDER, output_file))
        rtpt.step()

def load_ldm_model(device):
    scheduler = DDIMScheduler(beta_start=0.00085,
                            beta_end=0.012,
                            beta_schedule="scaled_linear",
                            clip_sample=False,
                            set_alpha_to_one=False)
    ldm_model = StableDiffusionPipeline.from_pretrained(
        MODEL_PATH,
        cache_dir='./weights',
        use_auth_token="",
        scheduler=scheduler).to(device)

    return ldm_model


if __name__ == '__main__':
    main()
