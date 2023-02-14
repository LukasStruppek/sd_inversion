import os

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from rtpt import RTPT
from tqdm import tqdm

from inversion.inference import run_and_display
from inversion.prompt_to_prompt import make_controller, AttentionStore

MODEL_PATH = 'runwayml/stable-diffusion-v1-5'
DATA_FOLDER = 'data/ffhq_latents'
OUTPUT_FOLDER = 'data/synthetic'
CROSS_REPLACE_STEPS = {'default_': 1.0}
BLEND_WORD = None
EQ_PARAMS = None

ATTRIBUTES = {
    'age': ('young', 'old'),
    'age_steps':
    ('10 to 19 years old', '20 to 29 years old', '30 to 39 years old',
     '40 to 49 years old', '50 to 59 years old', '60 to 69 years old',
     'older than 70 years'),
    'beard': ('goatee beard', 'mustache', 'no beard'),
    'eyebrows': ('arched eyebrows', 'bushy eyebrows'),
    'eyeglasses': ('eyeglasses', 'no eyeglasses'),
    'eyes': ('narrow eyes', 'bags under eyes'),
    'standard': (None, ),
    'gender': ('male, man', 'female, woman'),
    'hair_color':
    ('black hair', 'blond hair', 'brown hair', 'gray hair', 'bald'),
    'hair_style': ('bangs hair, fringe hair', 'straight hair', 'sideburns',
                   'receding hairline', 'wavy hair'),
    'lips': ('big lips', ),
    'makeup': ('heavy makeup', 'no makeup'),
    'nose': ('big nose', 'pointy nose'),
    'race': ('white race', 'black race', 'asian race', 'indian race',
             'latino hispanic race', 'east race', 'southeast asian race',
             'middle eastern race'),
    'shape': ('oval face', 'high cheekbones', 'double chin', 'no double chin',
              'chubby', 'not chubby'),
    'skin': ('pale skin', 'rosy cheeks')
}

SELF_REPLACE_STEPS = {
    'age': 0.4,
    'age_steps': 0.4,
    'beard': 0.4,
    'eyebrows': 0.3,
    'eyeglasses': 0.4,
    'eyes': 0.4,
    'gender': 0.4,
    'hair_color': 0.6,
    'hair_style': 0.4,
    'lips': 0.3,
    'makeup': 0.4,
    'nose': 0.3,
    'race': 0.4,
    'shape': 0.3,
    'skin': 0.5,
    'standard': 1.0,
}


def main():
    files = sorted(os.listdir(DATA_FOLDER))[:10]

    rtpt = RTPT('LS', 'Generate Attributes',
                len(files) * len(ATTRIBUTES.keys()))
    rtpt.start()

    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    ldm_model = load_ldm_model(device)

    for attribute_class in tqdm(sorted(ATTRIBUTES.keys())):
        attributes = sorted(ATTRIBUTES[attribute_class])

        # create output folder
        create_output_folder(OUTPUT_FOLDER, attribute_class, attributes)

        for file in files:
            # load latent vectors, embeddings and other information
            data = torch.load(os.path.join(DATA_FOLDER, file))

            # create prompts
            standard_prompt = ''.join(data['prompt'])
            prompts = create_prompts(standard_prompt, attributes)
            prompts = [standard_prompt] + prompts

            # define controller parameters
            if attribute_class == 'standard':
                controller = AttentionStore()
            else:
                controller = make_controller(
                    prompts, False, CROSS_REPLACE_STEPS,
                    SELF_REPLACE_STEPS[attribute_class], ldm_model.tokenizer,
                    device, BLEND_WORD, EQ_PARAMS)

            images, _ = run_and_display(
                prompts,
                ldm_model,
                data['num_inference_steps'],
                data['guidance_scale'],
                controller,
                run_baseline=False,
                latent=data['latents'],
                uncond_embeddings=data['uncond_embeddings'])

            # save images
            for img, attribute in zip(images[1:], attributes):
                img = Image.fromarray(img)
                file_name = file.split('.')[0] + '.jpg'
                if attribute is None:
                    attribute = 'standard'
                img.save(
                    os.path.join(OUTPUT_FOLDER, attribute_class,
                                 attribute.replace(' ', '_'), file_name))

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


def create_output_folder(output_folder, attribute_class, attributes):
    for attribute in attributes:
        if attribute is None:
            attribute = 'standard'
        attribute = attribute.replace(' ', '_')
        attribute_class = attribute_class.replace(' ', '_')
        output_folder = os.path.join(OUTPUT_FOLDER, attribute_class, attribute)
        os.makedirs(output_folder, exist_ok=True)


def create_prompts(standard_prompt, attributes):
    prompts = []
    for attribute in attributes:
        if attribute is not None:
            prompts.append(standard_prompt + ', ' + attribute)
        else:
            prompts.append(standard_prompt)
    return prompts


if __name__ == '__main__':
    main()
