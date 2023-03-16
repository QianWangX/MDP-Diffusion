import torch
from torchvision import transforms as tfms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def randn_tensor(
    shape,
    generator,
    device=None,
    dtype=None,
    layout=None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def pil_to_latent(vae, input_im, device, generator=None):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(device)*2-1) # Note scaling
    latent = 0.18215 * latent.latent_dist.sample(generator)
    
    return latent

def latents_to_pil(vae, latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def latents_to_img(vae, latents, normalize='0-1', size=512):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    # with torch.no_grad():
    image = vae.decode(latents).sample
    if normalize == '0-1':
        image = (image / 2 + 0.5)
    # resize tensor to 224x224 
    image = torch.nn.functional.interpolate(image, size=(size, size), mode='bilinear', align_corners=False)
    return image

def img_to_latents(vae, input_im, generator=None):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    latent = vae.encode(input_im)
    latent = 0.18215 * latent.latent_dist.sample(generator)
    
    return latent


def get_timesteps(scheduler, num_inference_steps, strength):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start


def prepare_latents(vae, scheduler, batch_size, num_channels_latents, height, width, device, generator=None):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    latents = randn_tensor(shape, generator=generator, device=device)
    latents = latents * scheduler.init_noise_sigma
    
    return latents

def encode_text(prompt, tokenizer, text_encoder, device):
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    text_embeddings = text_embeddings.clone()
    
    return text_embeddings

def visualize_latents(latents, title='Image'):
    model_output_img = latents_to_pil(latents)
    plt.imshow(model_output_img[0])
    plt.title(title)
    plt.show()
    
# define a function that visualize a list of pil images
def visualize_images(images, titles=None, cols=5, figsize=(3, 3)):   
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if n_images == 1:
        images[0].show()
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure(figsize=figsize)
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(int(np.ceil(n_images/float(cols))), int(cols), n + 1)
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    




