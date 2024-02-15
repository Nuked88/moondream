import argparse
from PIL import Image
from moondream.moondream import Moondream
from huggingface_hub import snapshot_download
import torch

from huggingface_hub import hf_hub_download

import os
from threading import Thread
from transformers import  CodeGenTokenizerFast as Tokenizer
import numpy as np
def detect_device():
    """
    Detects the appropriate device to run on, and return the device and dtype.
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    else:
        return torch.device("cpu"), torch.float32



# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)




# get_torch_device()
moondream = None
tokenizer = None


models_base_path = r"K:\VARIE\ComfyUI_windows_portable\ComfyUI\models\GPTcheckpoints\moondream"




def main2(image_path, prompt):
    # get_torch_device()
    moondream = None
    tokenizer = None

    image = Image.open(image_path)
    #image_embeds = vision_encoder(image)
    dtype = torch.float32
    selected_device='cuda'
    if selected_device=='cuda':
        device = torch.device("cuda")
        
    elif selected_device!=device.type:
        device=torch.device("cpu")
 
    if moondream ==None:

        config_json=os.path.join(models_base_path,'config.json')
        if os.path.exists(config_json)==False:
            hf_hub_download("vikhyatk/moondream1",
                                        local_dir=models_base_path,
                                        filename="config.json",
                                        endpoint='https://hf-mirror.com')
        
        model_safetensors=os.path.join(models_base_path,'model.safetensors')
        if os.path.exists(model_safetensors)==False:
            hf_hub_download("vikhyatk/moondream1",
                                       local_dir=models_base_path,
                                       filename="model.safetensors",
                                       endpoint='https://hf-mirror.com')
        
        tokenizer_json=os.path.join(models_base_path,'tokenizer.json')
        if os.path.exists(tokenizer_json)==False:
            hf_hub_download("vikhyatk/moondream1",
                                       local_dir=models_base_path,
                                       filename="tokenizer.json",
                                       endpoint='https://hf-mirror.com')
        
        tokenizer = Tokenizer.from_pretrained(models_base_path)
        moondream = Moondream.from_pretrained(models_base_path).to(device=device, dtype=dtype)
        moondream.eval()

     



    im=image
    im=pil2tensor(im)
    im=tensor2pil(im)

    image_embeds = moondream.encode_image(im)
    
    # streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    print(tokenizer)
    print(moondream)
    res=moondream.answer_question(image_embeds, prompt,tokenizer)

    print(res)












if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=False)
    args = parser.parse_args()
    main2(args.image, args.prompt)