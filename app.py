import torch
import base64

from sanic.response import json
from torch import nn
from transformers import CLIPProcessor, CLIPVisionModel
from PIL import Image
from path import Path
from io import BytesIO

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, embedDIM):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(embedDIM, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        yhat = self.linear_relu_stack(x)
        return yhat


class ClipEmbed(nn.Module):
    def __init__(self):
        super(ClipEmbed, self).__init__()
        self.CLIP = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")

    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt")
        outputs = self.CLIP(**inputs)
        pooled_output_list = outputs.pooler_output[0].tolist()
        return pooled_output_list


# Define model
class NeuralEmbed(nn.Module):
    def __init__(self, neuralnet, clipnet):
        super(NeuralEmbed, self).__init__()
        assert isinstance(neuralnet, NeuralNetwork)
        assert isinstance(clipnet, ClipEmbed)
        assert neuralnet.linear_relu_stack[0].in_features == clipnet.CLIP.config.hidden_size

        self.neuralnet = neuralnet
        self.clipnet = clipnet

    def forward(self, x):
        yhat = self.neuralnet(torch.tensor(self.clipnet(x)))
        return yhat

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    device = 0 if torch.cuda.is_available() else -1
    # load model
    embedDIM = 768
    neuralmodel = NeuralNetwork(embedDIM)
    assert Path('model.pt').exists() , "couldn't find model.pt"
    torch_load = torch.load('model.pt')
    neuralmodel.load_state_dict(torch_load)
    clipmodel = ClipEmbed()
    model = NeuralEmbed(neuralmodel, clipmodel) # .to(device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    image_byte_string = model_inputs.get('imageByteString', None)
    if image_byte_string is None:
        return {'error': 'imageByteString is required'}

    # convert image byte string to bytes
    image_encoded = image_byte_string.encode('utf-8')
    image_bytes = BytesIO(base64.b64decode(image_encoded))

    # Load image
    image = Image.open(image_bytes)
    
    # Run the model
    result = model(image)

    # Return the results as a dictionary
    return result.tolist()[0]
