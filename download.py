# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
from transformers import CLIPProcessor, CLIPVisionModel
import requests

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    # pipeline('fill-mask', model='bert-base-uncased')
    CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model_pt_url = "https://firebasestorage.googleapis.com/v0/b/authentication-374722.appspot.com/o/lindsay1807%40yahoo.ca.pt?alt=media&token=1916c226-dafc-421e-817d-3be7c9303c1a"
    r = requests.get(model_pt_url)
    with open("model.pt", "wb") as f:
        f.write(r.content)

if __name__ == "__main__":
    download_model()
