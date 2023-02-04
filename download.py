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
    model_pt_url = "https://firebasestorage.googleapis.com/v0/b/authentication-374722.appspot.com/o/timtjc08%40gmail.com.pt?alt=media&token=bbf23944-eca4-4ffb-9ccb-82e5f6067586"
    r = requests.get(model_pt_url)
    with open("model.pt", "wb") as f:
        f.write(r.content)

if __name__ == "__main__":
    download_model()
