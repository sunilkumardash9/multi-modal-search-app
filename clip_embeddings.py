import clip
import torch

from numpy import ndarray
from typing import List
from PIL import Image


class ClipEmbeddingsfunction:

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):   
        self.device = device
        self.model, self.preprocess = clip.load(model_name, self.device)

    def __call__(self, docs: List[str])->List[ndarray]:
        list_of_embeddings = []
        for image_path in docs:
            image = Image.open(image_path)
            image = image.resize((224, 224))
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embeddings = self.model.encode_image(image_input).cpu().detach().numpy()
            list_of_embeddings.append(list(embeddings[0]))
        return list_of_embeddings
    
    def get_text_embeddings(self, text: str) -> List[ndarray]:
        text_token = clip.tokenize(text)
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_token).cpu().detach().numpy()
        return list(text_embeddings[0])