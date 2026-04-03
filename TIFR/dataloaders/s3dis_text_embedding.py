import torch
import clip
import numpy as np

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP ViT-B/16 
model, preprocess = clip.load("ViT-B/16", device=device)

# category list
categories = [
    'unannotated',
    'ceiling', 
    'floor', 
    'wall', 
    'beam', 
    'column', 
    'window', 
    'door', 
    'table', 
    'chair', 
    'sofa', 
    'bookcase', 
    'board',

    'computer', 'keyboard', 'mouse', 'phone', 'trash bin',
    'printer', 'cabinet', 'plant', 'lamp', 'projector',
    'screen', 'cushion'
]

# text prompts
prompts = [f"Point cloud of a {cls}" for cls in categories]

# text embedding
with torch.no_grad():
    text_inputs = clip.tokenize(prompts).to(device)
    text_embeddings = model.encode_text(text_inputs)
    text_embeddings = text_embeddings.float()
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

# save
text_embeddings_np = text_embeddings.cpu().numpy()
np.save('s3dis_embeddings.npy', text_embeddings_np)

print(f"Saved embeddings shape: {text_embeddings_np.shape}") 
print(f"Embedding dtype: {text_embeddings_np.dtype}") 
print("文本嵌入生成完成！")
