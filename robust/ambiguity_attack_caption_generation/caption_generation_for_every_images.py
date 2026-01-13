from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import torch
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk

# step-1:获得caption
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

image_dir = "./wm_corgi/"
captions = []

for img_file in sorted(os.listdir(image_dir)):
    if not img_file.endswith((".jpg", ".png")): continue
    image = Image.open(os.path.join(image_dir, img_file)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    captions.append(caption)


print(captions)
#import pdb;pdb.set_trace()


# step-2:聚类
nltk.download('punkt_tab')

all_phrases = []
for cap in captions:
    tokens = word_tokenize(cap.lower())
    all_phrases.extend([" ".join(gram) for gram in ngrams(tokens, 2)])  # bigrams

top_phrases = Counter(all_phrases).most_common(10)
print("Top shared phrases:", top_phrases)
