import pandas as pd
import streamlit as st
import torch
from inference import beam_search
from torchvision.transforms import v2
from PIL import Image
from vocab import Vocabulary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = './flickr30k_processed/images'
checkpoint_5 = torch.load('./image_captioning_best_5.pth', map_location=device)
encoder_5 = checkpoint_5['encoder'].to(device)
decoder_5 = checkpoint_5['decoder'].to(device)
vocab = Vocabulary(vocab_file='./vocab10000.txt', vocab_size=10000)
vocab.load_vocab()
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>Deep Learning - Image Captioning</h1>",
            unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white;'>Create image captions at any time! &#128444 &#128394</h2>",
            unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose the picture you want to caption", key="uploader")

# keeping track of the uploaded images
if 'uploaded_images' not in st.session_state:
    st.session_state['uploaded_images'] = []

if uploaded_file:
    image_name = uploaded_file.name
    image_path = './flickr30k_processed/images/' + image_name
    st.session_state.uploaded_images.append(image_path)
    print(st.session_state.uploaded_images)

for i in range(0, len(st.session_state.uploaded_images), 3):
    row_container = st.columns(3)
    for j in range(3):
        if i + j < len(st.session_state.uploaded_images):
            col = row_container[j]
            image_path = st.session_state.uploaded_images[i + j]
            image = Image.open(image_path)
            image = image.resize((600, 400))
            col.image(image, caption=beam_search(encoder_5, decoder_5, image_path, vocab=vocab, vocab_size=10000,
                                                 device=device), width=None, use_column_width=None, clamp=False,
                      channels="RGB", output_format="auto")

    st.write("\n")
