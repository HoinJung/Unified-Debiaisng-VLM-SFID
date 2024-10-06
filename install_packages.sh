#!/bin/bash\

pip install transformers==4.42.3 
# pip install transformers==4.33.2
pip install diffusers==0.29.2
pip install datasets==2.20.0

pip install six==1.16.0
pip install easydict==1.9
pip install ftfy==6.1.1
pip install einops==0.6.1
pip install decord==0.6.0
pip install pytorch-lightning==1.9.0
pip install timm==0.4.5
pip install torchlibrosa==0.0.9
pip install librosa==0.9.2
pip install tqdm
pip install salesforce-lavis
pip install git+https://github.com/openai/CLIP.git
pip install torchaudio
pip install nltk

# # Download pre-trained weight for image captioning (CLIP-CAP and BLIP)
pip install gdown
gdown  1ocKr2gWCx20QRykqxCddhTwiXaEnRLS_
mv clip_cap_coco_weight.pt external/clipcap/
curl -o external/BLIP/model_base_capfilt_large.pth https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth


# Download pre-trained weight for CoDi
wget https://huggingface.co/ZinengTang/CoDi/resolve/main/checkpoints_fp16/CoDi_encoders.pth -P checkpoints/
wget https://huggingface.co/ZinengTang/CoDi/resolve/main/checkpoints_fp16/CoDi_text_diffuser.pth -P checkpoints/
wget https://huggingface.co/ZinengTang/CoDi/resolve/main/checkpoints_fp16/CoDi_audio_diffuser_m.pth -P checkpoints/
wget https://huggingface.co/ZinengTang/CoDi/resolve/main/checkpoints_fp16/CoDi_video_diffuser_8frames.pth -P checkpoints/


