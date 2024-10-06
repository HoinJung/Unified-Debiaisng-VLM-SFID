#!/bin/bash\
mkdir embedding
mkdir embedding/flickr
mkdir checkpoint


# Extract debiasing embedding for zero-shot classification and text-to-image retrieval
python preprocessing/clip_extract_embedding.py
python preprocessing/facet_extract_embedding.py
python preprocessing/flickr_extract_embedding.py

# Extract debiasing embedding for image captioning
python preprocessing/clipcap_extract_embedding.py
python preprocessing/blip_extract_embedding.py


# Extract debiasing embedding for text-to-image generation
python preprocessing/codi_extract_embedding.py
python preprocessing/sd_extract_embedding.py




