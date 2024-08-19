# Segmentation Substitution Framework

Based on the previous work [see here](https://github.com/geekyutao/Inpaint-Anything), except this uses prompt based SAM model (from [grounding-dino](https://github.com/IDEA-Research/GroundingDINO/tree/main)) instead of coordinate based model.

### TODO : 
* generate adversarial examples for webqa, nlvr2, vqav2
* integrate adversarial example with VoLTA data loaders


### Setup
Requires `python>=3.8`
```bash
python -m pip install torch torchvision torchaudio
python -m pip install -e segment_anything
python -m pip install diffusers transformers accelerate scipy safetensors
python -m pip install -r lama/requirements.txt 
```
Download the model checkpoints provided in [LaMa](./lama/README.md) and [big-lama](https://disk.yandex.ru/d/ouP6l8VJ0HpMZg)), and put them into `./pretrained_models`. For simplicity, you can also go [here](https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing), directly download [pretrained_models](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A?usp=sharing), put the directory into `./` and get `./pretrained_models`.

### Object removal

```bash
python remove_anything.py \
    --input_img ./example/remove-anything/dog.jpg \
    --prompt "dog"
    --dilate_kernel_size 15 \
    --output_dir ./results
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama
```

### Object infilling

```bash
python fill_anything.py \
    --input_img ./example/fill-anything/sample1.png \
    --seg_prompt "dog"
    --fill_prompt "a teddy bear on a bench" \
    --dilate_kernel_size 50 \
    --output_dir ./results \
```
