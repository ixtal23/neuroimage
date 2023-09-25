# NeuroImage

This is an image generation tool that implements the generating of images by [Diffusers](https://github.com/huggingface/diffusers) and [Kandinsky 2.2](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky_v22) solutions.

## Usage

```
git clone git@github.com:ixtal23/neuroimage.git

cd neuroimage

pip install -r requirements.txt

python main.py --output-file OUTPUT_FILE
               --prompt PROMPT
               [--negative-prompt NEGATIVE_PROMPT]
               [--seed SEED]
               [--guidance-scale GUIDANCE_SCALE]
               [--inference-steps [1-100]]
               [--height [16-1024]]
               [--width [16-1024]]
               [-h]
```

### Options

```
--output-file OUTPUT_FILE           a path to an output file
--prompt PROMPT                     prompt
--negative-prompt NEGATIVE_PROMPT   negative prompt
--seed SEED                         seed
--guidance-scale GUIDANCE_SCALE     guidance scale
--inference-steps [1-100]           the nummber of inference steps
--height [16-1024]                  the height in pixels of the generated image
--width [16-1024]                   the width in pixels of the generated image
-h, --help                          show this help message and exit
```

## Credits

Thanks a lot all developers behind libraries used in this project:
- [Diffusers](https://github.com/huggingface/diffusers) 
- [Kandinsky 2.2](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky_v22)
- [Kandinsky Community](https://huggingface.co/kandinsky-community)
