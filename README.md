# NeuroImage

This is an image generation tool that implements the generating of images by the following diffusion models
- [Kandinsky 2.2](https://huggingface.co/kandinsky-community/kandinsky-2-2-prior)
- [Kandinsky 3.0](https://huggingface.co/kandinsky-community/kandinsky-3)

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
               [--height [64-1024]]
               [--width [64-1024]]
               [--kandinsky-version {2.2,3.0}]
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
--height [64-1024]                  the height in pixels of the generated image
--width [64-1024]                   the width in pixels of the generated image
--kandinsky-version {2.2,3.0}       the version of Kandinsky model
-h, --help                          show this help message and exit
```

## Credits

Thanks a lot all developers behind libraries used in this project:
- [Diffusers Community](https://huggingface.co/docs/diffusers/index)
- [Diffusers GitHub](https://github.com/huggingface/diffusers) 
- [Kandinsky Community](https://huggingface.co/kandinsky-community)
- [Kandinsky 2.2 GitHub](https://github.com/ai-forever/Kandinsky-2)
- [Kandinsky 3.0 GitHub](https://github.com/ai-forever/Kandinsky-3)
