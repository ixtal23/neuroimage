import logging as log

from contextlib import ContextDecorator

from .configuration import Configuration

from diffusers import DiffusionPipeline

from torch import float16, Generator

# Windows not yet supported for torch.compile.
#from torch import channels_last, compile

class Application(ContextDecorator):
    def __init__(self):
        self.configuration = Configuration()

        log.basicConfig(level = self.configuration.log_level, format = self.configuration.log_format)

    def __enter__(self):
        log.info('Start')
        return self

    def __exit__(self, *args):
        log.info('Finish')

    def __generate(self) -> None:
        log.info(f'Generate image')
        log.info(f'prompt={self.configuration.prompt}')
        log.info(f'negative_prompt={self.configuration.negative_prompt}')
        log.info(f'seed={self.configuration.seed}')
        log.info(f'guidance_scale={self.configuration.guidance_scale}')
        log.info(f'num_inference_steps={self.configuration.num_inference_steps}')
        log.info(f'height={self.configuration.height}')
        log.info(f'width={self.configuration.width}')

        generator = Generator(device = "cuda")
        if self.configuration.seed > 0:
            generator.manual_seed(self.configuration.seed)

        log.info(f'initial_seed={generator.initial_seed()}')

        pipe_prior = DiffusionPipeline.from_pretrained('kandinsky-community/kandinsky-2-2-prior', torch_dtype = float16)
        pipe_prior.to("cuda")

        pipe_decoder = DiffusionPipeline.from_pretrained('kandinsky-community/kandinsky-2-2-decoder', torch_dtype = float16)
        pipe_decoder.to("cuda")

        # Windows not yet supported for torch.compile.
        #pipe_decoder.unet.to(memory_format = channels_last)
        #pipe_decoder.unet = compile(pipe_decoder.unet, mode = "reduce-overhead", fullgraph = True)

        image_embeds, negative_image_embeds = pipe_prior(
            prompt = self.configuration.prompt,
            negative_prompt = self.configuration.negative_prompt,
            guidance_scale = self.configuration.guidance_scale, 
            generator = generator
        ).to_tuple()

        image = pipe_decoder(
            image_embeds = image_embeds, 
            negative_image_embeds = negative_image_embeds,
            num_inference_steps = self.configuration.num_inference_steps,
            height = self.configuration.height,
            width = self.configuration.width,
            generator = generator
        ).images[0]

        image.save(self.configuration.output_file)

    def run(self) -> None:
        if self.configuration.parse_command_line():
            self.__generate()
