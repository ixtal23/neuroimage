import logging as log

from contextlib import ContextDecorator

from .configuration import Configuration

from diffusers import DiffusionPipeline, AutoPipelineForText2Image

from torch import float16, Generator

class Application(ContextDecorator):
    def __init__(self):
        self.configuration = Configuration()

        log.basicConfig(level = self.configuration.log_level, format = self.configuration.log_format)

    def __enter__(self):
        log.info('Start')
        return self

    def __exit__(self, *args):
        log.info('Finish')

    def __generate_kandinsky_2_2(self, generator : Generator):
        log.info(f'Generate image by Kandinsky 2.2')

        pipe_prior = DiffusionPipeline.from_pretrained('kandinsky-community/kandinsky-2-2-prior', torch_dtype = float16)
        pipe_prior.to("cuda")

        pipe_decoder = DiffusionPipeline.from_pretrained('kandinsky-community/kandinsky-2-2-decoder', torch_dtype = float16)
        pipe_decoder.to("cuda")

        image_embeds, negative_image_embeds = pipe_prior(
            prompt = self.configuration.prompt,
            negative_prompt = self.configuration.negative_prompt,
            guidance_scale = self.configuration.guidance_scale, 
            generator = generator
        ).to_tuple()

        return pipe_decoder(
            image_embeds = image_embeds, 
            negative_image_embeds = negative_image_embeds,
            num_inference_steps = self.configuration.num_inference_steps,
            height = self.configuration.height,
            width = self.configuration.width,
            generator = generator
        )

    def __generate_kandinsky_3_0(self, generator : Generator):
        log.info(f'Generate image by Kandinsky 3.0')

        pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-3", variant = "fp16", torch_dtype = float16)
        pipe.enable_model_cpu_offload()

        return pipe(
            prompt = self.configuration.prompt,
            negative_prompt = self.configuration.negative_prompt,
            guidance_scale = self.configuration.guidance_scale, 
            num_inference_steps = self.configuration.num_inference_steps,
            height = self.configuration.height,
            width = self.configuration.width,
            generator = generator
        )

    def __generate(self) -> None:
        log.info(f'Generate image')

        generator = Generator(device = "cuda")
        if self.configuration.seed > 0:
            generator.manual_seed(self.configuration.seed)

        log.info(f'initial_seed={generator.initial_seed()}')

        match self.configuration.kandinsky_version:
            case '2.2':
                result = self.__generate_kandinsky_2_2(generator)
            case '3.0':
                result = self.__generate_kandinsky_3_0(generator)

        result.images[0].save(self.configuration.output_file)

    def run(self) -> None:
        if self.configuration.parse_command_line():
            self.__generate()
