import logging as log

import argparse

from pathlib import Path

class Configuration:
    def __init__(self):
        self.output_file : Path = None

        self.prompt : str = None
        self.negative_prompt : str = None

        self.seed : int = 0
    
        self.guidance_scale : float = 3.0

        self.num_inference_steps : int = 100

        self.height : int = 512
        self.width : int = 512

        self.kandinsky_version : str = None

        self.log_level = log.DEBUG
        self.log_format : str = '%(asctime)s   %(levelname)s   %(message)s'

    def __get_height_and_width(self, h : int, w : int) -> [int, int]:
        new_h = h // 64
        if h % 64 != 0:
            new_h += 1

        new_w = w // 64
        if w % 64 != 0:
            new_w += 1

        return new_h * 64, new_w * 64

    def __log(self) -> None:
        log.info(f'Configuration:')
        log.info(f'output_file={self.output_file}')
        log.info(f'prompt={self.prompt}')
        log.info(f'negative_prompt={self.negative_prompt}')
        log.info(f'seed={self.seed}')
        log.info(f'guidance_scale={self.guidance_scale}')
        log.info(f'num_inference_steps={self.num_inference_steps}')
        log.info(f'height={self.height}')
        log.info(f'width={self.width}')               
        log.info(f'kandinsky_version={self.kandinsky_version}')

    def __validate(self) -> bool:
        log.info('Validate configuration')

        if self.output_file.exists():
            log.error(f'Output file {self.output_file} already exists')
            return False

        log.info('Configuration is ok')
        return True

    def parse_command_line(self) -> bool:
        log.info('Parse command line')

        parser = argparse.ArgumentParser(
            prog = 'NeuroImage',
            description = 'Image generation tool.',
            formatter_class = lambda prog : argparse.HelpFormatter(prog, max_help_position = 100)
        )

        parser.add_argument('--output-file', help = 'a path to an output file', dest = 'output_file', type = Path, required = True)

        parser.add_argument('--prompt', help = 'prompt', dest = 'prompt', type = str,  required = True)
        parser.add_argument('--negative-prompt', help = 'negative prompt', dest = 'negative_prompt', type = str)

        parser.add_argument('--seed', help = 'seed', dest = 'seed', type = int, default = 0)

        parser.add_argument('--guidance-scale', help = 'guidance scale', dest = 'guidance_scale', type = float, default = 3.0)

        parser.add_argument('--inference-steps', help = 'the nummber of inference steps', dest = 'num_inference_steps', type = int, choices = range(1, 101), metavar = '[1-100]', default = 100)

        parser.add_argument('--height', help = 'the height in pixels of the generated image', dest = 'height', type = int, choices = range(64, 1025), metavar = '[64-1024]', default = 512)
        parser.add_argument('--width', help = 'the width in pixels of the generated image', dest = 'width', type = int, choices = range(64, 1025), metavar = '[64-1024]', default = 512)

        parser.add_argument('--kandinsky-version', help = 'the version of Kandinsky model', dest = 'kandinsky_version', type = str, choices = ['2.2', '3.0'], default = '3.0')

        args = parser.parse_args()

        self.output_file = args.output_file
        self.prompt = args.prompt
        self.negative_prompt = args.negative_prompt
        self.seed = args.seed
        self.guidance_scale = args.guidance_scale
        self.num_inference_steps = args.num_inference_steps
        self.height, self.width = self.__get_height_and_width(args.height, args.width)
        self.kandinsky_version = args.kandinsky_version

        if self.__validate():
            self.__log()
            return True

        return False
