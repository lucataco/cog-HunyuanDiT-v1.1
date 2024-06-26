# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import argparse
import subprocess
from dialoggen.dialoggen_demo import DialogGen
from hydit.constants import SAMPLER_FACTORY
from hydit.inference import End2End

SAMPLERS = list(SAMPLER_FACTORY.keys())
SIZES = {"square": (1024, 1024), "landscape": (768, 1280), "portrait": (1280, 768)}


MODEL_URL = "https://weights.replicate.delivery/default/Tencent-Hunyuan/HunyuanDiT-v1.1/model.tar"
MODEL_CACHE = "model_cache"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        default_args = argparse.Namespace(
            prompt="a cute cat",
            image_size=[1024, 1024],
            seed=42,
            infer_steps=20,
            negative=None,
            infer_mode="torch",
            sampler="ddpm",
            enhance=False,
            model_root=MODEL_CACHE,
            load_key="ema",
            load_4bit=False,
            model="DiT-g/2",
            learn_sigma=True,
            text_states_dim=1024,
            text_states_dim_t5=2048,
            text_len=77,
            text_len_t5=256,
            norm="layer",
            use_flash_attn=False,
            qk_norm=True,
            lora_ckpt=None,
            noise_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.03,
            predict_type="v_prediction",
            use_fp16=True,
        )
        print(default_args)
        default_args.model_root = MODEL_CACHE
        self.gen = End2End(default_args, MODEL_CACHE)
        self.enhancer = DialogGen(f"{MODEL_CACHE}/dialoggen", default_args.load_4bit)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt", default="a cute cat"
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="",
        ),
        size: str = Input(
            description="Choose the output size. square: (1024, 1024), landscape: (768, 1280), portrait: (1280, 768).",
            choices=list(SIZES.keys()),
            default="square",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=40
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=6
        ),
        enhance_prompt: bool = Input(
            description="Choose to enhance the prompt.", default=False
        ),
        sampler: str = Input(
            default="ddpm", choices=SAMPLERS, description="Choose a sampler."
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        enhanced_prompt = None
        if enhance_prompt:
            _, enhanced_prompt = self.enhancer(prompt)

        height, width = SIZES[size]
        results = self.gen.predict(
            prompt,
            height=height,
            width=width,
            seed=seed,
            enhanced_prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            infer_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            batch_size=1,
            src_size_cond=(1024, 1024),
            sampler=sampler,
        )
        image = results["images"][0]
        output_path = "/tmp/out.png"
        image.save(output_path)
        return Path(output_path)
