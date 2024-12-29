import io
from fastapi import Response, HTTPException, Query, Request
from datetime import datetime, timezone
import os
import modal

def download_model():
    from diffusers import DiffusionPipeline
    import torch

    hf_token = os.environ["HUGGINGFACE_TOKEN"]

    DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        use_auth_token=hf_token,
        trust_remote_code=True
    )

image = (
    # Use Python 3.12 as specified
    modal.Image.debian_slim(python_version="3.12")
    # Upgrade pip to ensure compatibility with the latest packages
    .run_commands("python -m pip install --upgrade pip")
    # Install PyTorch and TorchVision with CUDA 11.8 support
    .pip_install(
        "torch==2.5.1+cu118",
        "torchvision==0.16.2+cu118",
        index_url="https://download.pytorch.org/whl/cu118",
        extra_index_url="https://pypi.org/simple"
    )
    # Install other dependencies
    .pip_install(
        "fastapi[standard]",
        "transformers",
        "accelerate",
        "diffusers",
        "requests",
        "sentencepiece",
        "safetensors",
        "scipy",
        "omegaconf",
        "pillow",
        "xformers",
        "triton"
    )
    .run_function(
        download_model,
        secrets=[modal.Secret.from_name("huggingface-secret")]
    )
)

app = modal.App("flux-demo", image=image)

@app.cls(
    image=image,
    gpu="A10G",
    secrets=[
        modal.Secret.from_name("API_KEY"),
        modal.Secret.from_name("huggingface-secret")
    ]
)
class Model:

    @modal.build()
    @modal.enter()
    def load_weights(self):
        from diffusers import DiffusionPipeline
        import torch

        hf_token = os.environ["HUGGINGFACE_TOKEN"]

        self.pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            use_auth_token=hf_token,
            trust_remote_code=True
        ).to("cuda")

        self.pipe.enable_model_cpu_offload()

        self.API_KEY = os.environ["API_KEY"]

    @modal.web_endpoint()
    async def generate(
        self, request: Request, prompt: str = Query(..., description="The prompt for image generation")
    ):
        import torch

        api_key = request.headers.get("X-API-Key")
        if api_key != self.API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized"
            )

        # Generate the image
        image = self.pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cuda").manual_seed(0)
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        return Response(content=buffer.getvalue(), media_type="image/jpeg")

    @modal.web_endpoint()
    async def health(self):
        """Lightweight endpoint for keeping the container warm"""
        return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

# Warm-keeping function that runs every 5 minutes
@app.function(
    schedule=modal.Cron("*/5 * * * *"),
    secrets=[
        modal.Secret.from_name("API_KEY"),
        modal.Secret.from_name("huggingface-secret")
    ]
)
def keep_warm():
    import requests
    health_url = "https://your-health-endpoint-url"
    generate_url = "https://your-generate-endpoint-url"

    # First check health endpoint (no API key needed)
    health_response = requests.get(health_url)
    print(f"Health check at: {health_response.json()['timestamp']}")

    # Then make a test request to generate endpoint with API key
    headers = {"X-API-Key": os.environ["API_KEY"]}
    params = {"prompt": "Test prompt"}
    generate_response = requests.get(generate_url, headers=headers, params=params)
    print(f"Generate endpoint tested successfully at: {datetime.now(timezone.utc).isoformat()}")