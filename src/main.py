import os
import io
import requests
from datetime import datetime, timezone
from fastapi import Response, HTTPException, Query, Request
import modal

# Define the image with required packages
image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi[standard]",
        "transformers",
        "accelerate",
        "diffusers",
        "torch",
        "requests"
    )
)

# Create the Modal app
app = modal.App("flux-demo", image=image)

@app.cls(
    gpu="A10G",
    secrets=[modal.Secret.from_name("API_KEY")]  # Add the secret here
)
class Model:
    def __init__(self):
        self.pipe = None

    @modal.enter()
    def initialize(self):
        """Initialize the model during startup."""
        from diffusers import FluxPipeline
        import torch

        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16
        )
        self.pipe.to("cuda")

    @modal.method()
    def generate_image(self, prompt: str):
        """Generate an image based on the prompt."""
        image = self.pipe(
            prompt,
            num_inference_steps=1,
            guidance_scale=0.0
        ).images[0]
        
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()

# Initialize the model
model = Model()

@app.function(secrets=[modal.Secret.from_name("API_KEY")])  # Add the secret here
@modal.web_endpoint()
async def generate(
    request: Request,
    prompt: str = Query(..., description="The prompt for image generation"),
):
    """Generate an image based on the prompt."""
    api_key = request.headers.get("X-API-Key")
    expected_api_key = os.environ["API_KEY"]  # Get the API key from environment
    
    if api_key != expected_api_key:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )

    try:
        image_bytes = model.generate_image.remote(prompt)
        return Response(content=image_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating image: {str(e)}"
        )

@app.function()
@modal.web_endpoint()
async def health():
    """Lightweight endpoint for keeping the container warm."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.function(
    schedule=modal.Cron("*/5 * * * *"),
    secrets=[modal.Secret.from_name("API_KEY")]  # Add the secret here
)
def keep_warm():
    """Function to keep the app warm by periodically pinging the endpoints."""
    health_url = "https://pentagram-app--flux-demo-health.modal.run"
    generate_url = "https://pentagram-app--flux-demo-generate.modal.run?prompt=test"

    # Check health endpoint
    try:
        health_response = requests.get(health_url)
        print(f"Health check at: {health_response.json()['timestamp']}")
    except Exception as e:
        print(f"Health check failed: {str(e)}")

    # Test generate endpoint
    try:
        headers = {"X-API-Key": os.environ["API_KEY"]}  # Get the API key from environment
        generate_response = requests.get(generate_url, headers=headers)
        print(f"Generate endpoint tested at: {datetime.now(timezone.utc).isoformat()}")
    except Exception as e:
        print(f"Generate test failed: {str(e)}")

if __name__ == "__main__":
    modal.runner.deploy_stub(app)