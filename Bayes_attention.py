import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from pathlib import Path
import os
from tqdm import tqdm

# Set number of workers for DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Cache for BayesVLM results
_bayes_cache = {}

def get_cached_bayes_results(image_path, hessian_dir):
    cache_key = f"{image_path}_{hessian_dir}"
    if cache_key not in _bayes_cache:
        from get_probcosine import get_probcosine_distribution
        _bayes_cache[cache_key] = get_probcosine_distribution(image_path, str(hessian_dir))
    return _bayes_cache[cache_key]

@torch.no_grad()
def adapter(embedding, num_samples=100, bayes_results=None, image_path=None):
    current_dir = Path(__file__).parent.absolute()
    hessian_dir = current_dir / "BayesVLM-main" / "hessians" / "hessian_CLIP-ViT-B-32-laion2B-s34B-b79K"
    
    # Print the path when getting results
    print(f"Using hessian directory: {hessian_dir}")
    
    try:
        if bayes_results is None:
            bayes_results = get_cached_bayes_results(image_path, str(hessian_dir))
        
        # Get BayesVLM parameters
        bayes_variance = bayes_results['variance'].item()
        kappa = bayes_results['kappa'].item()
        probas = bayes_results['probabilities'].squeeze()
        mean = bayes_results['mean'].squeeze()
        
        # Generate samples using BayesVLM distribution
        D = embedding.shape[-1]
        noise = torch.randn((num_samples, D), device=embedding.device) * torch.sqrt(bayes_variance)
        
        # Weight noise by probability distribution
        noise = noise * probas.unsqueeze(-1)
        
        # Apply kappa scaling and add to embedding
        samples = embedding + noise * kappa
        
        # Add influence from BayesVLM mean
        samples = samples + 0.1 * mean  # Small influence from BayesVLM mean
        
        # Normalize samples
        samples = torch.nn.functional.normalize(samples, p=2, dim=-1)
        return samples
    except Exception as e:
        print(f"Error in adapter: {str(e)}")
        # Fallback behavior
        D = embedding.shape[-1]
        noise = torch.randn((num_samples, D), device=embedding.device) * 0.1
        samples = embedding + noise
        samples = torch.nn.functional.normalize(samples, p=2, dim=-1)
        return samples

def main():
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Clear the cache
    global _bayes_cache
    _bayes_cache = {}
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Load and preprocess image
    image_path = "Rusty_Blackbird_0015_6885.jpg"
    print(f"Processing image: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    text = clip.tokenize(["a photo of a bird"]).to(device)

    # Get BayesVLM results - Fix the hessian directory path
    current_dir = Path(__file__).parent.absolute()
    hessian_dir = current_dir / "BayesVLM-main" / "hessians" / "hessian_CLIP-ViT-B-32-laion2B-s34B-b79K"
    
    # Print the path to verify
    print(f"Looking for hessian files in: {hessian_dir}")
    
    if not hessian_dir.exists():
        raise FileNotFoundError(f"Hessian directory not found at {hessian_dir}")
    
    # Check if required files exist
    required_files = ['A_img_analytic.pt', 'B_img_analytic.pt', 'A_txt_analytic.pt', 'B_txt_analytic.pt', 'prior_precision_analytic.json']
    for file in required_files:
        if not (hessian_dir / file).exists():
            raise FileNotFoundError(f"Required file {file} not found in {hessian_dir}")
    
    bayes_results = get_cached_bayes_results(image_path, str(hessian_dir))

    # Preprocess
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Encode text once
    with torch.no_grad():
        text_embedding = model.encode_text(text)
        text_embedding = torch.nn.functional.normalize(text_embedding, p=2, dim=-1)

    # Create a wrapper for the adapter that includes the image path
    def adapter_with_path(embedding, num_samples=100, bayes_results=None):
        return adapter(embedding, num_samples, bayes_results, image_path=image_path)

    # RISE with BayesVLM-informed Adapter
    from RISEWithAdapter import RISEWithAdapter
    from RISE import RISE

    # Generate RISE masks
    input_size = (224, 224)
    rise = RISE(model, input_size)

    if not os.path.exists('masks.npy'):
        print("Generating new masks...")
        rise.generate_masks(N=100, s=8, p1=0.5)
    else:
        print("Loading existing masks...")
        rise.load_masks('masks.npy')

    # Initialize RISEWithAdapter with the wrapped adapter
    explainer = RISEWithAdapter(
        clip_model=model,
        adapter=adapter_with_path,
        text_embedding=text_embedding,
        input_size=input_size,
        gpu_batch=50,
        num_samples=100
    )
    explainer.set_masks(rise.masks)

    # Compute saliency
    print("Computing saliency map...")
    with torch.no_grad():
        try:
            saliency_map = explainer(image_tensor, bayes_results=bayes_results)
        except Exception as e:
            print(f"Error computing saliency map: {str(e)}")
            raise

    # Process and visualize results
    saliency_np = saliency_map.cpu().numpy()
    saliency_np -= np.min(saliency_np)

    # Print statistics
    print("\nSaliency Map Statistics:")
    print(f"Min: {saliency_map.min().item():.4f}")
    print(f"Max: {saliency_map.max().item():.4f}")
    print(f"Mean: {saliency_map.mean().item():.4f}")
    print(f"Std Dev: {saliency_map.std().item():.4f}")

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(image)
    im = ax2.imshow(saliency_np, cmap='jet', alpha=0.5)
    plt.colorbar(im, ax=ax2, label='Saliency value')
    ax2.set_title("BayesVLM-informed Saliency Map")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()