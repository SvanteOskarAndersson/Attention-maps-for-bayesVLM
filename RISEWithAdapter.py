import torch
import torch.nn as nn
from tqdm import tqdm

class RISEWithAdapter(nn.Module):
    def __init__(self, clip_model, adapter, text_embedding, input_size, gpu_batch=50, num_samples=100):
        super(RISEWithAdapter, self).__init__()
        self.clip_model = clip_model.eval()           # CLIP image encoder
        self.adapter = adapter #.eval()                 # Adapter for embedding sampling
        self.text_embedding = text_embedding          # Precomputed CLIP text embedding
        self.input_size = input_size
        self.gpu_batch = gpu_batch
        self.num_samples = num_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_masks(self, masks):  # Expect pre-generated masks
        self.masks = masks.float().to(self.device)
        self.N = masks.shape[0]
#
    def forward(self, x, bayes_results=None):  # Added bayes_results parameter
        N = self.N
        _, _, H, W = x.size()
        saliency = torch.zeros((H, W), device=x.device)

        # Get original image embedding and similarity
        with torch.no_grad():
            orig_embedding = self.clip_model.encode_image(x, is_weights=None)
            orig_embedding = orig_embedding / orig_embedding.norm(dim=-1, keepdim=True)
            text_embedding = self.text_embedding / self.text_embedding.norm(dim=-1, keepdim=True)
            orig_similarity = (orig_embedding @ text_embedding.T).item()

        for i in tqdm(range(N), desc="Sampling attention via adapter"):
            masked_img = x * self.masks[i]  # [1, 3, H, W]
            with torch.no_grad():
                #compute embedding of pertubed image
                image_embedding = self.clip_model.encode_image(masked_img, is_weights=None)
                # get mu alpha and beta from ProbVLM here
                # sample num_samples from sample_ggd here, but now placeholder adapter
                sampled_embeddings = self.adapter(
                    image_embedding, 
                    self.num_samples,
                    bayes_results=bayes_results
                )
                #Normailze embedding tensor/matrix
                sampled_embeddings = sampled_embeddings / sampled_embeddings.norm(dim=-1, keepdim=True)
                #Normalize textembedding for cosine similarity 
                text_embedding = self.text_embedding / self.text_embedding.norm(dim=-1, keepdim=True)
                #Compute cosine similarities
                similarities = (sampled_embeddings @ text_embedding.T).squeeze(-1)
                #Weight similarities by probcosine distribution if available
                if bayes_results is not None:
                    probas = bayes_results['probabilities'].squeeze()
                    similarities = similarities * probas
                #Save the variance
                sim_variance = similarities.var().item()
                #compute unpertubed image embedding
                orig_embedding = self.clip_model.encode_image(x, is_weights=None)
                orig_embedding = orig_embedding / orig_embedding.norm(dim=-1, keepdim=True)
                #compute original similarity score
                orig_similarity = (orig_embedding @ text_embedding.T).item()
                mean_similarity = similarities.mean().item()
                delta_similarity = mean_similarity - orig_similarity  # Positive if similarity drops
                print(f"Mask {i}: Var={sim_variance:.4f}, Mean={mean_similarity:.4f}, Delta={delta_similarity:.4f}")
                uncertainty_weight = sim_variance / (sim_variance + 1e-6)  # Normalize to [0, 1]-ish range
                if bayes_results is not None:
                    prob_mean = bayes_results['mean'].squeeze().mean().item()
                    combined_score = delta_similarity * (1 + 0.5 * uncertainty_weight) * prob_mean
                else:
                    combined_score = delta_similarity * (1 + 0.5 * uncertainty_weight)

            saliency += combined_score * self.masks[i, 0]

        saliency = saliency / N
        return saliency
