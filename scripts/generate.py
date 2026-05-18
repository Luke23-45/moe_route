import argparse
from pathlib import Path
import torch
from moe_route.training.checkpoint import load_checkpoint
from moe_route.evaluation.ppl import load_cfg_from_checkpoint
from moe_route.models.transformer import DecoderOnlyLM, build_model_cfg
from moe_route.tokenization.tokenizers import build_tokenizer

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=40, device="cuda"):
    model.eval()
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if hasattr(tokenizer, "bos_token_id"):
        input_ids = [tokenizer.bos_token_id] + input_ids
        
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # We only care about the last token's logits, and we must not exceed max_seq_len
        context = input_tensor[:, -model.cfg.max_seq_len:] if hasattr(model, "cfg") else input_tensor[:, -256:]
        logits, _, _ = model(context)
        next_token_logits = logits[0, -1, :] / temperature
        
        # Top-k filtering
        if top_k > 0:
            top_k_vals, _ = torch.topk(next_token_logits, top_k)
            min_val = top_k_vals[-1]
            next_token_logits[next_token_logits < min_val] = float('-inf')
            
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        input_tensor = torch.cat((input_tensor, next_token.unsqueeze(0)), dim=1)
        
        if hasattr(tokenizer, "eos_token_id") and next_token.item() == tokenizer.eos_token_id:
            break

    output_text = tokenizer.decode(input_tensor[0].cpu().tolist())
    return output_text


def main():
    parser = argparse.ArgumentParser(description="Generate stories using a trained MoE/Dense checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained step_XXXX.pt checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little girl named Lily", help="Starting prompt")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading environment from {args.checkpoint}...")
    
    ckpt_path = Path(args.checkpoint)
    cfg = load_cfg_from_checkpoint(ckpt_path)
    
    tokenizer = build_tokenizer(cfg.tokenizer)
    model = DecoderOnlyLM(build_model_cfg(cfg)).to(device)
    load_checkpoint(ckpt_path, model)

    print(f"\n[Prompt]: {args.prompt}")
    print("-" * 50)
    
    story = generate(
        model, 
        tokenizer, 
        args.prompt, 
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device
    )
    
    print(story)
    print("-" * 50)

if __name__ == "__main__":
    main()
