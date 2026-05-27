import sys
import os
sys.path.append(os.path.abspath('src'))
import torch
import time
from moe_route.routing.routers import RouterConfig, build_router

def profile_router(kind: str, seq_len: int = 4096, batch_size: int = 8, d_model: int = 4096, num_experts: int = 64, top_k: int = 2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Profiling {kind} router on {device} ---")
    
    cfg = RouterConfig(
        kind=kind,
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        capacity_factor=1.25,
        routing_mode="sparse" if kind == "reflected_v2" else "dense"
    )
    
    router = build_router(cfg).to(device)
    
    # Dummy input
    x = torch.randn(batch_size * seq_len, d_model, device=device, requires_grad=True)
    
    # Warmup
    for _ in range(5):
        res = router(x)
        if hasattr(router, "post_optimizer_step"):
            router.post_optimizer_step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    num_iters = 50
    for _ in range(num_iters):
        res = router(x)
        loss = res.diagnostics.aux_loss
        if loss is not None and loss.requires_grad:
            loss.backward()
        if hasattr(router, "post_optimizer_step"):
            router.post_optimizer_step()
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    avg_time_ms = (end_time - start_time) / num_iters * 1000
    print(f"Average time per forward+backward step: {avg_time_ms:.2f} ms")

if __name__ == "__main__":
    profile_router("topk")
    profile_router("deepseek_lfb")
    profile_router("reflected_v2")
