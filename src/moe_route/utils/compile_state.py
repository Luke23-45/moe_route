from __future__ import annotations
from functools import wraps
import os
import torch
from torch._dynamo import disable as dynamo_disable

class CompileState:
    """Centralized State Machine for torch.compile and dynamic logic.
    
    Manages the interaction between trainer configuration, GPU hardware support, 
    and compiler/allocator overrides (dynamic logic).
    """
    _compile_enabled: bool = False
    _dynamic_logic_active: bool = False

    @classmethod
    def initialize(cls, compile_requested: bool, device_type: str, is_main: bool = True) -> tuple[bool, bool]:
        """Initializes the compile state machine.
        
        Args:
            compile_requested: Whether the user/config requested compilation (e.g. trainer.compile).
            device_type: Device type, e.g. 'cuda' or 'cpu'.
            is_main: Whether the current process is the main process (for clean logging).
            
        Returns:
            A tuple of (compile_enabled, dynamic_logic_active).
        """
        # If compilation is explicitly requested as False, disable compilation and deactivate all dynamic logic overrides.
        if not compile_requested:
            cls._compile_enabled = False
            cls._dynamic_logic_active = False
            if is_main:
                print("[train] Compilation explicitly disabled via config (trainer.compile=false). Dynamic compilation logic removed.")
            return False, False

        # If compile_requested is True, validate GPU support
        if device_type != "cuda":
            cls._compile_enabled = False
            cls._dynamic_logic_active = False
            if is_main:
                print("[train] WARNING: torch.compile requires CUDA. Disabling compilation and removing dynamic logic.")
            return False, False

        # Check GPU compute capability
        capability = torch.cuda.get_device_capability()
        
        # SOTA Check: GPUs with compute capability < 8.0 (like T4, Volta, etc.) 
        # do not gain performance benefits from torch.compile.
        # So we automatically disable compilation and its dynamic memory/compiler overhead logic.
        if capability[0] < 8:
            cls._compile_enabled = False
            cls._dynamic_logic_active = False
            if is_main:
                print(
                    f"[train] WARNING: GPU compute capability is {capability[0]}.{capability[1]} (e.g. T4/Volta architecture). "
                    f"torch.compile yields no performance benefits on this architecture. "
                    f"Automatically disabling compilation and removing all dynamic compile/allocator logic."
                )
            return False, False

        # GPU supports efficient compilation (Compute Capability >= 8.0, e.g., Ampere, Hopper)
        cls._compile_enabled = True
        cls._dynamic_logic_active = True

        if is_main:
            print(
                f"[train] GPU compute capability is {capability[0]}.{capability[1]} (Ampere/Hopper or newer). "
                f"SOTA compilation is fully supported! Enabling torch.compile and activating dynamic compiler/allocator logic."
            )

        # Activate dynamic compiler and allocator settings
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

        return True, True

    @classmethod
    def is_compile_enabled(cls) -> bool:
        return cls._compile_enabled

    @classmethod
    def is_dynamic_logic_active(cls) -> bool:
        return cls._dynamic_logic_active


def conditional_dynamo_disable(func):
    """Decorator that conditionally disables PyTorch Dynamo tracing on the decorated function.
    
    If CompileState dynamic logic is active (which requires hardware support and trainer config enabling compile),
    it applies torch._dynamo.disable to prevent dynamic shape recompilations/bloat.
    Otherwise, it returns the original function untouched to run natively.
    """
    disabled_func = dynamo_disable(func)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        if CompileState.is_dynamic_logic_active():
            return disabled_func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper
