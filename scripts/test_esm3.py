"""Test script to verify ESM-3 model loading and generation."""

import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, GenerationConfig, ESMProtein


def test_esm3():
    """Load ESM-3 and run a test generation from a fully masked sequence."""
    print("Loading ESM-3 model...")
    client: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1")
    
    print("Model loaded successfully.")
    print("Running test generation...")
    
    # Create an all-mask input of length 50
    # ESM3 uses underscores usually or specific mask tokens. 
    # Let's try to generate a sequence from a blank slate.
    protein_input = ESMProtein(sequence="_" * 50)
    
    try:
        # Generate with basic config
        protein = client.generate(
            input=protein_input,
            config=GenerationConfig(
                track="sequence", 
                num_steps=5, 
            )
        )
        print("Generation successful!")
        print(f"Generated sequence length: {len(protein.sequence)}")
        print(f"Sequence snippet: {protein.sequence[:20]}...")
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    test_esm3()
