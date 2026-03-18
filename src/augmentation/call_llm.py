"""
LLM API calls for data augmentation.
Supports OpenAI, Azure OpenAI, and Anthropic.

Data schema (RFC-BENCH):
  "Open-ended Verifiable Question" → paragraph text (after stripping prompt prefix)
  "Ground-True Answer"             → "The provided information is true/false."

Provider setup
--------------
OpenAI:
    os.environ['LLM_PROVIDER'] = 'openai'
    os.environ['OPENAI_API_KEY'] = 'sk-...'

Azure OpenAI:
    os.environ['LLM_PROVIDER']            = 'azure'
    os.environ['AZURE_OPENAI_API_KEY']    = '<32-char hex key>'
    os.environ['AZURE_OPENAI_ENDPOINT']   = 'https://<resource>.openai.azure.com/'
    os.environ['AZURE_OPENAI_DEPLOYMENT'] = '<deployment-name>'   # e.g. 'gpt-4o'
    os.environ['AZURE_OPENAI_API_VERSION'] = '2024-02-01'         # optional

Anthropic:
    os.environ['LLM_PROVIDER']    = 'anthropic'
    os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'

Usage (CLI):
    python call_llm.py --project-dir /path/to/fmd \\
                       --output data/augmented/augmented_train.json \\
                       --perturbations 3
"""
import os
import json
import sys
import time
import argparse
import random
from pathlib import Path
from tqdm import tqdm

# Allow imports whether run as a script or imported as a module
_src_dir = str(Path(__file__).resolve().parents[1])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from augmentation.perturbation_prompts import PERTURBATION_PROMPTS, build_prompt
from utils.data_loader import load_combined_data, load_raw_file


# ── LLM client setup ─────────────────────────────────────────────────────────

def get_llm_client():
    """
    Build and return (client, provider) based on LLM_PROVIDER env var.

    Supported providers: 'openai', 'azure', 'anthropic'.
    Raises EnvironmentError immediately if required env vars are missing.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "No API key found for OpenAI.\n"
                "Set:  os.environ['OPENAI_API_KEY'] = 'sk-...'"
            )
        from openai import OpenAI
        return OpenAI(api_key=api_key), provider

    if provider == "azure":
        api_key  = os.getenv("LLM_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        missing = [k for k, v in {
            "AZURE_OPENAI_API_KEY":    api_key,
            "AZURE_OPENAI_ENDPOINT":   endpoint,
            "AZURE_OPENAI_DEPLOYMENT": deployment,
        }.items() if not v]
        if missing:
            raise EnvironmentError(
                f"Missing Azure OpenAI env vars: {missing}\n"
                "Set:\n"
                "  os.environ['AZURE_OPENAI_API_KEY']    = '<hex-key>'\n"
                "  os.environ['AZURE_OPENAI_ENDPOINT']   = 'https://<resource>.openai.azure.com/'\n"
                "  os.environ['AZURE_OPENAI_DEPLOYMENT'] = '<deployment-name>'"
            )
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        # Store deployment name on client so call_llm can use it as the model
        client._fmd_deployment = deployment
        return client, provider

    if provider == "anthropic":
        api_key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "No API key found for Anthropic.\n"
                "Set:  os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'"
            )
        import anthropic
        return anthropic.Anthropic(api_key=api_key), provider

    raise ValueError(
        f"Unsupported LLM_PROVIDER: '{provider}'. "
        "Choose 'openai', 'azure', or 'anthropic'."
    )


def call_llm(client, provider: str, prompt: str,
             model: str = None, max_tokens: int = 512) -> str:
    """Call LLM and return the generated text."""
    if provider in ("openai", "azure"):
        # For Azure: use deployment name as model; for OpenAI: default to gpt-4o-mini
        default_model = getattr(client, "_fmd_deployment", None) or "gpt-4o-mini"
        resp = client.chat.completions.create(
            model=model or default_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()

    if provider == "anthropic":
        resp = client.messages.create(
            model=model or "claude-3-haiku-20240307",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    raise ValueError(f"Unknown provider: {provider}")


# ── Augmentation pipeline ─────────────────────────────────────────────────────

def augment_dataset(
    project_dir: str,
    output_path: str,
    perturbations_per_sample: int = 3,
    model: str = None,
    delay: float = 0.5,
    resume: bool = True,
) -> None:
    """
    Generate augmented (False-label) samples from True-label training data.

    Loads both SFT + RL raw files via data_loader, applies LLM perturbations
    only to True-label samples, and writes the combined dataset to output_path.

    Args:
        project_dir: Root of the FMD project (contains data/raw/).
        output_path: Where to write augmented dataset JSON.
        perturbations_per_sample: How many perturbation types to apply per sample.
        model: LLM model name override.
        delay: Seconds to sleep between API calls (rate limiting).
        resume: If True, skip already-processed samples (checkpoint resume).
    """
    client, provider = get_llm_client()

    # Load + normalise both raw files using the shared data loader
    all_records = load_combined_data(project_dir)

    # Only augment True-label samples (label == 1)
    true_samples = [r for r in all_records if r["label"] == 1]
    print(f"Found {len(true_samples)} True samples to augment.")

    # Load existing output for resume
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    existing_ids = set()
    if resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        existing_ids = {(d["source_id"], d["perturbation_type"]) for d in existing
                        if "source_id" in d}
        print(f"Resuming: {len(existing)} samples already generated.")

    augmented = list(existing)
    pert_types = list(PERTURBATION_PROMPTS.keys())

    for idx, sample in enumerate(tqdm(true_samples, desc="Augmenting")):
        sample_id = sample.get("id", idx)

        # Choose perturbation types for this sample
        selected = random.sample(pert_types, min(perturbations_per_sample, len(pert_types)))

        for pert_type in selected:
            if (sample_id, pert_type) in existing_ids:
                continue  # already done

            prompt = build_prompt(sample["text"], pert_type)
            try:
                generated_text = call_llm(client, provider, prompt, model=model)
            except Exception as e:
                print(f"⚠️  Error on sample {sample_id} / {pert_type}: {e}")
                time.sleep(5)
                continue

            augmented.append({
                "id":               f"{sample_id}_{pert_type}",
                "source_id":        sample_id,
                "text":             generated_text,
                "label":            0,  # False = misinformation
                "perturbation_type": pert_type,
                "original_text":    sample["text"],
            })
            existing_ids.add((sample_id, pert_type))
            time.sleep(delay)

        # Checkpoint every 50 source samples
        if (idx + 1) % 50 == 0:
            with open(output_path, "w") as f:
                json.dump(augmented, f, indent=2)
            print(f"✅ Checkpoint: {len(augmented)} augmented samples saved.")

    # Final save — include ALL original records + new augmented False samples
    # (original records already have both true and false labels from the raw data)
    original_records = [
        {"id": r["id"], "text": r["text"], "label": r["label"],
         "split": r["split"], "perturbation_type": None}
        for r in all_records
    ]

    final_dataset = original_records + augmented
    with open(output_path, "w") as f:
        json.dump(final_dataset, f, indent=2)

    n_true  = sum(1 for d in final_dataset if d["label"] == 1)
    n_false = sum(1 for d in final_dataset if d["label"] == 0)
    print(f"\n✅ Augmentation complete.")
    print(f"   True samples:  {n_true}")
    print(f"   False samples: {n_false}")
    print(f"   Total:         {len(final_dataset)}")
    print(f"   Saved to:      {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment FMD training data via LLM API.")
    parser.add_argument("--project-dir", required=True, help="FMD project root directory")
    parser.add_argument("--output", required=True, help="Path to output augmented JSON")
    parser.add_argument("--perturbations", type=int, default=3,
                        help="Number of perturbation types per sample (default: 3)")
    parser.add_argument("--model", default=None, help="LLM model name override")
    parser.add_argument("--delay", type=float, default=0.5, help="Sleep between API calls (s)")
    parser.add_argument("--no-resume", action="store_true", help="Restart from scratch")
    args = parser.parse_args()

    augment_dataset(
        project_dir=args.project_dir,
        output_path=args.output,
        perturbations_per_sample=args.perturbations,
        model=args.model,
        delay=args.delay,
        resume=not args.no_resume,
    )
