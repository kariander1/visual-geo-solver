import hydra
from loguru import logger
import subprocess
from datetime import datetime
from pathlib import Path


def remove_weight_prefixes(state_dict):
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        parts = k.split('.')
        # Remove 'module' and/or '_orig_mod' if they appear at the start
        while parts and parts[0] in ('module', '_orig_mod'):
            # module is added because of ddp
            # _orig_mod is added because of torch.compile()
            
            parts = parts[1:]
        new_key = '.'.join(parts)
        new_state_dict[new_key] = v

    return new_state_dict


def get_outputs_path():
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    return hydra_cfg['runtime']['output_dir']
    
    
def save_git_diff(output_dir, filename_prefix="git_diff"):
    """
    Saves the current uncommitted Git diff to a file in the specified directory.

    Args:
        output_dir (str or Path): Directory where the diff file will be saved.
        filename_prefix (str): Prefix for the filename (default is 'git_diff').

    Returns:
        Path to the saved diff file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    diff_path = output_dir / f"{filename_prefix}_{timestamp}.patch"

    try:
        # Run `git diff`
        result = subprocess.run(
            ["git", "diff"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        with open(diff_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)

        logger.info(f"Git diff saved to: {diff_path}")
        return diff_path

    except subprocess.CalledProcessError as e:
        logger.error("Failed to get git diff:", e.stderr)
        return None