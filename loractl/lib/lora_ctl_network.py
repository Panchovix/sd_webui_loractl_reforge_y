from modules import extra_networks, script_callbacks, shared
from loractl.lib import utils

import sys, importlib
from pathlib import Path

# extensions-builtin isn't normally referencable due to the dash; this hacks around that
lora_path = str(Path(__file__).parent.parent.parent.parent.parent / "extensions-builtin" / "Lora")
sys.path.insert(0, lora_path)
import network, networks, extra_networks_lora
sys.path.remove(lora_path)

from modules.processing import StableDiffusionProcessing
from modules.extra_networks import ExtraNetworkParams
from modules_forge.forge_sampler import sampling_prepare

lora_weights = {}

params_map = {}

def reset_weights():
    lora_weights.clear()
    params_map.clear()

class LoraCtlNetwork(extra_networks_lora.ExtraNetworkLora):
    def __init__(self):
        self.params_list = []
        super().__init__()

    def clear(self):
        self.params_list = []
        reset_weights()

    def reload_weights_for_step(self, p, d):
        if not utils.is_active():
            return

        # We could skip reload on step 0, but I'm not sure how this interracts with batching?
        if d['i'] in lora_weights:
            for (name, weight) in lora_weights[d['i']].items():
                params_map[name].positional = [name, str(weight)]
                params_map[name].named = {}
            super().activate(p, self.params_list)
            sampling_prepare(p.sd_model.forge_objects.unet, d['x'])
            p.sd_model.forge_objects = p.sampler.model_wrap.inner_model.forge_objects_after_applying_lora.shallow_copy()
            p.scripts.process_before_every_sampling(p)


    def activate(self, p: StableDiffusionProcessing, params_list: list[ExtraNetworkParams]):
        if not utils.is_active():
            return super().activate(p, params_list)

        self.params_list = params_list
        for params in params_list:
            assert params.items
            name = params.positional[0]
            # Get the initial weight if it exists
            initial_weight = 1.0
            if len(params.positional) > 1:
                try:
                    # First convert to string to safely check for special characters
                    weight_str = str(params.positional[1])
                    # Check if it's a simple weight value
                    if '@' not in weight_str and ',' not in weight_str:
                        initial_weight = float(params.positional[1])
                except ValueError:
                    pass
            weights = utils.params_to_weights(params, p.steps)
            for (start_step, value) in weights.items():
                if start_step not in lora_weights:
                    lora_weights[start_step] = {}
                lora_weights[start_step][name] = value
            # Use the initial weight instead of hardcoded 1
            params.positional = [name, initial_weight]
            params.named = {}
            params_map[name] = params
        
        for step in sorted(lora_weights, reverse = True):
            for (name, weight) in lora_weights[step].items():
                params_map[name].positional = [name, str(weight)]
                params_map[name].named = {}
        super().activate(p, self.params_list)
