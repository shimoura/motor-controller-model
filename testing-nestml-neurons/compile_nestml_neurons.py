# Generate NEST target code from NESTML files

from pynestml.codegeneration.nest_code_generator_utils import NESTCodeGeneratorUtils
from pathlib import Path

nestml_neuron_path = "../nestml-neurons/controller_module.nestml"
nestml_target_path = "../nestml-neurons/nestml_target/"

print(f"Generating NEST target code from {nestml_neuron_path} to {nestml_target_path}")

NESTCodeGeneratorUtils.generate_code_for(
    nestml_neuron_model=str(nestml_neuron_path),
    target_path=str(nestml_target_path),
    module_name="motor_neuron_module",
)

import nest
from pathlib import Path

# --- 1. Setup ---
nest.ResetKernel()
try:
    nest.Install("motor_neuron_module")
    print("NESTML target code generated and installed successfully.")
except nest.NESTError:
    exit()
