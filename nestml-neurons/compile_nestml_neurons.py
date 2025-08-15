# Generate NEST target code from NESTML files

from pynestml.frontend.pynestml_frontend import generate_nest_target

nestml_file_path = "."
nestml_target_path = nestml_file_path + "/nestml_target/"
nestml_install_path = nestml_file_path + "/nestml_install/"

print(f"Generating NEST target code from {nestml_file_path} to {nestml_target_path}")

generate_nest_target(
    input_path=str(nestml_file_path),
    target_path=str(nestml_target_path),
    install_path=str(nestml_install_path),
    module_name="motor_neuron_module",
)

import nest

# --- 1. Setup ---
nest.ResetKernel()
try:
    nest.Install("motor_neuron_module")
    print("NESTML target code generated and installed successfully.")
except nest.NESTError:
    exit()
