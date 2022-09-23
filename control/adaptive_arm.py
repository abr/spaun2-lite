import logging
import timeit

import nengo
import numpy as np

from nengo_fpga.networks import FpgaPesEnsembleNetwork
import nengo_fpga

import matplotlib.pyplot as plt

# Set the nengo logging level to 'info' to display all of the information
# coming back over the ssh connection.
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# ---------------- BOARD SELECT ----------------------- #
# Change this to your desired device name
board = "pynq2"
# ---------------- BOARD SELECT ----------------------- #

def input_func(t):
    return [
            np.sin(t * 2*np.pi),
            np.cos(t * 2*np.pi),
            np.sin(t * 3*np.pi),
            np.cos(t * 3*np.pi),
            np.sin(t * 7*np.pi),
        ]

dt = 0.001

with nengo.Network(seed=3) as model:
    model.times = []
    def output_func(t, x):
        if t == dt:
            model.times.append(0)
            model.last = timeit.default_timer()
        else:
            now = timeit.default_timer()
            model.times.append(now - model.last)
            model.last = now

        return x


    # Input stimulus
    input_node = nengo.Node(input_func)

    # "Pre" ensemble of neurons, and connection from the input
    ens_fpga = FpgaPesEnsembleNetwork(board, n_neurons=6500,
                                      dimensions=5,
                                      learning_rate=1e-4)
    nengo.Connection(input_node, ens_fpga.input)  # Note the added '.input'

    # "Post" ensemble of neurons, and connection from "Pre"
    output = nengo.Node(output_func, size_in=5, size_out=5)
    conn = nengo.Connection(ens_fpga.output, output)  # Note the added '.output'

    # Create an ensemble for the error signal
    # Error = actual - target = "output" - input
    error = nengo.Node(size_in=5, size_out=5)
    nengo.Connection(output, error)
    nengo.Connection(input_node, error, transform=-1)

    # Connect the error into the learning rule
    nengo.Connection(error, ens_fpga.error)  # Note the added '.error'

    in_probe = nengo.Probe(input_node)
    out_probe = nengo.Probe(output)

with nengo_fpga.Simulator(model, dt=dt) as sim:
   sim.run(10)

plt.figure(figsize=(8,8))
plt.plot(model.times)
plt.savefig('times.png')

fig, axes = plt.subplots(5, 1, figsize=(8,8))
axes[0].set_title('Learning Communication Channel')
for ii in range(0, 5):
    axes[ii].plot(sim.data[out_probe][:, ii], label='learned output')
for ii in range(0, 5):
    axes[ii].plot(sim.data[in_probe][:, ii], linestyle='--', label='input')
    axes[ii].legend()

plt.tight_layout()
plt.show()
