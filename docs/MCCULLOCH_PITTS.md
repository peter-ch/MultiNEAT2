# McCulloch-Pitts neurons and 3D HyperNEAT

MultiNEAT2 supports McCulloch-Pitts (MCP) neurons as evolvable, stateful
spiking units. The implementation preserves the model's defining binary
threshold rule and absolute inhibitory veto while integrating it with
refractory timing, delayed exponential synapses, recurrent NEAT topology,
STDP, e-prop, recording, checkpointing, and spatial HyperNEAT phenotypes.

## Dynamics

At solver tick `t`, the current `I_i(t)` is the sum of the input and synaptic
currents arriving at neuron `i`. The neuron emits one binary spike when

```text
s_i(t) = 1  if I_i(t) >= threshold_i
                 and its refractory timer is zero
                 and no active inhibitory afferent vetoes it
         0  otherwise
```

An afferent is inhibitory when its delivered current is negative. With
`m_mcp_inhibitory_veto = true`, any such afferent suppresses the tick even if
the total excitatory drive exceeds threshold, matching the canonical
McCulloch-Pitts rule. Set it to `false` to use ordinary signed summation.
Synaptic current decay, axonal delay, refractory period, filtered firing-rate
readout, STDP, and spike recording use the common spiking solver.

The MCP model is intentionally discrete and nondifferentiable; it is not a
conductance-based membrane model. Its biologically grounded features here are
explicit excitation/inhibition, absolute inhibitory action, refractory
timing, physical axon length and conduction delay, recurrent connectivity,
and optional synaptic plasticity.

## Evolution

```python
import pymultineat as neat

params = neat.Parameters()
params.ConfigureMcCullochPitts(
    inhibitory_veto=True,
    enable_stdp=False,
)
params.PopulationSize = 120
params.DontUseBiasNeuron = True

init = neat.GenomeInitStruct()
init.NumInputs = 4
init.NumHidden = 4
init.NumLayers = 1
init.NumOutputs = 2
init.SeedType = neat.LAYERED
init.HiddenActType = neat.MCCULLOCH_PITTS
init.OutputActType = neat.MCCULLOCH_PITTS

seed = neat.Genome(params, init)
population = neat.Population(seed, params, True, 5.0, 42)
```

`ConfigureMcCullochPitts()` enables activation-type evolution, recurrent
structural mutation, and neuron/link spiking-parameter mutation. The relevant
controls are:

- `ActivationFunction_McCullochPitts_Prob`
- `InitialMCPInhibitoryVetoProb`
- `MutateMCPInhibitoryVetoProb`
- `MinSpikeThreshold` and `MaxSpikeThreshold`
- `MinRefractoryPeriod` and `MaxRefractoryPeriod`
- the common delay, synaptic-time-constant, STDP, and compatibility settings

The inhibitory-veto bit participates in crossover, compatibility distance,
genome serialization, and phenotype construction.

## Stateful execution

```python
network = neat.NeuralNetwork()
seed.BuildPhenotype(network)
network.SetSpikingInputMode(neat.BINARY_SPIKE_INPUT)
network.SetSpikingOutputMode(neat.SPIKE_OUTPUT)

for input_spikes in sequence:
    output_spikes = network.StepSpiking(input_spikes, 0.001)
```

`Flush()` resets all MCP runtime state. Both `NeuralNetwork.Serialize()` and
the file `Save()` API preserve the model configuration and live state.

## Fixed 3D HyperNEAT substrates

Coordinates may have three components. Generated phenotype neurons expose
them through `m_x`, `m_y`, `m_z`, and `m_substrate_coords`. Every connection
stores its Euclidean `m_length`. A substrate can reject long axons and derive
synaptic delay from distance and conduction velocity:

```python
inputs = [[-1.0, 0.0, -1.0], [0.0, 0.0, 0.0]]
hidden = [[0.0, -0.5, 0.5], [0.0, 0.5, -0.5]]
outputs = [[1.0, 0.0, 1.0]]
substrate = neat.Substrate(inputs, hidden, outputs)
substrate.m_hidden_nodes_activation = neat.MCCULLOCH_PITTS
substrate.m_output_nodes_activation = neat.MCCULLOCH_PITTS
substrate.m_max_connection_length = 2.5
substrate.m_use_spatial_distance_for_delays = True
substrate.m_conduction_velocity = 4.0

cppn_init = neat.GenomeInitStruct()
cppn_init.NumInputs = substrate.GetMinCPPNInputs()   # x1,y1,z1,x2,y2,z2,bias
cppn_init.NumOutputs = substrate.GetMinCPPNOutputs()
cppn_params = neat.Parameters()  # CPPNs normally use smooth rate activations
cppn = neat.Genome(cppn_params, cppn_init)

network = neat.NeuralNetwork()
cppn.BuildHyperNEATPhenotype(network, substrate)
```

`m_synaptic_delay` is set to `m_length / m_conduction_velocity`. Calling
`network.UpdateConnectionGeometry(True, velocity)` recomputes both after
manually changing neuron positions.

## 3D ES-HyperNEAT

ES-HyperNEAT automatically selects octree subdivision when any substrate
coordinate has a third component. Its CPPN receives source and target
`x, y, z` followed by optional distance and bias. `Parameters.Depth` and
`Parameters.Qtree_Z` control the spatial search volume in addition to
`Width`, `Height`, `Qtree_X`, and `Qtree_Y`.

```python
es = neat.Parameters()
es.InitialDepth = 1
es.MaxDepth = 3
es.Depth = 2.0
es.Qtree_Z = 0.0

substrate.m_hidden_coords = []  # ES-HyperNEAT discovers these positions
cppn.BuildESHyperNEATPhenotype(network, substrate, es)
```

The 3D safe depth limit is 6 (the 2D quadtree limit remains 9). Generated
hidden neurons, physical axon lengths, distance pruning, and conduction
delays use the same phenotype API as fixed-substrate HyperNEAT.

## Example variants

Every shipped example that can run a spiking policy accepts
`--mcculloch-pitts`, including the dedicated spiking examples, XOR,
Asteroids, every Box2D/MuJoCo wrapper, and the family suite runner:

```sh
python demos/hyperneat_3d.py --smoke --mcculloch-pitts
python demos/spiking_pattern.py --smoke --mcculloch-pitts
python demos/xor.py --smoke --mcculloch-pitts
python demos/asteroid_nav.py --smoke --mcculloch-pitts
python demos/box2d/lunar_lander_box2d.py --smoke --mcculloch-pitts
python demos/mujoco/inverted_pendulum_mujoco.py --smoke --mcculloch-pitts
python demos/run_gymnasium_suite.py --family box2d --smoke --mcculloch-pitts
build/Release/multineat_exe --smoke --mcculloch-pitts --seed 42
```

The root `python demos.py` launcher exposes the same choice as a dedicated
McCulloch-Pitts switch.
