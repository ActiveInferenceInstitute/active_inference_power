# GNN Example: Two-Target Action-Loop Reference Model

## GNNSection

ActInfTwoTargetActionLoop

## GNNVersionAndFlags

GNN v1.1

## ModelName

Two-Target Binary-Truth Action-Loop Reference Model

## ModelAnnotation

The agent-side generative model of `src/active_inference_power/scenario.py::make_reference_action_scenario`:
two targets x two contexts x binary truth (8 hidden states), a binary
observation modality (`low`, `high`), and six actions (per-target
cheap/precise sensing, `switch_context`, `stop`). Likelihood and transition
are action-conditioned; in the reference scenario the generative process
carries the same arrays, and `make_static_binary_e_process_scenario` reuses
the model with a state-invariant emission channel. The declared
`InitialParameterization` is that builder's default configuration
(`reliability=0.8`, `horizon=12`, `alpha=0.05`). Every declared array is
deterministic, never fitted: `D` is uniform over the eight states, `C` is
flat, each `B` column is one-hot, and `A` places the quality
`min(0.98, reliability + 0.08 precise + 0.04 context)` on the observation that
matches the state's truth. `src/active_inference_power/gnn_specs.py` binds
every declared matrix to the runtime array. `models/gnn-pin.json` records the
pinned Generalized_Notation_Notation release this file is checked against.

## StateSpaceBlock

s[8,1,type=float]      # hidden state: target (2) x context (2) x truth (2)
o[2,1,type=int]        # observation: o[0] = low, o[1] = high
u[6,1,type=int]        # action index: target0_cheap, target0_precise, target1_cheap,
                       # target1_precise, switch_context, stop
A[6,2,8,type=float]    # action-conditioned likelihood A[action, observation, state]
B[6,8,8,type=float]    # action-conditioned transition B[action, next_state, state]
C[2,type=float]        # flat preferences over observations
D[8,type=float]        # uniform prior over the 8 states

## Connections

D>s:prior_initialization
A>o:observation_mapping
B>s:state_transition
u>B:action_selection
u>A:sensor_selection
s>A:likelihood
s>o:emission
C>u:preference_bias

## InitialParameterization

# make_reference_action_scenario() defaults: reliability 0.8, horizon 12,
# alpha 0.05. All four arrays are deterministic, none is fitted.
# A: one block per action, holding the (low, high) emission rows over the
#    eight states (target-major, then context, then truth).
# B: one block per action, one row per next state over the eight current
#    states; every column is one-hot.
# C: flat preferences. D: uniform prior over the eight states.

A={
  (
    (0.8, 0.2, 0.84, 0.16, 0.8, 0.2, 0.84, 0.16),
    (0.2, 0.8, 0.16, 0.84, 0.2, 0.8, 0.16, 0.84)
  ),
  (
    (0.88, 0.12, 0.92, 0.08, 0.88, 0.12, 0.92, 0.08),
    (0.12, 0.88, 0.08, 0.92, 0.12, 0.88, 0.08, 0.92)
  ),
  (
    (0.8, 0.2, 0.84, 0.16, 0.8, 0.2, 0.84, 0.16),
    (0.2, 0.8, 0.16, 0.84, 0.2, 0.8, 0.16, 0.84)
  ),
  (
    (0.88, 0.12, 0.92, 0.08, 0.88, 0.12, 0.92, 0.08),
    (0.12, 0.88, 0.08, 0.92, 0.12, 0.88, 0.08, 0.92)
  ),
  (
    (0.8, 0.2, 0.84, 0.16, 0.8, 0.2, 0.84, 0.16),
    (0.2, 0.8, 0.16, 0.84, 0.2, 0.8, 0.16, 0.84)
  ),
  (
    (0.8, 0.2, 0.84, 0.16, 0.8, 0.2, 0.84, 0.16),
    (0.2, 0.8, 0.16, 0.84, 0.2, 0.8, 0.16, 0.84)
  )
}

B={
  (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 0, 0, 1, 1, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 1, 0, 0, 1, 1),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0)
  ),
  (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 0, 0, 1, 1, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 1, 0, 0, 1, 1),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0)
  ),
  (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 0, 0, 1, 1, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 1, 0, 0, 1, 1),
    (0, 0, 0, 0, 0, 0, 0, 0)
  ),
  (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 0, 0, 1, 1, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 1, 0, 0, 1, 1),
    (0, 0, 0, 0, 0, 0, 0, 0)
  ),
  (
    (0, 0, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0, 0),
    (1, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 1),
    (0, 0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 0, 0)
  ),
  (
    (1, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0, 0),
    (0, 0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 0, 0),
    (0, 0, 0, 0, 0, 0, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 1)
  )
}

C={(0.0, 0.0)}

D={(0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125)}

## Time

Dynamic

## ActInfOntologyAnnotation

s=HiddenState
o=Observation
u=Action
A=LikelihoodMatrix
B=TransitionMatrix
C=Preferences
D=Prior

## ModelParameters

a_tensor_order: action_observation_state
b_tensor_order: action_next_state_state
num_hidden_states: 8
num_obs: 2
num_actions: 6
num_targets: 2
num_contexts: 2

## Footer

Two-target binary-truth action-loop reference model — GNN representation.
