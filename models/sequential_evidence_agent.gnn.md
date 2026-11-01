# GNN Example: Sequential Evidence-Calibration Agent

## GNNSection

ActInfSequentialEvidenceAgent

## GNNVersionAndFlags

GNN v1.1

## ModelName

Sequential Evidence-Calibration Agent (Fixed Horizon)

## ModelAnnotation

The agent-side generative model behind `src/active_inference_power/active_inference.py`:
one binary hidden-state factor "truth" (`H0` null, `H1` signal), one binary
observation modality (`low`, `high`), and a single dummy action whose identity
transition makes the posterior at step t the prior at step t+1 — exact
sequential Bayesian evidence accumulation realized through the real
`inferactively-pymdp` agent. Preferences are flat: this is inference, not
control. The declared `InitialParameterization` is the default `AITestConfig`
(reliability 0.8, prior_signal 0.5); `build_generative_model` parameterizes the
same structure by reliability and prior signal, and the binding check in
`src/active_inference_power/gnn_specs.py` verifies the declared arrays against
the runtime model. `models/gnn-pin.json` records the pinned
Generalized_Notation_Notation release this file is checked against.

## StateSpaceBlock

s[2,1,type=float]     # hidden truth: s[0] = H0 (null), s[1] = H1 (signal)
o[2,1,type=int]       # observation: o[0] = low, o[1] = high
u[1,type=int]         # single dummy action; truth is static
A[2,2,type=float]     # likelihood P(o | s); rows = observations, columns = states
B[2,2,1,type=float]   # identity transition B[next_state, state, action]
C[2,type=float]       # flat preferences over observations (inference, not control)
D[2,type=float]       # prior over {H0, H1}
q_s[2,1,type=float]   # posterior belief in {H0, H1} after each observation

## Connections

D>s:prior_initialization
A>o:observation_mapping
B>s:state_transition
s-B:identity_transition
u>B:action_selection
s>A:likelihood
C>u:preference_bias
s>q_s:posterior_update
o>q_s:evidence_update

## InitialParameterization

A={
  (0.8, 0.2),
  (0.2, 0.8)
}

C={(0.0, 0.0)}

D={(0.5, 0.5)}

## Equations

A[o,s] = P(o | s): the columns are the observation distributions under H0 and
H1. With identity B the posterior after each observation is the prior for the
next step, q_s(t) proportional to A[o(t), :] * q_s(t-1). The agent rejects H0
iff the final signal belief reaches the declared posterior threshold. The
closed-form oracle used by the tests is the sequential likelihood-ratio
posterior odds o = o0 * LR ** n_high * LR ** (-n_low).

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
q_s=PosteriorBelief

## ModelParameters

b_tensor_order: next_state_state_action
num_hidden_states: 2
num_obs: 2
num_actions: 1
num_timesteps: 20
reliability: 0.8
prior_signal: 0.5
posterior_threshold: 0.9

## Footer

Sequential evidence-calibration agent (fixed horizon) — GNN representation.
