/**
 * Static example payloads for UI documentation / Storybook only.
 * Do not import from `Dashboard.tsx` or any routed screen — the live app
 * must only render hypotheses, models, and citations from the API.
 */
import type { Citation, Hypothesis, ModelResult } from "@/types";

export const EXAMPLE_HYPOTHESES: Hypothesis[] = [
  {
    id: "example-h1",
    hypothesis_id: "H-01",
    round: 1,
    description:
      "Surfactant-assisted plasma shifts PFCA degradation away from persistent intermediate accumulation and toward short-chain-dominated pathway states.",
    rationale:
      "The hypothesis follows from pathway metrics where entropy collapses after the intermediate-rich stage while fluoride yield increases.",
    primary_variables: ["surfactant", "time", "C2 fraction", "Shannon entropy"],
    model_family: "panel_fixed_effects",
    priority_score: 0.91,
    is_refinement: false,
  },
  {
    id: "example-h2",
    hypothesis_id: "H-02",
    round: 1,
    description:
      "Gas atmosphere contributes an independent secondary axis of PFBS degradation performance after solution chemistry effects are accounted for.",
    rationale:
      "PCA separates gas type from bulk-solution properties; regression tests whether that axis is predictive after time effects are separated.",
    primary_variables: ["gas_used", "conductivity", "initial PFBS", "time"],
    model_family: "panel_fixed_effects",
    priority_score: 0.84,
    is_refinement: false,
  },
  {
    id: "example-h3",
    hypothesis_id: "H-03",
    round: 2,
    description:
      "Initial chain-length composition controls maximum fluoride yield in PFCA mixtures, while surfactant addition modifies degradation kinetics.",
    rationale:
      "Mixture PCA and kinetic overlays show different drivers for defluorination and total PFCA disappearance.",
    primary_variables: ["long_chain_fraction", "short_chain_fraction", "surfactant", "k_total"],
    model_family: "panel_fixed_effects",
    priority_score: 0.79,
    is_refinement: true,
  },
];

export const EXAMPLE_MODELS: ModelResult[] = [
  {
    id: "example-m1",
    hypothesis_id: "H-01",
    model_type: "fixed_effect_panel",
    r_squared: 0.835,
    adj_r_squared: 0.812,
    n_observations: 186,
    coefficients: {
      time: 0.74,
      surfactant: 0.42,
      h2o2: 0.31,
      "time × surfactant": 0.18,
      "uv × h2o2": -0.12,
    },
    p_values: {
      time: 0.0001,
      surfactant: 0.008,
      h2o2: 0.022,
      "time × surfactant": 0.046,
      "uv × h2o2": 0.19,
    },
    significant_variables: ["time", "surfactant", "h2o2", "time × surfactant"],
    match_score: 0.88,
    validation_passed: true,
  },
  {
    id: "example-m2",
    hypothesis_id: "H-02",
    model_type: "fixed_effect_panel",
    r_squared: 0.731,
    adj_r_squared: 0.697,
    n_observations: 92,
    coefficients: {
      time: 0.61,
      gas_used: 0.39,
      conductivity: 0.28,
      additive: -0.16,
      polarity: -0.08,
    },
    p_values: {
      time: 0.0003,
      gas_used: 0.013,
      conductivity: 0.037,
      additive: 0.21,
      polarity: 0.48,
    },
    significant_variables: ["time", "gas_used", "conductivity"],
    match_score: 0.81,
    validation_passed: true,
  },
  {
    id: "example-m3",
    hypothesis_id: "H-03",
    model_type: "fixed_effect_panel",
    r_squared: 0.786,
    adj_r_squared: 0.761,
    n_observations: 144,
    coefficients: {
      short_chain_fraction: 0.58,
      surfactant: 0.36,
      long_chain_fraction: -0.29,
      "surfactant × composition": 0.21,
    },
    p_values: {
      short_chain_fraction: 0.004,
      surfactant: 0.018,
      long_chain_fraction: 0.041,
      "surfactant × composition": 0.066,
    },
    significant_variables: ["short_chain_fraction", "surfactant", "long_chain_fraction"],
    match_score: 0.83,
    validation_passed: true,
  },
];

export const EXAMPLE_CITATIONS: Citation[] = [
  {
    id: "example-c1",
    source: "corpus",
    title:
      "Development and Application of Different Non-thermal Plasma Reactors for the Removal of Perfluorosurfactants in Water",
    url: null,
    year: "2019",
    similarity_score: 0.92,
    variable: "gas_used",
  },
  {
    id: "example-c2",
    source: "arxiv",
    title:
      "Mechanistic Modeling of Plasma-Induced PFAS Mineralization and Intermediate Pathway Evolution",
    url: null,
    year: "2024",
    similarity_score: 0.87,
    variable: "intermediate pathway",
  },
  {
    id: "example-c3",
    source: "semantic_scholar",
    title:
      "Degradation of Emerging Per- and Polyfluoroalkyl Substances Using an Electrochemical Plug Flow Reactor",
    url: null,
    year: "2023",
    similarity_score: 0.84,
    variable: "time",
  },
  {
    id: "example-c4",
    source: "semantic_scholar",
    title:
      "Incinerability of PFOA and HFPO-DA: Mechanisms, Kinetics, and Thermal Stability Ranking",
    url: null,
    year: "2023",
    similarity_score: 0.78,
    variable: "radical pathway",
  },
];
