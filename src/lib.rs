//! # decision-tree-rs
//!
//! A pure-Rust decision tree library for classification with ID3 and CART-style algorithms.
//!
//! ## Modules
//!
//! - [`tree`] — Core decision tree data structure
//! - [`gain`] — Information gain and entropy computations
//! - [`gini`] — Gini impurity and Gini gain
//! - [`prune`] — Reduced error pruning
//! - [`predict`] — Prediction and evaluation

pub mod gain;
pub mod gini;
pub mod predict;
pub mod prune;
pub mod tree;

pub use tree::DecisionTree;
