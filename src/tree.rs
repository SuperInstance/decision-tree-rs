//! Core decision tree data structure and ID3/CART building algorithms.

use crate::gain::best_split_info_gain;
use crate::gini::best_split_gini;

/// A feature value (currently only discrete/categorical represented as strings).
pub type FeatureValue = String;

/// A class label.
pub type Label = String;

/// A single data sample: a vector of feature values.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    /// Feature values, one per feature.
    pub features: Vec<FeatureValue>,
    /// The class label.
    pub label: Label,
}

impl Sample {
    /// Create a new sample.
    pub fn new(features: Vec<FeatureValue>, label: Label) -> Self {
        Self { features, label }
    }
}

/// A decision tree node.
#[derive(Debug, Clone, PartialEq)]
pub enum TreeNode {
    /// Internal node: split on a feature.
    Internal {
        /// Index of the feature to split on.
        feature_index: usize,
        /// Feature name (optional, for display).
        feature_name: Option<String>,
        /// Children: maps feature value to child node.
        children: Vec<(FeatureValue, TreeNode)>,
    },
    /// Leaf node: predicts a class.
    Leaf {
        /// The predicted class label.
        label: Label,
        /// Number of samples at this leaf.
        samples: usize,
    },
}

/// A decision tree for classification.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionTree {
    /// The root node.
    pub root: TreeNode,
    /// Feature names (optional).
    pub feature_names: Vec<String>,
}

/// Split criterion for building the tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitCriterion {
    /// Use information gain (ID3).
    InformationGain,
    /// Use Gini impurity reduction (CART).
    Gini,
}

impl DecisionTree {
    /// Build a decision tree using the specified criterion.
    pub fn build(samples: &[Sample], criterion: SplitCriterion) -> Self {
        let num_features = samples.first().map_or(0, |s| s.features.len());
        let root = Self::build_recursive(samples, num_features, criterion, 0);
        DecisionTree {
            root,
            feature_names: vec![],
        }
    }

    /// Build with feature names.
    pub fn build_with_names(
        samples: &[Sample],
        feature_names: Vec<String>,
        criterion: SplitCriterion,
    ) -> Self {
        let num_features = samples.first().map_or(0, |s| s.features.len());
        let root = Self::build_recursive(samples, num_features, criterion, 0);
        DecisionTree { root, feature_names }
    }

    fn build_recursive(
        samples: &[Sample],
        num_features: usize,
        criterion: SplitCriterion,
        depth: usize,
    ) -> TreeNode {
        // Base case: all samples have the same label
        let labels: Vec<&Label> = samples.iter().map(|s| &s.label).collect();
        if labels.windows(2).all(|w| w[0] == w[1]) {
            return TreeNode::Leaf {
                label: labels[0].clone(),
                samples: samples.len(),
            };
        }

        // Base case: no features left or max depth
        if samples.is_empty() || num_features == 0 {
            return TreeNode::Leaf {
                label: majority_label(samples),
                samples: samples.len(),
            };
        }

        // Find best feature to split on
        let best_feature = match criterion {
            SplitCriterion::InformationGain => best_split_info_gain(samples, num_features),
            SplitCriterion::Gini => best_split_gini(samples, num_features),
        };

        let best_idx = match best_feature {
            Some(idx) => idx,
            None => {
                return TreeNode::Leaf {
                    label: majority_label(samples),
                    samples: samples.len(),
                }
            }
        };

        // Split on best feature
        let mut partitions: std::collections::HashMap<FeatureValue, Vec<&Sample>> =
            std::collections::HashMap::new();
        for s in samples {
            partitions
                .entry(s.features[best_idx].clone())
                .or_default()
                .push(s);
        }

        let mut children = vec![];
        for (value, subset) in partitions {
            if subset.is_empty() {
                children.push((value, TreeNode::Leaf {
                    label: majority_label(samples),
                    samples: 0,
                }));
            } else {
                let owned: Vec<Sample> = subset.into_iter().cloned().collect();
                children.push((value, Self::build_recursive(&owned, num_features, criterion, depth + 1)));
            }
        }

        TreeNode::Internal {
            feature_index: best_idx,
            feature_name: None,
            children,
        }
    }

    /// Predict the label for a sample.
    pub fn predict(&self, sample: &Sample) -> Label {
        Self::predict_node(&self.root, sample)
    }

    fn predict_node(node: &TreeNode, sample: &Sample) -> Label {
        match node {
            TreeNode::Leaf { label, .. } => label.clone(),
            TreeNode::Internal { feature_index, children, .. } => {
                let value = &sample.features[*feature_index];
                for (v, child) in children {
                    if v == value {
                        return Self::predict_node(child, sample);
                    }
                }
                // Value not found in children; return first child's prediction as fallback
                children
                    .first()
                    .map(|(_, c)| Self::predict_node(c, sample))
                    .unwrap_or_else(|| "unknown".to_string())
            }
        }
    }

    /// Count the number of leaves in the tree.
    pub fn num_leaves(&self) -> usize {
        count_leaves(&self.root)
    }

    /// Compute the depth of the tree.
    pub fn depth(&self) -> usize {
        node_depth(&self.root)
    }

    /// Format the tree as an indented string.
    pub fn display(&self) -> String {
        let mut s = String::new();
        display_node(&self.root, 0, &mut s, &self.feature_names);
        s
    }
}

fn majority_label(samples: &[Sample]) -> Label {
    let mut counts = std::collections::HashMap::new();
    for s in samples {
        *counts.entry(&s.label).or_insert(0usize) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map(|(l, _)| l.clone())
        .unwrap_or_else(|| "unknown".to_string())
}

fn count_leaves(node: &TreeNode) -> usize {
    match node {
        TreeNode::Leaf { .. } => 1,
        TreeNode::Internal { children, .. } => children.iter().map(|(_, c)| count_leaves(c)).sum(),
    }
}

fn node_depth(node: &TreeNode) -> usize {
    match node {
        TreeNode::Leaf { .. } => 0,
        TreeNode::Internal { children, .. } => {
            1 + children.iter().map(|(_, c)| node_depth(c)).max().unwrap_or(0)
        }
    }
}

fn display_node(node: &TreeNode, indent: usize, s: &mut String, names: &[String]) {
    let pad = "  ".repeat(indent);
    match node {
        TreeNode::Leaf { label, samples } => {
            s.push_str(&format!("{}→ {} ({} samples)\n", pad, label, samples));
        }
        TreeNode::Internal { feature_index, children, .. } => {
            let name = names.get(*feature_index)
                .cloned()
                .unwrap_or_else(|| format!("feature_{}", feature_index));
            for (value, child) in children {
                s.push_str(&format!("{}{} = {}?\n", pad, name, value));
                display_node(child, indent + 1, s, names);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dataset() -> Vec<Sample> {
        vec![
            Sample::new(vec!["sunny".into(), "hot".into(), "high".into()], "no".into()),
            Sample::new(vec!["sunny".into(), "hot".into(), "normal".into()], "no".into()),
            Sample::new(vec!["overcast".into(), "hot".into(), "high".into()], "yes".into()),
            Sample::new(vec!["rainy".into(), "mild".into(), "high".into()], "yes".into()),
            Sample::new(vec!["rainy".into(), "cool".into(), "normal".into()], "yes".into()),
            Sample::new(vec!["rainy".into(), "cool".into(), "normal".into()], "no".into()),
            Sample::new(vec!["overcast".into(), "cool".into(), "normal".into()], "yes".into()),
            Sample::new(vec!["sunny".into(), "mild".into(), "high".into()], "no".into()),
            Sample::new(vec!["sunny".into(), "cool".into(), "normal".into()], "yes".into()),
            Sample::new(vec!["rainy".into(), "mild".into(), "normal".into()], "yes".into()),
            Sample::new(vec!["sunny".into(), "mild".into(), "normal".into()], "yes".into()),
            Sample::new(vec!["overcast".into(), "mild".into(), "high".into()], "yes".into()),
            Sample::new(vec!["overcast".into(), "hot".into(), "normal".into()], "yes".into()),
            Sample::new(vec!["rainy".into(), "mild".into(), "high".into()], "no".into()),
        ]
    }

    #[test]
    fn test_build_id3() {
        let data = make_dataset();
        let tree = DecisionTree::build(&data, SplitCriterion::InformationGain);
        assert!(tree.num_leaves() > 0);
        assert!(tree.depth() > 0);
    }

    #[test]
    fn test_build_gini() {
        let data = make_dataset();
        let tree = DecisionTree::build(&data, SplitCriterion::Gini);
        assert!(tree.num_leaves() > 0);
    }

    #[test]
    fn test_predict_simple() {
        let data = vec![
            Sample::new(vec!["a".into()], "x".into()),
            Sample::new(vec!["b".into()], "y".into()),
        ];
        let tree = DecisionTree::build(&data, SplitCriterion::InformationGain);
        assert_eq!(tree.predict(&Sample::new(vec!["a".into()], "x".into())), "x");
        assert_eq!(tree.predict(&Sample::new(vec!["b".into()], "y".into())), "y");
    }

    #[test]
    fn test_pure_dataset() {
        let data = vec![
            Sample::new(vec!["a".into()], "x".into()),
            Sample::new(vec!["b".into()], "x".into()),
        ];
        let tree = DecisionTree::build(&data, SplitCriterion::InformationGain);
        assert_eq!(tree.num_leaves(), 1); // All same label → single leaf
    }

    #[test]
    fn test_weather_dataset() {
        let data = make_dataset();
        let tree = DecisionTree::build(&data, SplitCriterion::InformationGain);
        // Overcast should always predict yes
        let overcast_sample = Sample::new(vec!["overcast".into(), "hot".into(), "high".into()], "".into());
        assert_eq!(tree.predict(&overcast_sample), "yes");
    }

    #[test]
    fn test_with_feature_names() {
        let data = make_dataset();
        let tree = DecisionTree::build_with_names(
            &data,
            vec!["outlook".into(), "temperature".into(), "humidity".into()],
            SplitCriterion::Gini,
        );
        let display = tree.display();
        assert!(display.contains("outlook") || display.contains("temperature") || display.contains("humidity"));
    }

    #[test]
    fn test_depth() {
        let data = make_dataset();
        let tree = DecisionTree::build(&data, SplitCriterion::InformationGain);
        assert!(tree.depth() >= 1);
        assert!(tree.depth() <= 3); // Weather dataset shouldn't be too deep
    }

    #[test]
    fn test_empty_features() {
        // Single feature, binary split
        let data = vec![
            Sample::new(vec!["yes".into()], "pos".into()),
            Sample::new(vec!["no".into()], "neg".into()),
        ];
        let tree = DecisionTree::build(&data, SplitCriterion::InformationGain);
        assert_eq!(tree.predict(&Sample::new(vec!["yes".into()], "".into())), "pos");
    }
}
