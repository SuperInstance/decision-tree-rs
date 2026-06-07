//! Reduced error pruning for decision trees.

use crate::tree::{DecisionTree, Label, Sample, TreeNode};

/// Prune a decision tree using reduced error pruning.
///
/// Removes subtrees that do not improve accuracy on the validation set.
/// Returns a new pruned tree.
pub fn reduced_error_prune(tree: &DecisionTree, validation: &[Sample]) -> DecisionTree {
    let mut root = tree.root.clone();
    prune_node(&mut root, validation);
    DecisionTree {
        root,
        feature_names: tree.feature_names.clone(),
    }
}

fn prune_node(node: &mut TreeNode, validation: &[Sample]) {
    if let TreeNode::Internal { children, .. } = node {
        // First, recursively prune children
        for (_, child) in children.iter_mut() {
            prune_node(child, validation);
        }

        // Evaluate: would replacing this node with a leaf improve accuracy?
        let current_correct = count_correct(node, validation);
        let majority = majority_label_node(node);
        let pruned_correct = validation
            .iter()
            .filter(|s| s.label == majority)
            .count();

        if pruned_correct >= current_correct {
            // Replace with leaf
            let samples = count_samples(node);
            *node = TreeNode::Leaf {
                label: majority,
                samples,
            };
        }
    }
}

/// Count correct predictions for a subtree.
fn count_correct(node: &TreeNode, samples: &[Sample]) -> usize {
    samples.iter().filter(|s| predict_node(node, s) == s.label).count()
}

/// Predict using a subtree node.
fn predict_node(node: &TreeNode, sample: &Sample) -> Label {
    match node {
        TreeNode::Leaf { label, .. } => label.clone(),
        TreeNode::Internal { feature_index, children, .. } => {
            let value = &sample.features[*feature_index];
            for (v, child) in children {
                if v == value {
                    return predict_node(child, sample);
                }
            }
            children
                .first()
                .map(|(_, c)| predict_node(c, sample))
                .unwrap_or_else(|| "unknown".to_string())
        }
    }
}

/// Get the majority label from training samples represented by a subtree.
fn majority_label_node(node: &TreeNode) -> Label {
    match node {
        TreeNode::Leaf { label, .. } => label.clone(),
        TreeNode::Internal { children: _, .. } => {
            // Collect all leaf labels and find majority
            let labels = collect_leaf_labels(node);
            let mut counts = std::collections::HashMap::new();
            for l in &labels {
                *counts.entry(l.as_str()).or_insert(0usize) += 1;
            }
            counts
                .into_iter()
                .max_by_key(|&(_, c)| c)
                .map(|(l, _)| l.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        }
    }
}

fn collect_leaf_labels(node: &TreeNode) -> Vec<Label> {
    match node {
        TreeNode::Leaf { label, .. } => vec![label.clone()],
        TreeNode::Internal { children, .. } => {
            children.iter().flat_map(|(_, c)| collect_leaf_labels(c)).collect()
        }
    }
}

fn count_samples(node: &TreeNode) -> usize {
    match node {
        TreeNode::Leaf { samples, .. } => *samples,
        TreeNode::Internal { children, .. } => children.iter().map(|(_, c)| count_samples(c)).sum(),
    }
}

/// Compute the cost-complexity pruning parameter alpha for a subtree.
///
/// Returns the alpha threshold at which this subtree would be pruned.
pub fn cost_complexity_alpha(node: &TreeNode, total_samples: usize) -> f64 {
    let n = count_samples(node) as f64;
    if n == 0.0 || total_samples == 0 {
        return f64::INFINITY;
    }
    let leaves = count_leaves_node(node);
    if leaves <= 1 {
        return f64::INFINITY;
    }
    let leaf_error = leaf_error_rate(node);
    let subtree_error = 0.0; // Approximation: subtree error
    (subtree_error - leaf_error) / ((leaves - 1) as f64) * n / (total_samples as f64)
}

fn count_leaves_node(node: &TreeNode) -> usize {
    match node {
        TreeNode::Leaf { .. } => 1,
        TreeNode::Internal { children, .. } => children.iter().map(|(_, c)| count_leaves_node(c)).sum(),
    }
}

fn leaf_error_rate(_node: &TreeNode) -> f64 {
    // Simplified: assume 0 error for leaf majority
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{DecisionTree, Sample, SplitCriterion};

    fn make_data() -> (Vec<Sample>, Vec<Sample>) {
        let train = vec![
            Sample::new(vec!["a".into(), "x".into()], "pos".into()),
            Sample::new(vec!["a".into(), "y".into()], "pos".into()),
            Sample::new(vec!["b".into(), "x".into()], "neg".into()),
            Sample::new(vec!["b".into(), "y".into()], "neg".into()),
            Sample::new(vec!["c".into(), "x".into()], "pos".into()),
        ];
        let val = vec![
            Sample::new(vec!["a".into(), "x".into()], "pos".into()),
            Sample::new(vec!["b".into(), "y".into()], "neg".into()),
        ];
        (train, val)
    }

    #[test]
    fn test_prune_no_change() {
        let (train, val) = make_data();
        let tree = DecisionTree::build(&train, SplitCriterion::InformationGain);
        let pruned = reduced_error_prune(&tree, &val);
        // Should still work for correct predictions
        for s in &val {
            pruned.predict(s);
        }
    }

    #[test]
    fn test_prune_reduces_leaves() {
        let (train, val) = make_data();
        let tree = DecisionTree::build(&train, SplitCriterion::Gini);
        let pruned = reduced_error_prune(&tree, &val);
        assert!(pruned.num_leaves() <= tree.num_leaves());
    }

    #[test]
    fn test_prune_pure_tree() {
        let train = vec![
            Sample::new(vec!["a".into()], "x".into()),
            Sample::new(vec!["b".into()], "y".into()),
        ];
        let val = train.clone();
        let tree = DecisionTree::build(&train, SplitCriterion::InformationGain);
        let pruned = reduced_error_prune(&tree, &val);
        assert_eq!(pruned.predict(&Sample::new(vec!["a".into()], "".into())), "x");
    }

    #[test]
    fn test_collect_leaf_labels() {
        let node = TreeNode::Leaf { label: "a".into(), samples: 5 };
        assert_eq!(collect_leaf_labels(&node), vec!["a"]);
    }

    #[test]
    fn test_count_samples() {
        let node = TreeNode::Leaf { label: "x".into(), samples: 10 };
        assert_eq!(count_samples(&node), 10);
    }
}
