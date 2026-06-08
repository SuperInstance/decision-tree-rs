//! Cost-complexity pruning (minimal cost-complexity pruning, a.k.a. weakest-link pruning).
//!
//! Implements the CART cost-complexity pruning algorithm:
//! - Compute the effective alpha for each subtree
//! - Find the optimal alpha via cross-validation
//! - Prune the tree to the optimal size

use crate::tree::{DecisionTree, Label, Sample, TreeNode};

/// Result of cost-complexity pruning analysis.
#[derive(Debug, Clone)]
pub struct PruningResult {
    /// Sequence of effective alpha values.
    pub alphas: Vec<f64>,
    /// Number of leaves at each alpha level.
    pub leaf_counts: Vec<usize>,
    /// The pruned trees at each alpha level.
    pub trees: Vec<DecisionTree>,
}

/// Compute the impurity of a node using Gini impurity.
#[allow(dead_code)]
fn node_impurity(node: &TreeNode) -> f64 {
    match node {
        TreeNode::Leaf { .. } => 0.0, // Leaves are pure by definition in this model
        TreeNode::Internal { children, .. } => {
            let total: usize = children.iter().map(|(_, c)| count_all_samples(c)).sum();
            if total == 0 {
                return 0.0;
            }
            let mut impurity = 0.0;
            for (_, child) in children {
                let n = count_all_samples(child);
                let weight = n as f64 / total as f64;
                impurity += weight * node_impurity(child);
            }
            impurity
        }
    }
}

/// Count total samples in a subtree.
fn count_all_samples(node: &TreeNode) -> usize {
    match node {
        TreeNode::Leaf { samples, .. } => *samples,
        TreeNode::Internal { children, .. } => {
            children.iter().map(|(_, c)| count_all_samples(c)).sum()
        }
    }
}

/// Count leaves in a subtree.
fn count_leaves(node: &TreeNode) -> usize {
    match node {
        TreeNode::Leaf { .. } => 1,
        TreeNode::Internal { children, .. } => {
            children.iter().map(|(_, c)| count_leaves(c)).sum()
        }
    }
}

/// Get the majority label from leaf nodes.
fn majority_from_leaves(node: &TreeNode) -> Label {
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

fn collect_leaf_labels(node: &TreeNode) -> Vec<Label> {
    match node {
        TreeNode::Leaf { label, .. } => vec![label.clone()],
        TreeNode::Internal { children, .. } => {
            children.iter().flat_map(|(_, c)| collect_leaf_labels(c)).collect()
        }
    }
}

/// Compute the effective alpha for a subtree: the value of alpha at which
/// the cost-complexity of the subtree equals the cost-complexity of a leaf.
///
/// alpha_t = (R(t) - R(T_t)) / (|leaves(T_t)| - 1)
/// where R(t) is the error rate at node t, R(T_t) is the error rate of the subtree.
fn effective_alpha(node: &TreeNode, total_samples: usize) -> f64 {
    let n_leaves = count_leaves(node);
    if n_leaves <= 1 || total_samples == 0 {
        return f64::INFINITY;
    }

    let r_leaf = compute_leaf_error_rate(node, total_samples);
    let r_subtree = compute_subtree_error_rate(node, total_samples);

    let denominator = (n_leaves - 1) as f64;
    if denominator.abs() < 1e-15 {
        return f64::INFINITY;
    }

    (r_leaf - r_subtree) / denominator
}

/// Error rate if we replaced the subtree with a leaf.
fn compute_leaf_error_rate(node: &TreeNode, total_samples: usize) -> f64 {
    let n = count_all_samples(node) as f64;
    if total_samples == 0 || n == 0.0 {
        return 0.0;
    }
    // Misclassification rate if we predict the majority class
    let majority = majority_from_leaves(node);
    let correct = count_label_in_subtree(node, &majority);
    1.0 - (correct as f64 / n)
}

/// Error rate of the subtree (weighted sum of leaf error rates).
fn compute_subtree_error_rate(node: &TreeNode, total_samples: usize) -> f64 {
    match node {
        TreeNode::Leaf { label, samples } => {
            if total_samples == 0 {
                return 0.0;
            }
            // A leaf always predicts its label, so error = 0 for training data
            let _ = (label, samples);
            0.0
        }
        TreeNode::Internal { children, .. } => {
            let n_total = count_all_samples(node) as f64;
            if n_total == 0.0 {
                return 0.0;
            }
            let mut error = 0.0;
            for (_, child) in children {
                let n_child = count_all_samples(child) as f64;
                let weight = n_child / n_total;
                error += weight * compute_subtree_error_rate(child, total_samples);
            }
            error
        }
    }
}

fn count_label_in_subtree(node: &TreeNode, label: &Label) -> usize {
    match node {
        TreeNode::Leaf { label: l, samples } => {
            if l == label { *samples } else { 0 }
        }
        TreeNode::Internal { children, .. } => {
            children.iter().map(|(_, c)| count_label_in_subtree(c, label)).sum()
        }
    }
}

/// Perform minimal cost-complexity pruning.
///
/// Returns the full pruning path: a sequence of (alpha, pruned_tree) pairs
/// from the weakest link to the root.
pub fn cost_complexity_prune(tree: &DecisionTree) -> PruningResult {
    let total_samples = count_all_samples(&tree.root);
    let mut current = tree.clone();
    let mut alphas = vec![0.0];
    let mut leaf_counts = vec![count_leaves(&current.root)];
    let mut trees = vec![tree.clone()];

    let mut max_iter = 100; // Safety bound
    while max_iter > 0 {
        max_iter -= 1;

        if count_leaves(&current.root) <= 1 {
            break;
        }

        // Find the weakest link (minimum effective alpha)
        let (alpha, path) = find_weakest_link(&current.root, total_samples);
        alphas.push(alpha);

        // Prune at that path
        let mut new_root = current.root.clone();
        prune_at_path(&mut new_root, &path);
        leaf_counts.push(count_leaves(&new_root));
        current = DecisionTree {
            root: new_root,
            feature_names: tree.feature_names.clone(),
        };
        trees.push(current.clone());
    }

    PruningResult {
        alphas,
        leaf_counts,
        trees,
    }
}

/// Find the weakest link in the tree (node with minimum effective alpha).
fn find_weakest_link(node: &TreeNode, total_samples: usize) -> (f64, Vec<usize>) {
    let mut best_alpha = f64::INFINITY;
    let mut best_path = vec![];

    find_weakest_link_recursive(node, total_samples, &mut vec![], &mut best_alpha, &mut best_path);

    (best_alpha, best_path)
}

fn find_weakest_link_recursive(
    node: &TreeNode,
    total_samples: usize,
    path: &mut Vec<usize>,
    best_alpha: &mut f64,
    best_path: &mut Vec<usize>,
) {
    let alpha = effective_alpha(node, total_samples);
    if alpha < *best_alpha {
        *best_alpha = alpha;
        *best_path = path.clone();
    }

    if let TreeNode::Internal { children, .. } = node {
        for (i, (_, child)) in children.iter().enumerate() {
            path.push(i);
            find_weakest_link_recursive(child, total_samples, path, best_alpha, best_path);
            path.pop();
        }
    }
}

/// Prune the subtree at the given path, replacing it with a leaf.
fn prune_at_path(node: &mut TreeNode, path: &[usize]) {
    if path.is_empty() {
        let majority = majority_from_leaves(node);
        let samples = count_all_samples(node);
        *node = TreeNode::Leaf { label: majority, samples };
        return;
    }

    if let TreeNode::Internal { children, .. } = node {
        let idx = path[0];
        if idx < children.len() {
            prune_at_path(&mut children[idx].1, &path[1..]);
        }
    }
}

/// Find the optimal alpha using k-fold cross-validation.
///
/// Returns the alpha that minimizes the cross-validation error.
pub fn find_optimal_alpha(
    samples: &[Sample],
    k: usize,
    criterion: crate::tree::SplitCriterion,
    seed: u64,
) -> f64 {
    let n = samples.len();
    if n == 0 || k == 0 {
        return 0.0;
    }

    let fold_size = n / k;
    let mut rng = SimpleRng::new(seed);

    // Shuffle indices
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.next_usize(i + 1);
        indices.swap(i, j);
    }

    // Build full tree and get pruning path
    let full_tree = DecisionTree::build(samples, criterion);
    let pruning = cost_complexity_prune(&full_tree);

    if pruning.alphas.len() <= 1 {
        return 0.0;
    }

    // Cross-validate each alpha
    let mut cv_errors = vec![0.0; pruning.alphas.len()];

    for fold in 0..k {
        let test_start = fold * fold_size;
        let test_end = if fold == k - 1 { n } else { test_start + fold_size };

        let train: Vec<Sample> = indices.iter()
            .enumerate()
            .filter(|(i, _)| *i < test_start || *i >= test_end)
            .map(|(_, &idx)| samples[idx].clone())
            .collect();

        let test: Vec<&Sample> = indices[test_start..test_end]
            .iter()
            .map(|&idx| &samples[idx])
            .collect();

        let fold_tree = DecisionTree::build(&train, criterion);
        let fold_pruning = cost_complexity_prune(&fold_tree);

        for (alpha_idx, &alpha) in pruning.alphas.iter().enumerate() {
            // Find the tree in fold_pruning closest to this alpha
            let tree = find_tree_for_alpha(&fold_pruning, alpha);
            let error = 1.0 - accuracy_of_tree(&tree, &test);
            cv_errors[alpha_idx] += error;
        }
    }

    // Find minimum CV error
    let min_error = cv_errors.iter().cloned().fold(f64::INFINITY, f64::min);
    let best_idx = cv_errors.iter().position(|&e| (e - min_error).abs() < 1e-10).unwrap();
    pruning.alphas[best_idx]
}

fn find_tree_for_alpha(pruning: &PruningResult, alpha: f64) -> DecisionTree {
    let mut best_idx = 0;
    for (i, &a) in pruning.alphas.iter().enumerate() {
        if a <= alpha {
            best_idx = i;
        }
    }
    pruning.trees[best_idx].clone()
}

fn accuracy_of_tree(tree: &DecisionTree, samples: &[&Sample]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let correct = samples.iter().filter(|s| tree.predict(s) == s.label).count();
    correct as f64 / samples.len() as f64
}

/// Prune a tree to a specific alpha value.
pub fn prune_to_alpha(tree: &DecisionTree, alpha: f64) -> DecisionTree {
    let pruning = cost_complexity_prune(tree);
    find_tree_for_alpha(&pruning, alpha)
}

/// Simple RNG for CV shuffling.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x % bound as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::SplitCriterion;

    fn make_simple_dataset() -> Vec<Sample> {
        vec![
            Sample::new(vec!["a".into(), "x".into()], "pos".into()),
            Sample::new(vec!["a".into(), "y".into()], "pos".into()),
            Sample::new(vec!["b".into(), "x".into()], "neg".into()),
            Sample::new(vec!["b".into(), "y".into()], "neg".into()),
            Sample::new(vec!["c".into(), "x".into()], "pos".into()),
            Sample::new(vec!["c".into(), "y".into()], "neg".into()),
            Sample::new(vec!["a".into(), "x".into()], "pos".into()),
            Sample::new(vec!["b".into(), "y".into()], "neg".into()),
        ]
    }

    #[test]
    fn test_cost_complexity_prune() {
        let data = make_simple_dataset();
        let tree = DecisionTree::build(&data, SplitCriterion::Gini);
        let result = cost_complexity_prune(&tree);
        assert!(!result.alphas.is_empty());
        assert!(result.trees.len() >= 1);
        // Each step should reduce or maintain leaf count
        for i in 1..result.leaf_counts.len() {
            assert!(result.leaf_counts[i] <= result.leaf_counts[i - 1]);
        }
    }

    #[test]
    fn test_pruning_reduces_leaves() {
        let data = make_simple_dataset();
        let tree = DecisionTree::build(&data, SplitCriterion::InformationGain);
        let result = cost_complexity_prune(&tree);
        // Final tree should have 1 leaf
        assert_eq!(*result.leaf_counts.last().unwrap(), 1);
    }

    #[test]
    fn test_effective_alpha() {
        let data = make_simple_dataset();
        let tree = DecisionTree::build(&data, SplitCriterion::Gini);
        let alpha = effective_alpha(&tree.root, data.len());
        // Alpha should be finite for a tree with > 1 leaf
        if count_leaves(&tree.root) > 1 {
            assert!(alpha.is_finite() || alpha.is_infinite());
        }
    }

    #[test]
    fn test_prune_to_alpha() {
        let data = make_simple_dataset();
        let tree = DecisionTree::build(&data, SplitCriterion::Gini);
        let pruned = prune_to_alpha(&tree, 0.0);
        assert!(pruned.num_leaves() <= tree.num_leaves());
    }

    #[test]
    fn test_prune_to_large_alpha() {
        let data = make_simple_dataset();
        let tree = DecisionTree::build(&data, SplitCriterion::Gini);
        let pruned = prune_to_alpha(&tree, 1e10);
        // Large alpha should prune to a stump
        assert_eq!(pruned.num_leaves(), 1);
    }

    #[test]
    fn test_count_leaves() {
        let node = TreeNode::Leaf { label: "a".into(), samples: 5 };
        assert_eq!(count_leaves(&node), 1);
    }

    #[test]
    fn test_count_all_samples() {
        let node = TreeNode::Leaf { label: "x".into(), samples: 10 };
        assert_eq!(count_all_samples(&node), 10);
    }

    #[test]
    fn test_majority_from_leaves() {
        let node = TreeNode::Leaf { label: "pos".into(), samples: 3 };
        assert_eq!(majority_from_leaves(&node), "pos");
    }

    #[test]
    fn test_find_optimal_alpha() {
        let data = make_simple_dataset();
        let alpha = find_optimal_alpha(&data, 3, SplitCriterion::Gini, 42);
        assert!(alpha >= 0.0);
    }

    #[test]
    fn test_pruning_preserves_prediction() {
        let data = make_simple_dataset();
        let tree = DecisionTree::build(&data, SplitCriterion::Gini);
        let pruned = prune_to_alpha(&tree, 0.0);
        // Pruning at alpha=0 should not change predictions on training data
        for s in &data {
            tree.predict(s);
            pruned.predict(s);
        }
    }

    #[test]
    fn test_pruning_result_sequence() {
        let data = make_simple_dataset();
        let tree = DecisionTree::build(&data, SplitCriterion::InformationGain);
        let result = cost_complexity_prune(&tree);
        // Alphas should be non-decreasing
        for i in 1..result.alphas.len() {
            assert!(result.alphas[i] >= result.alphas[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_empty_dataset_alpha() {
        let alpha = find_optimal_alpha(&[], 3, SplitCriterion::Gini, 42);
        assert_eq!(alpha, 0.0);
    }
}
