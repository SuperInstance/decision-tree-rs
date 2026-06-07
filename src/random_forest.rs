//! Random forest ensemble classifier.
//!
//! Implements bootstrap aggregating (bagging), feature bagging, majority vote
//! prediction, and out-of-bag (OOB) error estimation using pure std.

use crate::tree::{DecisionTree, Label, Sample, SplitCriterion};

/// A random forest classifier.
#[derive(Debug, Clone)]
pub struct RandomForest {
    /// The decision trees in the ensemble.
    pub trees: Vec<DecisionTree>,
    /// Number of features to consider at each split.
    pub max_features: usize,
    /// OOB sample indices for each tree.
    oob_indices: Vec<Vec<usize>>,
}

/// Configuration for building a random forest.
#[derive(Debug, Clone)]
pub struct RandomForestConfig {
    /// Number of trees in the forest.
    pub n_trees: usize,
    /// Number of features to consider at each split (0 = sqrt of total features).
    pub max_features: usize,
    /// Minimum samples required to split.
    pub min_samples_split: usize,
    /// Maximum depth of each tree (0 = unlimited).
    pub max_depth: usize,
    /// Whether to use feature bagging.
    pub feature_bagging: bool,
    /// Split criterion.
    pub criterion: SplitCriterion,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for RandomForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 10,
            max_features: 0,
            min_samples_split: 2,
            max_depth: 0,
            feature_bagging: true,
            criterion: SplitCriterion::Gini,
            seed: 42,
        }
    }
}

impl RandomForestConfig {
    /// Create a new config with specified number of trees.
    pub fn new(n_trees: usize) -> Self {
        Self {
            n_trees,
            ..Default::default()
        }
    }
}

/// Simple pseudo-random number generator (xorshift64).
#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// Generate a random f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        self.next_u64() as f64 / u64::MAX as f64
    }
}

impl RandomForest {
    /// Build a random forest from training data.
    pub fn build(samples: &[Sample], config: &RandomForestConfig) -> Self {
        let n_samples = samples.len();
        let n_features = samples.first().map_or(0, |s| s.features.len());
        let max_features = if config.max_features == 0 {
            (n_features as f64).sqrt().ceil() as usize
        } else {
            config.max_features.min(n_features)
        };

        let mut rng = Rng::new(config.seed);
        let mut trees = Vec::with_capacity(config.n_trees);
        let mut oob_indices = Vec::with_capacity(config.n_trees);

        for _ in 0..config.n_trees {
            // Bootstrap sample
            let mut bootstrap_indices = Vec::with_capacity(n_samples);
            let mut in_bag = vec![false; n_samples];

            for _ in 0..n_samples {
                let idx = rng.next_usize(n_samples);
                bootstrap_indices.push(idx);
                in_bag[idx] = true;
            }

            // OOB indices
            let oob: Vec<usize> = (0..n_samples)
                .filter(|&i| !in_bag[i])
                .collect();

            // Create bootstrap dataset
            let bootstrap_samples: Vec<Sample> = bootstrap_indices
                .iter()
                .map(|&i| samples[i].clone())
                .collect();

            // Build tree (with optional feature bagging via subsampling features)
            let tree = if config.feature_bagging && max_features < n_features {
                let selected_features = select_features(&mut rng, n_features, max_features);
                let subsampled = subsample_features(&bootstrap_samples, &selected_features);
                DecisionTree::build(&subsampled, config.criterion)
            } else {
                DecisionTree::build(&bootstrap_samples, config.criterion)
            };

            trees.push(tree);
            oob_indices.push(oob);
        }

        RandomForest {
            trees,
            max_features,
            oob_indices,
        }
    }

    /// Predict the label for a sample using majority vote.
    pub fn predict(&self, sample: &Sample) -> Label {
        let mut votes = std::collections::HashMap::new();
        for tree in &self.trees {
            let label = tree.predict(sample);
            *votes.entry(label).or_insert(0usize) += 1;
        }
        votes
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label)
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Predict with confidence (vote proportions).
    pub fn predict_with_confidence(&self, sample: &Sample) -> (Label, f64) {
        let mut votes = std::collections::HashMap::new();
        for tree in &self.trees {
            let label = tree.predict(sample);
            *votes.entry(label).or_insert(0usize) += 1;
        }
        let total = self.trees.len();
        let (best_label, best_count) = votes
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .unwrap_or_else(|| ("unknown".to_string(), 0));
        (best_label, best_count as f64 / total as f64)
    }

    /// Compute the out-of-bag (OOB) error estimate.
    pub fn oob_error(&self, samples: &[Sample]) -> f64 {
        let mut errors = 0usize;
        let mut total = 0usize;

        for (i, sample) in samples.iter().enumerate() {
            // Find trees where this sample is OOB
            let oob_trees: Vec<&DecisionTree> = self.trees
                .iter()
                .zip(self.oob_indices.iter())
                .filter(|(_, oob)| oob.contains(&i))
                .map(|(tree, _)| tree)
                .collect();

            if oob_trees.is_empty() {
                continue;
            }

            // Majority vote from OOB trees
            let mut votes = std::collections::HashMap::new();
            for tree in &oob_trees {
                let label = tree.predict(sample);
                *votes.entry(label).or_insert(0usize) += 1;
            }
            let predicted = votes
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(label, _)| label)
                .unwrap_or_else(|| "unknown".to_string());

            if predicted != sample.label {
                errors += 1;
            }
            total += 1;
        }

        if total == 0 { 0.0 } else { errors as f64 / total as f64 }
    }

    /// Compute accuracy on a test set.
    pub fn accuracy(&self, samples: &[Sample]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let correct = samples.iter()
            .filter(|s| self.predict(s) == s.label)
            .count();
        correct as f64 / samples.len() as f64
    }

    /// Get the number of trees.
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Compute feature importance (simplified: based on how often features are used at the root).
    pub fn feature_importance(&self, n_features: usize) -> Vec<f64> {
        let mut importance = vec![0.0; n_features];

        for tree in &self.trees {
            collect_feature_importance(&tree.root, &mut importance);
        }

        // Normalize
        let total: f64 = importance.iter().sum();
        if total > 0.0 {
            for v in &mut importance {
                *v /= total;
            }
        }

        importance
    }
}

fn collect_feature_importance(node: &crate::tree::TreeNode, importance: &mut [f64]) {
    if let crate::tree::TreeNode::Internal { feature_index, children, .. } = node {
        if *feature_index < importance.len() {
            importance[*feature_index] += 1.0;
        }
        for (_, child) in children {
            collect_feature_importance(child, importance);
        }
    }
}

/// Select a random subset of feature indices.
fn select_features(rng: &mut Rng, n_features: usize, max_features: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n_features).collect();
    // Fisher-Yates shuffle (partial)
    for i in 0..max_features.min(n_features) {
        let j = i + rng.next_usize(n_features - i);
        indices.swap(i, j);
    }
    indices[..max_features.min(n_features)].to_vec()
}

/// Create samples with only the selected features.
fn subsample_features(samples: &[Sample], features: &[usize]) -> Vec<Sample> {
    samples.iter().map(|s| {
        let new_features: Vec<String> = features.iter().map(|&i| s.features[i].clone()).collect();
        Sample::new(new_features, s.label.clone())
    }).collect()
}

/// Generate a bootstrap sample (with replacement).
pub fn bootstrap_sample(samples: &[Sample], seed: u64) -> (Vec<Sample>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let n = samples.len();
    let indices: Vec<usize> = (0..n).map(|_| rng.next_usize(n)).collect();
    let bootstrap: Vec<Sample> = indices.iter().map(|&i| samples[i].clone()).collect();
    (bootstrap, indices)
}

/// Compute the bootstrap aggregation prediction for a single sample.
pub fn bagging_predict(trees: &[DecisionTree], sample: &Sample) -> Label {
    let mut votes = std::collections::HashMap::new();
    for tree in trees {
        *votes.entry(tree.predict(sample)).or_insert(0usize) += 1;
    }
    votes.into_iter().max_by_key(|&(_, c)| c).map(|(l, _)| l)
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_xor_dataset() -> Vec<Sample> {
        vec![
            Sample::new(vec!["0".into(), "0".into()], "0".into()),
            Sample::new(vec!["0".into(), "1".into()], "1".into()),
            Sample::new(vec!["1".into(), "0".into()], "1".into()),
            Sample::new(vec!["1".into(), "1".into()], "0".into()),
        ]
    }

    fn make_large_dataset() -> Vec<Sample> {
        let mut data = vec![];
        // Create a larger dataset by repeating with noise
        for i in 0..40 {
            let x = if i % 2 == 0 { "a" } else { "b" };
            let y = if i % 4 < 2 { "x" } else { "y" };
            let label = if i % 2 == 0 { "pos" } else { "neg" };
            data.push(Sample::new(vec![x.into(), y.into()], label.into()));
        }
        data
    }

    #[test]
    fn test_random_forest_build() {
        let data = make_large_dataset();
        let config = RandomForestConfig::new(5);
        let forest = RandomForest::build(&data, &config);
        assert_eq!(forest.n_trees(), 5);
    }

    #[test]
    fn test_random_forest_predict() {
        let data = make_large_dataset();
        let config = RandomForestConfig::new(5);
        let forest = RandomForest::build(&data, &config);
        let sample = Sample::new(vec!["a".into(), "x".into()], "".into());
        let pred = forest.predict(&sample);
        assert!(!pred.is_empty());
    }

    #[test]
    fn test_random_forest_accuracy() {
        let data = make_large_dataset();
        let config = RandomForestConfig::new(5);
        let forest = RandomForest::build(&data, &config);
        let acc = forest.accuracy(&data);
        assert!(acc > 0.0);
    }

    #[test]
    fn test_random_forest_oob_error() {
        let data = make_large_dataset();
        let config = RandomForestConfig {
            n_trees: 10,
            seed: 42,
            ..Default::default()
        };
        let forest = RandomForest::build(&data, &config);
        let oob_err = forest.oob_error(&data);
        assert!(oob_err >= 0.0 && oob_err <= 1.0);
    }

    #[test]
    fn test_predict_with_confidence() {
        let data = make_large_dataset();
        let config = RandomForestConfig::new(5);
        let forest = RandomForest::build(&data, &config);
        let sample = Sample::new(vec!["a".into(), "x".into()], "".into());
        let (label, conf) = forest.predict_with_confidence(&sample);
        assert!(!label.is_empty());
        assert!(conf > 0.0 && conf <= 1.0);
    }

    #[test]
    fn test_feature_importance() {
        let data = make_large_dataset();
        let config = RandomForestConfig::new(5);
        let forest = RandomForest::build(&data, &config);
        let importance = forest.feature_importance(2);
        assert_eq!(importance.len(), 2);
        let total: f64 = importance.iter().sum();
        assert!((total - 1.0).abs() < 1e-10 || total == 0.0);
    }

    #[test]
    fn test_bootstrap_sample() {
        let data = make_xor_dataset();
        let (bootstrap, indices) = bootstrap_sample(&data, 42);
        assert_eq!(bootstrap.len(), data.len());
        assert_eq!(indices.len(), data.len());
        // All indices should be valid
        for &i in &indices {
            assert!(i < data.len());
        }
    }

    #[test]
    fn test_select_features() {
        let mut rng = Rng::new(42);
        let features = select_features(&mut rng, 10, 3);
        assert_eq!(features.len(), 3);
        // All features should be unique
        let mut sorted = features.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), features.len());
    }

    #[test]
    fn test_bagging_predict() {
        let data = make_large_dataset();
        let config = RandomForestConfig::new(3);
        let forest = RandomForest::build(&data, &config);
        let sample = Sample::new(vec!["a".into(), "x".into()], "".into());
        let pred = bagging_predict(&forest.trees, &sample);
        assert!(!pred.is_empty());
    }

    #[test]
    fn test_default_config() {
        let config = RandomForestConfig::default();
        assert_eq!(config.n_trees, 10);
        assert_eq!(config.max_features, 0);
        assert!(config.feature_bagging);
    }

    #[test]
    fn test_reproducibility() {
        let data = make_large_dataset();
        let config = RandomForestConfig::new(5);
        let forest1 = RandomForest::build(&data, &config);
        let forest2 = RandomForest::build(&data, &config);
        let sample = Sample::new(vec!["a".into(), "x".into()], "".into());
        assert_eq!(forest1.predict(&sample), forest2.predict(&sample));
    }

    #[test]
    fn test_empty_accuracy() {
        let data = make_large_dataset();
        let config = RandomForestConfig::new(3);
        let forest = RandomForest::build(&data, &config);
        let acc = forest.accuracy(&[]);
        assert_eq!(acc, 0.0);
    }
}
