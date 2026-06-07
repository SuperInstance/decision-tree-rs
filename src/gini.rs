//! Gini impurity computations for CART-style decision tree splitting.

use crate::tree::{Label, Sample};

/// Compute the Gini impurity of a label distribution.
///
/// Gini(S) = 1 - Σ p_i^2
pub fn gini_impurity(labels: &[Label]) -> f64 {
    if labels.is_empty() {
        return 0.0;
    }
    let n = labels.len() as f64;
    let mut counts = std::collections::HashMap::new();
    for l in labels {
        *counts.entry(l.as_str()).or_insert(0usize) += 1;
    }
    1.0 - counts
        .values()
        .map(|&c| {
            let p = c as f64 / n;
            p * p
        })
        .sum::<f64>()
}

/// Compute the Gini impurity of a sample set.
pub fn sample_gini(samples: &[Sample]) -> f64 {
    let labels: Vec<Label> = samples.iter().map(|s| s.label.clone()).collect();
    gini_impurity(&labels)
}

/// Compute the Gini gain (impurity reduction) of splitting on a feature.
///
/// GiniGain(S, feature) = Gini(S) - Σ (|S_v|/|S|) * Gini(S_v)
pub fn gini_gain(samples: &[Sample], feature_index: usize) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let total_gini = sample_gini(samples);
    let n = samples.len() as f64;

    let mut partitions: std::collections::HashMap<&str, Vec<&Sample>> =
        std::collections::HashMap::new();
    for s in samples {
        partitions
            .entry(&s.features[feature_index])
            .or_default()
            .push(s);
    }

    let mut weighted_gini: f64 = 0.0;
    for subset in partitions.values() {
        let labels: Vec<Label> = subset.iter().map(|s| s.label.clone()).collect();
        weighted_gini += (subset.len() as f64 / n) * gini_impurity(&labels);
    }

    total_gini - weighted_gini
}

/// Find the best feature to split on using Gini impurity reduction.
pub fn best_split_gini(samples: &[Sample], num_features: usize) -> Option<usize> {
    let mut best_idx = None;
    let mut best_gain = f64::NEG_INFINITY;

    for i in 0..num_features {
        let gain = gini_gain(samples, i);
        if gain > best_gain {
            best_gain = gain;
            best_idx = Some(i);
        }
    }

    if best_gain > 1e-10 {
        best_idx
    } else {
        None
    }
}

/// Compute all Gini gains for all features.
pub fn all_gini_gains(samples: &[Sample], num_features: usize) -> Vec<(usize, f64)> {
    (0..num_features).map(|i| (i, gini_gain(samples, i))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::Sample;

    #[test]
    fn test_gini_pure() {
        let labels: Vec<Label> = vec!["a".into(), "a".into()];
        assert!((gini_impurity(&labels) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gini_uniform_binary() {
        let labels: Vec<Label> = vec!["a".into(), "b".into()];
        let g = gini_impurity(&labels);
        assert!((g - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gini_three_uniform() {
        let labels: Vec<Label> = vec!["a".into(), "b".into(), "c".into()];
        let g = gini_impurity(&labels);
        // 1 - 3*(1/3)^2 = 1 - 1/3 = 2/3
        assert!((g - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_gini_empty() {
        let labels: Vec<Label> = vec![];
        assert!((gini_impurity(&labels) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gini_range() {
        // Gini impurity should be in [0, 1 - 1/k]
        let labels: Vec<Label> = vec!["a".into(), "b".into(), "a".into()];
        let g = gini_impurity(&labels);
        assert!(g >= 0.0 && g <= 1.0);
    }

    #[test]
    fn test_gini_gain_perfect() {
        let samples = vec![
            Sample::new(vec!["a".into()], "x".into()),
            Sample::new(vec!["a".into()], "x".into()),
            Sample::new(vec!["b".into()], "y".into()),
            Sample::new(vec!["b".into()], "y".into()),
        ];
        let gg = gini_gain(&samples, 0);
        assert!(gg > 0.4, "Perfect split should have high Gini gain, got {}", gg);
    }

    #[test]
    fn test_gini_gain_zero() {
        let samples = vec![
            Sample::new(vec!["a".into()], "x".into()),
            Sample::new(vec!["a".into()], "y".into()),
        ];
        let gg = gini_gain(&samples, 0);
        assert!(gg.abs() < 1e-10);
    }

    #[test]
    fn test_best_split_gini() {
        let samples = vec![
            Sample::new(vec!["a".into(), "x".into()], "pos".into()),
            Sample::new(vec!["b".into(), "x".into()], "neg".into()),
            Sample::new(vec!["a".into(), "y".into()], "pos".into()),
            Sample::new(vec!["b".into(), "y".into()], "neg".into()),
        ];
        let best = best_split_gini(&samples, 2);
        assert_eq!(best, Some(0));
    }

    #[test]
    fn test_all_gini_gains() {
        let samples = vec![
            Sample::new(vec!["a".into()], "p".into()),
            Sample::new(vec!["b".into()], "n".into()),
        ];
        let gains = all_gini_gains(&samples, 1);
        assert_eq!(gains.len(), 1);
        assert!(gains[0].1 > 0.0);
    }
}
