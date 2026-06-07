//! Information gain and entropy computations for decision tree splitting.

use crate::tree::{Label, Sample};

/// Compute the Shannon entropy of a label distribution.
///
/// H(S) = -Σ p_i * log2(p_i)
pub fn entropy(labels: &[Label]) -> f64 {
    if labels.is_empty() {
        return 0.0;
    }
    let n = labels.len() as f64;
    let mut counts = std::collections::HashMap::new();
    for l in labels {
        *counts.entry(l.as_str()).or_insert(0usize) += 1;
    }
    counts
        .values()
        .map(|&c| {
            let p = c as f64 / n;
            if p > 0.0 { -p * p.log2() } else { 0.0 }
        })
        .sum()
}

/// Compute the entropy of a sample set.
pub fn sample_entropy(samples: &[Sample]) -> f64 {
    let labels: Vec<&Label> = samples.iter().map(|s| &s.label).collect();
    let owned: Vec<Label> = labels.into_iter().cloned().collect();
    entropy(&owned)
}

/// Compute the information gain of splitting on a feature.
///
/// IG(S, feature) = H(S) - Σ (|S_v|/|S|) * H(S_v)
pub fn information_gain(samples: &[Sample], feature_index: usize) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let total_entropy = sample_entropy(samples);
    let n = samples.len() as f64;

    let mut partitions: std::collections::HashMap<&str, Vec<&Sample>> =
        std::collections::HashMap::new();
    for s in samples {
        partitions
            .entry(&s.features[feature_index])
            .or_default()
            .push(s);
    }

    let mut weighted_entropy: f64 = 0.0;
    for subset in partitions.values() {
        let labels: Vec<Label> = subset.iter().map(|s| s.label.clone()).collect();
        weighted_entropy += (subset.len() as f64 / n) * entropy(&labels);
    }

    total_entropy - weighted_entropy
}

/// Find the best feature to split on using information gain.
///
/// Returns the feature index with the highest information gain.
pub fn best_split_info_gain(samples: &[Sample], num_features: usize) -> Option<usize> {
    let mut best_idx = None;
    let mut best_gain = f64::NEG_INFINITY;

    for i in 0..num_features {
        let gain = information_gain(samples, i);
        if gain > best_gain {
            best_gain = gain;
            best_idx = Some(i);
        }
    }

    // Only split if there's positive gain
    if best_gain > 1e-10 {
        best_idx
    } else {
        None
    }
}

/// Compute all information gains for all features.
pub fn all_information_gains(samples: &[Sample], num_features: usize) -> Vec<(usize, f64)> {
    (0..num_features)
        .map(|i| (i, information_gain(samples, i)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::Sample;

    #[test]
    fn test_entropy_uniform() {
        let labels: Vec<Label> = vec!["a".into(), "b".into()];
        let e = entropy(&labels);
        assert!((e - 1.0).abs() < 1e-10, "Entropy of uniform binary should be 1.0, got {}", e);
    }

    #[test]
    fn test_entropy_pure() {
        let labels: Vec<Label> = vec!["a".into(), "a".into()];
        let e = entropy(&labels);
        assert!(e.abs() < 1e-10, "Entropy of pure set should be 0.0, got {}", e);
    }

    #[test]
    fn test_entropy_single() {
        let labels: Vec<Label> = vec!["a".into()];
        let e = entropy(&labels);
        assert!(e.abs() < 1e-10);
    }

    #[test]
    fn test_entropy_empty() {
        let labels: Vec<Label> = vec![];
        let e = entropy(&labels);
        assert!(e.abs() < 1e-10);
    }

    #[test]
    fn test_entropy_three_classes() {
        let labels: Vec<Label> = vec!["a".into(), "b".into(), "c".into()];
        let e = entropy(&labels);
        assert!((e - (3.0f64).log2()).abs() < 1e-10);
    }

    #[test]
    fn test_information_gain_perfect_split() {
        let samples = vec![
            Sample::new(vec!["a".into()], "x".into()),
            Sample::new(vec!["a".into()], "x".into()),
            Sample::new(vec!["b".into()], "y".into()),
            Sample::new(vec!["b".into()], "y".into()),
        ];
        let ig = information_gain(&samples, 0);
        assert!(ig > 0.9, "Perfect split should have high IG, got {}", ig);
    }

    #[test]
    fn test_information_gain_no_split() {
        let samples = vec![
            Sample::new(vec!["a".into()], "x".into()),
            Sample::new(vec!["a".into()], "y".into()),
        ];
        let ig = information_gain(&samples, 0);
        assert!(ig.abs() < 1e-10, "No information gain when split doesn't help");
    }

    #[test]
    fn test_best_split() {
        let samples = vec![
            Sample::new(vec!["a".into(), "x".into()], "pos".into()),
            Sample::new(vec!["b".into(), "x".into()], "neg".into()),
            Sample::new(vec!["a".into(), "y".into()], "pos".into()),
            Sample::new(vec!["b".into(), "y".into()], "neg".into()),
        ];
        let best = best_split_info_gain(&samples, 2);
        assert_eq!(best, Some(0)); // Feature 0 perfectly separates
    }

    #[test]
    fn test_all_information_gains() {
        let samples = vec![
            Sample::new(vec!["a".into(), "x".into()], "p".into()),
            Sample::new(vec!["b".into(), "y".into()], "n".into()),
        ];
        let gains = all_information_gains(&samples, 2);
        assert_eq!(gains.len(), 2);
    }
}
