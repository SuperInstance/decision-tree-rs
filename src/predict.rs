//! Prediction and evaluation utilities for decision trees.

use crate::tree::{DecisionTree, Label, Sample};

/// Prediction result with confidence information.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionResult {
    /// The predicted label.
    pub label: Label,
    /// Confidence score (proportion of training samples at the leaf).
    pub confidence: f64,
}

/// Confusion matrix for classification evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfusionMatrix {
    /// Map of (actual, predicted) → count.
    pub matrix: std::collections::HashMap<(String, String), usize>,
    /// All unique labels.
    pub labels: Vec<String>,
}

impl ConfusionMatrix {
    /// Build a confusion matrix from actual and predicted labels.
    pub fn new(actual: &[Label], predicted: &[Label]) -> Self {
        let mut label_set = std::collections::BTreeSet::new();
        for l in actual.iter().chain(predicted.iter()) {
            label_set.insert(l.clone());
        }
        let labels: Vec<String> = label_set.into_iter().collect();

        let mut matrix = std::collections::HashMap::new();
        for (a, p) in actual.iter().zip(predicted.iter()) {
            *matrix.entry((a.clone(), p.clone())).or_insert(0) += 1;
        }

        Self { matrix, labels }
    }

    /// Overall accuracy: (correct predictions) / (total predictions).
    pub fn accuracy(&self) -> f64 {
        let total: usize = self.matrix.values().sum();
        if total == 0 {
            return 0.0;
        }
        let correct: usize = self
            .matrix
            .iter()
            .filter(|((a, p), _)| a == p)
            .map(|(_, &c)| c)
            .sum();
        correct as f64 / total as f64
    }

    /// Precision for a given class.
    pub fn precision(&self, class: &str) -> f64 {
        let tp = self.matrix.get(&(class.to_string(), class.to_string())).copied().unwrap_or(0);
        let predicted_pos: usize = self
            .matrix
            .iter()
            .filter(|((_, p), _)| p == class)
            .map(|(_, &c)| c)
            .sum();
        if predicted_pos == 0 {
            0.0
        } else {
            tp as f64 / predicted_pos as f64
        }
    }

    /// Recall for a given class.
    pub fn recall(&self, class: &str) -> f64 {
        let tp = self.matrix.get(&(class.to_string(), class.to_string())).copied().unwrap_or(0);
        let actual_pos: usize = self
            .matrix
            .iter()
            .filter(|((a, _), _)| a == class)
            .map(|(_, &c)| c)
            .sum();
        if actual_pos == 0 {
            0.0
        } else {
            tp as f64 / actual_pos as f64
        }
    }

    /// F1 score for a given class.
    pub fn f1(&self, class: &str) -> f64 {
        let p = self.precision(class);
        let r = self.recall(class);
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

/// Predict labels for a batch of samples.
pub fn predict_batch(tree: &DecisionTree, samples: &[Sample]) -> Vec<Label> {
    samples.iter().map(|s| tree.predict(s)).collect()
}

/// Evaluate a decision tree on test data, returning the confusion matrix.
pub fn evaluate(tree: &DecisionTree, test: &[Sample]) -> ConfusionMatrix {
    let actual: Vec<Label> = test.iter().map(|s| s.label.clone()).collect();
    let predicted: Vec<Label> = test.iter().map(|s| tree.predict(s)).collect();
    ConfusionMatrix::new(&actual, &predicted)
}

/// Compute classification accuracy on test data.
pub fn accuracy(tree: &DecisionTree, test: &[Sample]) -> f64 {
    if test.is_empty() {
        return 0.0;
    }
    let correct = test.iter().filter(|s| tree.predict(s) == s.label).count();
    correct as f64 / test.len() as f64
}

/// Cross-validation: k-fold accuracy estimate.
pub fn cross_validate(samples: &[Sample], k: usize, criterion: crate::tree::SplitCriterion) -> f64 {
    if samples.len() < k {
        return 0.0;
    }
    let fold_size = samples.len() / k;
    let mut accuracies = vec![];

    for fold in 0..k {
        let test_start = fold * fold_size;
        let test_end = if fold == k - 1 { samples.len() } else { test_start + fold_size };

        let train: Vec<Sample> = samples
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < test_start || *i >= test_end)
            .map(|(_, s)| s.clone())
            .collect();

        let test: Vec<Sample> = samples[test_start..test_end].to_vec();

        if train.is_empty() || test.is_empty() {
            continue;
        }

        let tree = DecisionTree::build(&train, criterion);
        let acc = accuracy(&tree, &test);
        accuracies.push(acc);
    }

    if accuracies.is_empty() {
        0.0
    } else {
        accuracies.iter().sum::<f64>() / accuracies.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{DecisionTree, Sample, SplitCriterion};

    fn make_dataset() -> Vec<Sample> {
        vec![
            Sample::new(vec!["a".into()], "pos".into()),
            Sample::new(vec!["b".into()], "neg".into()),
            Sample::new(vec!["a".into()], "pos".into()),
            Sample::new(vec!["b".into()], "neg".into()),
        ]
    }

    #[test]
    fn test_predict_batch() {
        let data = make_dataset();
        let tree = DecisionTree::build(&data, SplitCriterion::InformationGain);
        let preds = predict_batch(&tree, &data);
        assert_eq!(preds.len(), 4);
    }

    #[test]
    fn test_accuracy_perfect() {
        let data = make_dataset();
        let tree = DecisionTree::build(&data, SplitCriterion::InformationGain);
        let acc = accuracy(&tree, &data);
        assert!((acc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate() {
        let data = make_dataset();
        let tree = DecisionTree::build(&data, SplitCriterion::Gini);
        let cm = evaluate(&tree, &data);
        assert!((cm.accuracy() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_accuracy() {
        let actual = vec!["a".to_string(), "b".to_string(), "a".to_string()];
        let predicted = vec!["a".to_string(), "a".to_string(), "a".to_string()];
        let cm = ConfusionMatrix::new(&actual, &predicted);
        assert!((cm.accuracy() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_recall() {
        let actual = vec!["pos".to_string(), "pos".to_string(), "neg".to_string(), "neg".to_string()];
        let predicted = vec!["pos".to_string(), "neg".to_string(), "neg".to_string(), "neg".to_string()];
        let cm = ConfusionMatrix::new(&actual, &predicted);
        assert!((cm.precision("neg") - 2.0 / 3.0).abs() < 1e-10);
        assert!((cm.recall("neg") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_f1_score() {
        let actual = vec!["a".to_string(), "a".to_string(), "b".to_string()];
        let predicted = vec!["a".to_string(), "a".to_string(), "a".to_string()];
        let cm = ConfusionMatrix::new(&actual, &predicted);
        let f1_a = cm.f1("a");
        assert!(f1_a > 0.0 && f1_a <= 1.0);
    }

    #[test]
    fn test_cross_validation() {
        let mut data = vec![];
        for _ in 0..20 {
            data.push(Sample::new(vec!["a".into()], "pos".into()));
            data.push(Sample::new(vec!["b".into()], "neg".into()));
        }
        let cv_acc = cross_validate(&data, 5, SplitCriterion::InformationGain);
        assert!((cv_acc - 1.0).abs() < 1e-10, "Should get 100% on this separable data, got {}", cv_acc);
    }

    #[test]
    fn test_empty_accuracy() {
        let tree = DecisionTree::build(&make_dataset(), SplitCriterion::InformationGain);
        assert!((accuracy(&tree, &[]) - 0.0).abs() < 1e-10);
    }
}
