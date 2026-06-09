//! # decision-tree-rs Tutorial
//!
//! A progressive walkthrough of decision tree classification in pure Rust,
//! covering tree building, splitting criteria, pruning, evaluation, and
//! random forests.
//!
//! ## Lessons
//!
//! 1. **Your First Tree** — build, predict, and display a decision tree
//! 2. **Splitting Criteria** — information gain (ID3) vs Gini impurity (CART)
//! 3. **Evaluation & Metrics** — confusion matrix, accuracy, precision, recall, F1
//! 4. **Pruning** — reduced error pruning and cost-complexity pruning
//! 5. **Random Forests** — ensemble learning with bagging and OOB estimation
//! 6. **Cross-Validation & Feature Importance** — robust model assessment

fn main() {
    println!("=== decision-tree-rs Tutorial ===\n");

    lesson_1_first_tree();
    lesson_2_splitting_criteria();
    lesson_3_evaluation();
    lesson_4_pruning();
    lesson_5_random_forest();
    lesson_6_cv_and_importance();

    println!("=== Tutorial Complete! ===");
}

// ─── Lesson 1: Your First Tree ───────────────────────────────────────────────

fn lesson_1_first_tree() {
    println!("Lesson 1: Your First Decision Tree");
    println!("-----------------------------------\n");

    use decision_tree_rs::{DecisionTree, tree::Sample};
    use decision_tree_rs::tree::SplitCriterion;

    // --- The classic "play tennis" dataset ---
    let data = vec![
        Sample::new(vec!["sunny".into(),    "hot".into(),  "high".into()],   "no".into()),
        Sample::new(vec!["sunny".into(),    "hot".into(),  "normal".into()], "no".into()),
        Sample::new(vec!["overcast".into(), "hot".into(),  "high".into()],   "yes".into()),
        Sample::new(vec!["rainy".into(),    "mild".into(), "high".into()],   "yes".into()),
        Sample::new(vec!["rainy".into(),    "cool".into(), "normal".into()], "yes".into()),
        Sample::new(vec!["rainy".into(),    "cool".into(), "normal".into()], "no".into()),
        Sample::new(vec!["overcast".into(), "cool".into(), "normal".into()], "yes".into()),
        Sample::new(vec!["sunny".into(),    "mild".into(), "high".into()],   "no".into()),
        Sample::new(vec!["sunny".into(),    "cool".into(), "normal".into()], "yes".into()),
        Sample::new(vec!["rainy".into(),    "mild".into(), "normal".into()], "yes".into()),
        Sample::new(vec!["sunny".into(),    "mild".into(), "normal".into()], "yes".into()),
        Sample::new(vec!["overcast".into(), "mild".into(), "high".into()],   "yes".into()),
        Sample::new(vec!["overcast".into(), "hot".into(),  "normal".into()], "yes".into()),
        Sample::new(vec!["rainy".into(),    "mild".into(), "high".into()],   "no".into()),
    ];

    // Build with feature names for readable output
    let tree = DecisionTree::build_with_names(
        &data,
        vec!["outlook".into(), "temperature".into(), "humidity".into()],
        SplitCriterion::InformationGain,
    );

    println!("Tree depth: {}", tree.depth());
    println!("Number of leaves: {}", tree.num_leaves());

    // Display the tree structure
    println!("\nTree structure:");
    println!("{}", tree.display());

    // --- Make predictions ---
    let test = Sample::new(vec!["overcast".into(), "hot".into(), "high".into()], "".into());
    let pred = tree.predict(&test);
    println!("Prediction for (overcast, hot, high): {}", pred);
    assert_eq!(pred, "yes");

    let test2 = Sample::new(vec!["sunny".into(), "hot".into(), "high".into()], "".into());
    let pred2 = tree.predict(&test2);
    println!("Prediction for (sunny, hot, high): {}", pred2);
    assert_eq!(pred2, "no");

    println!();
}

// ─── Lesson 2: Splitting Criteria ────────────────────────────────────────────

fn lesson_2_splitting_criteria() {
    println!("Lesson 2: Splitting Criteria");
    println!("-----------------------------\n");

    use decision_tree_rs::tree::Sample;
    use decision_tree_rs::gain::{self, entropy, information_gain, all_information_gains};
    use decision_tree_rs::gini::{self, gini_impurity, gini_gain, all_gini_gains};
    use decision_tree_rs::tree::SplitCriterion;
    use decision_tree_rs::DecisionTree;

    let data = vec![
        Sample::new(vec!["a".into(), "x".into()], "pos".into()),
        Sample::new(vec!["a".into(), "y".into()], "pos".into()),
        Sample::new(vec!["b".into(), "x".into()], "neg".into()),
        Sample::new(vec!["b".into(), "y".into()], "neg".into()),
        Sample::new(vec!["c".into(), "x".into()], "pos".into()),
        Sample::new(vec!["c".into(), "y".into()], "neg".into()),
    ];

    // --- Entropy ---
    let all_pos = vec!["pos".into(), "pos".into()];
    let half_half = vec!["pos".into(), "neg".into()];
    let three_way = vec!["a".into(), "b".into(), "c".into()];

    println!("Entropy:");
    println!("  All same:       H = {:.4}", entropy(&all_pos));
    println!("  50/50 split:    H = {:.4}", entropy(&half_half));
    println!("  3-way uniform:  H = {:.4} (log₂3 ≈ {:.4})", entropy(&three_way), 3.0f64.log2());
    assert!((entropy(&all_pos) - 0.0).abs() < 1e-10);
    assert!((entropy(&half_half) - 1.0).abs() < 1e-10);

    // --- Information Gain ---
    println!("\nInformation gain per feature:");
    let gains = all_information_gains(&data, 2);
    for (idx, g) in &gains {
        println!("  Feature {}: IG = {:.4}", idx, g);
    }

    // Feature 0 perfectly separates pos/neg (a→pos, b→neg, c→mixed)
    let ig0 = information_gain(&data, 0);
    let ig1 = information_gain(&data, 1);
    println!("  Best feature: {} (IG = {:.4})", if ig0 > ig1 { 0 } else { 1 }, ig0.max(ig1));

    // --- Gini Impurity ---
    println!("\nGini impurity:");
    println!("  Pure:           Gini = {:.4}", gini_impurity(&all_pos));
    println!("  50/50:          Gini = {:.4}", gini_impurity(&half_half));
    println!("  3-way uniform:  Gini = {:.4}", gini_impurity(&three_way));
    assert!((gini_impurity(&all_pos) - 0.0).abs() < 1e-10);
    assert!((gini_impurity(&half_half) - 0.5).abs() < 1e-10);

    // --- Gini Gain ---
    println!("\nGini gain per feature:");
    let gg = all_gini_gains(&data, 2);
    for (idx, g) in &gg {
        println!("  Feature {}: GiniGain = {:.4}", idx, g);
    }

    // --- Compare ID3 vs CART trees ---
    let tree_id3 = DecisionTree::build(&data, SplitCriterion::InformationGain);
    let tree_gini = DecisionTree::build(&data, SplitCriterion::Gini);

    println!("\nID3 tree: depth={}, leaves={}", tree_id3.depth(), tree_id3.num_leaves());
    println!("CART tree: depth={}, leaves={}", tree_gini.depth(), tree_gini.num_leaves());

    // Both should predict correctly on training data
    for s in &data {
        assert_eq!(tree_id3.predict(s), s.label);
        assert_eq!(tree_gini.predict(s), s.label);
    }
    println!("  ✓ Both trees predict correctly on all training samples");

    println!();
}

// ─── Lesson 3: Evaluation & Metrics ──────────────────────────────────────────

fn lesson_3_evaluation() {
    println!("Lesson 3: Evaluation & Metrics");
    println!("-------------------------------\n");

    use decision_tree_rs::{DecisionTree, tree::Sample};
    use decision_tree_rs::tree::SplitCriterion;
    use decision_tree_rs::predict::{self, ConfusionMatrix, evaluate, accuracy};

    let data = vec![
        Sample::new(vec!["low".into(),    "no".into()],  "no".into()),
        Sample::new(vec!["low".into(),    "yes".into()], "yes".into()),
        Sample::new(vec!["medium".into(), "no".into()],  "yes".into()),
        Sample::new(vec!["medium".into(), "yes".into()], "yes".into()),
        Sample::new(vec!["high".into(),   "no".into()],  "yes".into()),
        Sample::new(vec!["high".into(),   "yes".into()], "yes".into()),
    ];

    let tree = DecisionTree::build_with_names(
        &data,
        vec!["income".into(), "student".into()],
        SplitCriterion::Gini,
    );

    // --- Batch prediction ---
    let preds = predict::predict_batch(&tree, &data);
    println!("Batch predictions: {:?}", preds);

    // --- Accuracy ---
    let acc = accuracy(&tree, &data);
    println!("Training accuracy: {:.2}%", acc * 100.0);

    // --- Confusion Matrix ---
    let cm = evaluate(&tree, &data);
    println!("\nConfusion matrix:");
    println!("  Labels: {:?}", cm.labels);
    for label in &cm.labels {
        println!(
            "  Class '{}': precision={:.3}, recall={:.3}, F1={:.3}",
            label,
            cm.precision(label),
            cm.recall(label),
            cm.f1(label)
        );
    }

    // --- Manual confusion matrix ---
    let actual = vec!["pos".into(), "pos".into(), "neg".into(), "neg".into()];
    let predicted = vec!["pos".into(), "neg".into(), "neg".into(), "pos".into()];
    let manual_cm = ConfusionMatrix::new(&actual, &predicted);
    println!("\nManual CM example:");
    println!("  Accuracy:  {:.3}", manual_cm.accuracy());
    println!("  Precision(pos): {:.3}", manual_cm.precision("pos"));
    println!("  Recall(pos):    {:.3}", manual_cm.recall("pos"));
    println!("  F1(pos):        {:.3}", manual_cm.f1("pos"));
    println!("  Precision(neg): {:.3}", manual_cm.precision("neg"));
    println!("  Recall(neg):    {:.3}", manual_cm.recall("neg"));
    assert!((manual_cm.accuracy() - 0.5).abs() < 1e-10); // 2/4 correct

    println!();
}

// ─── Lesson 4: Pruning ───────────────────────────────────────────────────────

fn lesson_4_pruning() {
    println!("Lesson 4: Pruning");
    println!("-----------------\n");

    use decision_tree_rs::{DecisionTree, tree::Sample};
    use decision_tree_rs::tree::SplitCriterion;
    use decision_tree_rs::prune::reduced_error_prune;
    use decision_tree_rs::pruning::{cost_complexity_prune, prune_to_alpha, find_optimal_alpha};

    // Create a dataset where some features are noisy
    let train = vec![
        Sample::new(vec!["a".into(), "x".into(), "p".into()], "pos".into()),
        Sample::new(vec!["a".into(), "y".into(), "q".into()], "pos".into()),
        Sample::new(vec!["a".into(), "x".into(), "p".into()], "pos".into()),
        Sample::new(vec!["b".into(), "x".into(), "p".into()], "neg".into()),
        Sample::new(vec!["b".into(), "y".into(), "q".into()], "neg".into()),
        Sample::new(vec!["b".into(), "y".into(), "p".into()], "neg".into()),
        Sample::new(vec!["c".into(), "x".into(), "q".into()], "pos".into()),
        Sample::new(vec!["c".into(), "y".into(), "p".into()], "neg".into()),
    ];

    let validation = vec![
        Sample::new(vec!["a".into(), "x".into(), "p".into()], "pos".into()),
        Sample::new(vec!["b".into(), "y".into(), "q".into()], "neg".into()),
        Sample::new(vec!["c".into(), "x".into(), "q".into()], "pos".into()),
    ];

    let tree = DecisionTree::build(&train, SplitCriterion::Gini);
    println!("Original tree: depth={}, leaves={}", tree.depth(), tree.num_leaves());
    println!("{}", tree.display());

    // --- Reduced Error Pruning ---
    let rep_pruned = reduced_error_prune(&tree, &validation);
    println!(
        "After reduced-error pruning: depth={}, leaves={}",
        rep_pruned.depth(),
        rep_pruned.num_leaves()
    );
    assert!(rep_pruned.num_leaves() <= tree.num_leaves());

    // Still correct on validation
    for s in &validation {
        let p = rep_pruned.predict(s);
        println!("  Predicted {} for {:?} → actual {}", p, s.features, s.label);
    }

    // --- Cost-Complexity Pruning ---
    let ccp_result = cost_complexity_prune(&tree);
    println!("\nCost-complexity pruning path:");
    for (i, (alpha, leaves)) in ccp_result.alphas.iter()
        .zip(ccp_result.leaf_counts.iter())
        .enumerate()
    {
        println!("  Step {}: α={:.6}, leaves={}", i, alpha, leaves);
    }

    // Prune to specific alpha
    let alpha_pruned = prune_to_alpha(&tree, 0.0);
    println!(
        "\nPruned at α=0: depth={}, leaves={}",
        alpha_pruned.depth(),
        alpha_pruned.num_leaves()
    );

    // Prune to large alpha → stump
    let stump = prune_to_alpha(&tree, 1e10);
    println!(
        "Pruned at α=1e10: depth={}, leaves={}",
        stump.depth(),
        stump.num_leaves()
    );
    assert_eq!(stump.num_leaves(), 1);

    // Find optimal alpha via cross-validation
    let optimal = find_optimal_alpha(&train, 3, SplitCriterion::Gini, 42);
    println!("Optimal α (3-fold CV): {:.6}", optimal);

    println!();
}

// ─── Lesson 5: Random Forests ────────────────────────────────────────────────

fn lesson_5_random_forest() {
    println!("Lesson 5: Random Forests");
    println!("------------------------\n");

    use decision_tree_rs::random_forest::{
        RandomForest, RandomForestConfig, bootstrap_sample, bagging_predict,
    };
    use decision_tree_rs::tree::Sample;

    // Create a larger dataset
    let mut data = Vec::new();
    for i in 0..60 {
        let x = if i % 3 == 0 { "a" } else if i % 3 == 1 { "b" } else { "c" };
        let y = if i % 2 == 0 { "hot" } else { "cold" };
        let label = if i % 3 == 0 { "positive" } else { "negative" };
        data.push(Sample::new(vec![x.into(), y.into()], label.into()));
    }

    // --- Build a random forest ---
    let config = RandomForestConfig {
        n_trees: 15,
        max_features: 0, // auto = sqrt(num_features)
        feature_bagging: true,
        criterion: decision_tree_rs::tree::SplitCriterion::Gini,
        seed: 42,
        ..Default::default()
    };

    let forest = RandomForest::build(&data, &config);
    println!("Forest built: {} trees", forest.n_trees());

    // --- Predict with majority vote ---
    let test = Sample::new(vec!["a".into(), "hot".into()], "".into());
    let pred = forest.predict(&test);
    println!("Prediction for (a, hot): {}", pred);
    assert_eq!(pred, "positive");

    let test2 = Sample::new(vec!["b".into(), "cold".into()], "".into());
    let pred2 = forest.predict(&test2);
    println!("Prediction for (b, cold): {}", pred2);
    assert_eq!(pred2, "negative");

    // --- Predict with confidence ---
    let (label, conf) = forest.predict_with_confidence(&test);
    println!("Confidence for (a, hot): {} ({:.1}%)", label, conf * 100.0);

    // --- Accuracy ---
    let acc = forest.accuracy(&data);
    println!("Training accuracy: {:.2}%", acc * 100.0);

    // --- Out-of-Bag error estimate ---
    let oob = forest.oob_error(&data);
    println!("OOB error estimate: {:.2}%", oob * 100.0);

    // --- Feature importance ---
    let importance = forest.feature_importance(2);
    println!("\nFeature importance:");
    println!("  Feature 0: {:.4}", importance[0]);
    println!("  Feature 1: {:.4}", importance[1]);

    // --- Bootstrap sampling ---
    let (bootstrap, indices) = bootstrap_sample(&data, 123);
    println!("\nBootstrap sample (seed=123): {} samples", bootstrap.len());
    println!("  Sampled indices (first 10): {:?}", &indices[..10.min(indices.len())]);

    // --- Manual bagging ---
    let bag_pred = bagging_predict(&forest.trees, &test);
    println!("Bagging prediction for (a, hot): {}", bag_pred);

    // --- Reproducibility ---
    let config2 = RandomForestConfig {
        n_trees: 15,
        seed: 42,
        ..Default::default()
    };
    let forest2 = RandomForest::build(&data, &config2);
    assert_eq!(forest.predict(&test), forest2.predict(&test));
    println!("\n  ✓ Same seed → same predictions (reproducible)");

    println!();
}

// ─── Lesson 6: Cross-Validation & Feature Importance ─────────────────────────

fn lesson_6_cv_and_importance() {
    println!("Lesson 6: Cross-Validation & Feature Importance");
    println!("------------------------------------------------\n");

    use decision_tree_rs::{DecisionTree, tree::Sample};
    use decision_tree_rs::tree::SplitCriterion;
    use decision_tree_rs::predict::{cross_validate, accuracy, evaluate};
    use decision_tree_rs::random_forest::{RandomForest, RandomForestConfig};

    // --- A more realistic dataset with multiple features ---
    let mut data = Vec::new();
    // Pattern: "young" + "low" → "no", "old" + "high" → "yes"
    for _ in 0..15 {
        data.push(Sample::new(vec!["young".into(), "low".into(),   "no".into()],  "no".into()));
    }
    for _ in 0..15 {
        data.push(Sample::new(vec!["old".into(),   "high".into(),  "yes".into()], "yes".into()));
    }
    for _ in 0..10 {
        data.push(Sample::new(vec!["young".into(), "high".into(),  "yes".into()], "yes".into()));
    }
    for _ in 0..10 {
        data.push(Sample::new(vec!["old".into(),   "low".into(),   "no".into()],  "no".into()));
    }
    println!("Dataset: {} samples, 3 features, 2 classes", data.len());

    // --- K-Fold Cross-Validation for Decision Tree ---
    let cv_acc_ig = cross_validate(&data, 5, SplitCriterion::InformationGain);
    let cv_acc_gini = cross_validate(&data, 5, SplitCriterion::Gini);
    println!("\n5-fold CV accuracy:");
    println!("  ID3 (InfoGain):  {:.2}%", cv_acc_ig * 100.0);
    println!("  CART (Gini):     {:.2}%", cv_acc_gini * 100.0);

    // --- Compare tree vs forest ---
    let tree = DecisionTree::build(&data, SplitCriterion::Gini);
    let tree_acc = accuracy(&tree, &data);

    let config = RandomForestConfig {
        n_trees: 20,
        seed: 42,
        ..Default::default()
    };
    let forest = RandomForest::build(&data, &config);
    let forest_acc = forest.accuracy(&data);

    println!("\nTraining accuracy comparison:");
    println!("  Single tree:  {:.2}%", tree_acc * 100.0);
    println!("  Random forest: {:.2}%", forest_acc * 100.0);

    // --- Feature importance ---
    let importance = forest.feature_importance(3);
    let names = ["age", "income", "subscribed"];
    println!("\nRandom forest feature importance:");
    for (i, name) in names.iter().enumerate() {
        println!("  {} (feature {}): {:.4}", name, i, importance[i]);
    }

    // --- Detailed evaluation ---
    let cm = evaluate(&tree, &data);
    println!("\nDecision tree detailed metrics:");
    for label in &cm.labels {
        println!(
            "  Class '{}': precision={:.3}, recall={:.3}, F1={:.3}",
            label,
            cm.precision(label),
            cm.recall(label),
            cm.f1(label)
        );
    }

    // --- OOB as free validation ---
    let oob = forest.oob_error(&data);
    println!("\nRandom forest OOB error: {:.2}%  (≈ test error)", oob * 100.0);

    println!();
}
