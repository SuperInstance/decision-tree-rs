# decision-tree-rs

Decision tree implementation: ID3 algorithm, information gain, Gini impurity, pruning, and prediction.

## Features

- **Tree**: ID3 and CART tree building with configurable split criteria
- **Gain**: Shannon entropy and information gain computation
- **Gini**: Gini impurity and Gini gain
- **Prune**: Reduced error pruning
- **Predict**: Batch prediction, confusion matrix, cross-validation, F1 score

Pure Rust, no external dependencies.

## Usage

```rust
use decision_tree_rs::{DecisionTree, tree::{Sample, SplitCriterion}};

let data = vec![
    Sample::new(vec!["sunny".into()], "no".into()),
    Sample::new(vec!["overcast".into()], "yes".into()),
];
let tree = DecisionTree::build(&data, SplitCriterion::InformationGain);
```

License: MIT OR Apache-2.0
