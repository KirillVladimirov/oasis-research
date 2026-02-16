Domain: Deep Active Learning (deep_active_learning). DAL combines active learning (selecting which unlabeled
instances to annotate under a limited budget) with deep neural models. Core elements include
pool-based and stream-based settings, acquisition functions (e.g., uncertainty-based,
diversity-based, Bayesian methods), annotation budget and cost models, stopping criteria,
noisy labels, class imbalance, domain shift, and task families such as classification, NER,
and detection. A typical loop trains a model on labeled data, scores unlabeled candidates,
queries an oracle, updates the labeled set, and repeats; evaluation uses data efficiency and
performance vs. labeling cost.
