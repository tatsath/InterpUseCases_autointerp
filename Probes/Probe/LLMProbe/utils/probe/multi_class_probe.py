import torch
import torch.nn.functional as F

class MultiClassLinearProbe(torch.nn.Module):
    """Multi-class linear probe for classification with more than 2 classes"""
    
    def __init__(self, dim, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        return self.linear(x)  # Return logits for multi-class

class MultiClassProbeTrainer:
    """Trainer for multi-class probes"""
    
    @staticmethod
    def train_probe(features, labels, epochs=100, lr=1e-2, device=torch.device("cpu")):
        """Train a multi-class probe"""
        num_classes = len(torch.unique(labels))
        probe = MultiClassLinearProbe(features.shape[1], num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = probe(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        return probe, loss.item()

    @staticmethod
    def train_and_evaluate_model(train_hidden_states, train_labels, test_hidden_states, test_labels,
                                num_layers, use_control_tasks, progress_callback=None, epochs=100, lr=0.01,
                                device=torch.device("cpu")):
        """Train multi-class probes across all layers and evaluate performance"""
        probes = []
        accuracies = []
        control_accuracies = []
        selectivities = []
        losses = []
        test_losses = []

        num_classes = len(torch.unique(train_labels))
        
        for layer in range(num_layers):
            # Update main progress
            main_progress = (layer) / num_layers
            progress_callback(main_progress, f"Training probe for layer {layer+1}/{num_layers}",
                              f"Working on layer {layer+1} of {num_layers}")

            train_feats = train_hidden_states[:, layer, :]
            test_feats = test_hidden_states[:, layer, :]

            # Train probe
            probe = MultiClassLinearProbe(train_feats.shape[1], num_classes).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

            # Print initial loss and accuracy
            with torch.no_grad():
                outputs = probe(train_feats)
                loss = criterion(outputs, train_labels).item()
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == train_labels).float().mean().item()
                output_msg = f"Layer {layer+1}/{num_layers} - Initial: loss={loss:.4f}, acc={acc:.4f}"
                print(output_msg)

            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = probe(train_feats)
                loss = criterion(outputs, train_labels)
                loss.backward()
                optimizer.step()

            # Evaluate on test set
            with torch.no_grad():
                test_outputs = probe(test_feats)
                test_loss = criterion(test_outputs, test_labels).item()
                test_preds = torch.argmax(test_outputs, dim=1)
                acc = (test_preds == test_labels).float().mean().item()

            probes.append(probe)
            accuracies.append(acc)
            losses.append(loss.item())
            test_losses.append(test_loss)

            # Control task (if enabled)
            if use_control_tasks:
                progress_callback(main_progress + 0.6/num_layers,
                                  f"Layer {layer+1}/{num_layers}: Control task",
                                  f"Training with shuffled labels to measure selectivity")

                shuffled_labels = train_labels[torch.randperm(train_labels.size(0))]
                ctrl_probe, _ = MultiClassProbeTrainer.train_probe(
                    train_feats, shuffled_labels, epochs=epochs, lr=lr, device=device)

                with torch.no_grad():
                    ctrl_outputs = ctrl_probe(test_feats)
                    ctrl_preds = torch.argmax(ctrl_outputs, dim=1)
                    ctrl_acc = (ctrl_preds == test_labels).float().mean().item()
                    control_accuracies.append(ctrl_acc)

                    selectivity = acc - ctrl_acc
                    selectivities.append(selectivity)

        # Update to 100%
        progress_callback(1.0, "Completed training all probes",
                          f"Trained probes for {num_layers} layers with best accuracy: {max(accuracies):.4f}")

        results = {
            'probes': probes,
            'accuracies': accuracies,
            'control_accuracies': control_accuracies if use_control_tasks else None,
            'selectivities': selectivities if use_control_tasks else None,
            'losses': losses,
            'test_losses': test_losses
        }

        return results
