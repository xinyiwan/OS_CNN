import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def features_clf(model, device, data_loader):
    # Extract features and check if they're useful AT ALL
    model.eval()
    model.fc = nn.Identity()  # Remove FC temporarily

    train_features, train_labels = [], []
    with torch.no_grad():
        for batch_idx, (batch_data, batch_labels, batch_meta) in enumerate(data_loader):
            feats = model(batch_data.to(device))
            train_features.append(feats.cpu().numpy())
            train_labels.append(batch_labels.cpu().numpy())

    train_features = np.concatenate(train_features)
    train_labels = np.concatenate(train_labels)

    print(f"Extracted features shape: {train_features.shape}")
    print(f"Extracted labels shape: {train_labels.shape}")

    # Check with PCA visualization and test with simple logistic regression and 

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(train_features)

    # Visualize
    import matplotlib.pyplot as plt
    plt.scatter(features_2d[train_labels==0, 0], features_2d[train_labels==0, 1], label='Class 0', alpha=0.6)
    plt.scatter(features_2d[train_labels==1, 0], features_2d[train_labels==1, 1], label='Class 1', alpha=0.6)
    plt.legend()
    plt.title('Feature space separability')
    plt.savefig('PCA_feature_space.png')
    plt.show()
    
    # Test if features are predictive
    clf = LogisticRegression(max_iter=1000)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, train_features, train_labels, cv=3)
    print(f"Cross-val accuracy on frozen features: {scores.mean():.3f}")
    # If this is ~0.5, frozen features are USELESS