import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

def perform_scaler(X_train, X_test):
    """Standaryzuje dane treningowe i testowe."""
    scaler = StandardScaler()
    train_std = scaler.fit_transform(X_train)
    test_std = scaler.transform(X_test)
    return train_std, test_std

def perform_pca(X_train, X_test, n_components=100):
    """Przeprowadza redukcję wymiarowości za pomocą PCA."""
    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(X_train)
    test_pca = pca.transform(X_test)
    return train_pca, test_pca

def optimize_svm(X_train, y_train, n_trials=20):
    """Szuka najlepszych hiperparametrów dla modelu SVM przy użyciu Optuny."""
    def objective(trial):
        C = trial.suggest_float("C", 0.01, 10.0)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        max_iter = trial.suggest_int('max_iter', 1000, 2000)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        
        mdl = SVC(C=C, kernel=kernel, max_iter=max_iter, gamma=gamma, random_state=42)
        scores = cross_validate(mdl, X_train, y_train,
                                scoring={'f1_macro': 'f1_weighted'},
                                cv=StratifiedKFold(n_splits=5),
                                return_train_score=False)
        return np.mean(scores['test_f1_macro'])

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params, study.best_value

def optimize_rf(X_train, y_train, n_trials=5):
    """Szuka najlepszych hiperparametrów dla lasu losowego przy użyciu Optuny."""
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 200, 600)
        max_depth = trial.suggest_int("max_depth", 5, 25)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        
        mdl = RandomForestClassifier(n_estimators=n_estimators, 
                                     max_depth=max_depth, 
                                     min_samples_split=min_samples_split, 
                                     n_jobs=-1, 
                                     random_state=42, 
                                     class_weight='balanced')
        scores = cross_validate(mdl, X_train, y_train,
                                scoring={'f1_macro': 'f1_weighted'},
                                cv=StratifiedKFold(n_splits=5),
                                return_train_score=False)
        return np.mean(scores['test_f1_macro'])

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params, study.best_value