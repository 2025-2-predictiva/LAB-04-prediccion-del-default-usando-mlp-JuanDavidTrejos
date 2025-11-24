"""
Script para entrenar el modelo de clasificación de default de crédito usando SVC.
"""

import pandas as pd
import numpy as np
import json
import gzip
import pickle
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_sample_weight


def load_data():
    """Carga los datasets de entrenamiento y prueba."""
    train_data = pd.read_csv('../files/input/train_data.csv.zip', compression='zip')
    test_data = pd.read_csv('../files/input/test_data.csv.zip', compression='zip')
    return train_data, test_data


def clean_datasets(df1, df2):
    """Limpia los datasets removiendo valores faltantes y estandarizando categorías."""
    # Renombrar columna
    df1 = df1.rename(columns={"default payment next month": "default"})
    df2 = df2.rename(columns={"default payment next month": "default"})

    # Remover columna ID
    df1 = df1.drop(columns=["ID"])
    df2 = df2.drop(columns=["ID"])

    # Eliminar registros con información no disponible
    df1 = df1.replace("?", pd.NA)
    df2 = df2.replace("?", pd.NA)
    
    df1 = df1[df1['EDUCATION'] != 0]
    df2 = df2[df2['EDUCATION'] != 0]
    
    df1 = df1[df1['MARRIAGE'] != 0]
    df2 = df2[df2['MARRIAGE'] != 0]
    
    df1 = df1.dropna()
    df2 = df2.dropna()

    # Agrupar educación superior en categoría 4
    df1['EDUCATION'] = df1['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    df2['EDUCATION'] = df2['EDUCATION'].apply(lambda x: 4 if x > 4 else x)

    return df1, df2


def split_features_target(train_data, test_data):
    """Divide los datasets en características y variable objetivo."""
    x_train = train_data.drop(columns=['default'])
    y_train = train_data['default']
    
    x_test = test_data.drop(columns=['default'])
    y_test = test_data['default']
    
    return x_train, y_train, x_test, y_test


def create_pipeline(x_train, y_train):
    """Crea y entrena el pipeline de clasificación con optimización de hiperparámetros."""
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('feature_selection', SelectKBest(f_classif)),
        ('classifier', MLPClassifier(
            random_state=42, 
            max_iter=1000, 
            early_stopping=True,  
            validation_fraction=0.1,  
            n_iter_no_change=20,  
            verbose=False
            ))
    ])

    param_grid = {
        'pca__n_components': [0.85, 0.90, 0.95],  
        'feature_selection__k': [15, 20, 25], 
        'classifier__hidden_layer_sizes': [  
            (50,),       # Small
            (100,),      # Medium
            (100, 50),   # Deep
        ],  
        'classifier__alpha': [0.01, 0.05, 0.1, 0.2],  
        'classifier__learning_rate_init': [0.001]  
    }

    # Compute sample weights to handle class imbalance
    sample_weights = compute_sample_weight('balanced', y_train)
    
    print(f"\nClass distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Sample weights: class 0 = {sample_weights[y_train == 0][0]:.4f}, class 1 = {sample_weights[y_train == 1][0]:.4f}\n")

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring={
            'balanced_accuracy': 'balanced_accuracy',
            'precision': 'precision'
        },
        refit='precision',  
        n_jobs=-1,
        verbose=2
    )

    # Fit with sample weights to handle class imbalance
    grid_search.fit(x_train, y_train, classifier__sample_weight=sample_weights)

    # Print best parameters
    print("\n" + "="*50)
    print("BEST PARAMETERS FOUND:")
    print("="*50)
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV Score (Precision): {grid_search.best_score_:.4f}")
    print("="*50 + "\n")

    return grid_search


def calculate_metrics(y_true, y_pred):
    """Calcula las métricas de evaluación."""
    return {
        'precision': round(precision_score(y_true, y_pred), 4),
        'balanced_accuracy': round(balanced_accuracy_score(y_true, y_pred), 4),
        'recall': round(recall_score(y_true, y_pred), 4),
        'f1_score': round(f1_score(y_true, y_pred), 4)
    }


def calculate_confusion_matrix(y_true, y_pred):
    """Calcula la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        'true_0': {
            "predicted_0": int(cm[0][0]),
            "predicted_1": int(cm[0][1])
        },
        'true_1': {
            "predicted_0": int(cm[1][0]),
            "predicted_1": int(cm[1][1])
        }
    }


def save_metrics(train_metrics, test_metrics, train_cm, test_cm):
    """Guarda las métricas en formato JSON."""
    os.makedirs('../files/output', exist_ok=True)
    
    train_metrics['type'] = 'metrics'
    train_metrics['dataset'] = 'train'
    
    test_metrics['type'] = 'metrics'
    test_metrics['dataset'] = 'test'
    
    train_cm['type'] = 'cm_matrix'
    train_cm['dataset'] = 'train'
    
    test_cm['type'] = 'cm_matrix'
    test_cm['dataset'] = 'test'
    
    with open("../files/output/metrics.json", "w") as f:
        f.write(json.dumps(train_metrics) + '\n')
        f.write(json.dumps(test_metrics) + '\n')
        f.write(json.dumps(train_cm) + '\n')
        f.write(json.dumps(test_cm) + '\n')


def save_model(pipeline, model_path):
    """Guarda el modelo entrenado."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with gzip.open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)


def load_model(model_path):
    """Carga un modelo guardado."""
    with gzip.open(model_path, 'rb') as f:
        return pickle.load(f)


def main():
    """Función principal que ejecuta el entrenamiento completo."""
    # Cargar y limpiar datos
    train_data, test_data = load_data()
    train_data, test_data = clean_datasets(train_data, test_data)
    
    # Dividir en características y objetivo
    x_train, y_train, x_test, y_test = split_features_target(train_data, test_data)
    
    # Verificar si existe modelo guardado
    model_path = Path('../files/models/model.pkl.gz')
    
    if model_path.exists():
        pipeline = load_model(model_path)
        print("\n✓ Loaded existing model from disk\n")
    else:
        # Crear y entrenar pipeline
        pipeline = create_pipeline(x_train, y_train)
        
        # Guardar modelo
        save_model(pipeline, model_path)
    
    # Find optimal threshold
    # best_threshold = find_best_threshold(pipeline, x_train, y_train)
    best_threshold = 0.78  

    # Make predictions with optimal threshold
    y_train_proba = pipeline.predict_proba(x_train)[:, 1]
    y_test_proba = pipeline.predict_proba(x_test)[:, 1]
    
    y_train_pred = (y_train_proba >= best_threshold).astype(int)
    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    
    # Calcular métricas
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Calcular matrices de confusión
    train_cm = calculate_confusion_matrix(y_train, y_train_pred)
    test_cm = calculate_confusion_matrix(y_test, y_test_pred)
    
    # Guardar métricas
    save_metrics(train_metrics, test_metrics, train_cm, test_cm)
    
    return pipeline, train_metrics, test_metrics


if __name__ == "__main__":
    pipeline, train_metrics, test_metrics = main()
