"""Reproduce the original Abalone regression design; never overwrite archived metrics.

Age = Rings + 1.5 years. The identical held-out set is used for every method.
No estimator is selected using the held-out test scores.
"""
from pathlib import Path
import hashlib
import json
import platform
import warnings
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

SEED = 42
BASE = Path(__file__).resolve().parent
NUMERIC = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight',
           'Viscera_weight', 'Shell_weight']
COLUMNS = ['Sex'] + NUMERIC + ['Rings']
RF_SEARCH = dict(n_estimators=[50, 100, 200, 300], max_depth=[None, 10, 20, 30],
                 min_samples_split=[2, 5, 10], min_samples_leaf=[1, 2, 4],
                 max_features=['sqrt', 'log2', None])
ARCHIVED_RF_BEST = dict(n_estimators=300, max_depth=20, min_samples_split=10,
                        min_samples_leaf=2, max_features='log2')

def prepare_data(base=BASE):
    """Use official UCI rows, seed-42 80/20 split, and train-only numeric scaling."""
    base = Path(base)
    df = pd.read_csv(base / 'abalone.data', header=None, names=COLUMNS)
    assert df.shape == (4177, 9), 'Unexpected dataset shape'
    assert not df.isna().any().any(), 'Missing or malformed data'
    assert set(df.Sex.unique()) == {'F', 'I', 'M'}
    assert not df.duplicated().any(), 'Unexpected duplicate records'
    X = pd.get_dummies(df.drop(columns='Rings'), columns=['Sex'])
    assert X.shape[1] == 10  # seven measurements plus F/I/M, all retained
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=.2,
                                           random_state=SEED)
    raw_train, raw_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    scaled_train, scaled_test = raw_train.copy(), raw_test.copy()
    scaler = StandardScaler().fit(raw_train[NUMERIC])
    scaled_train[NUMERIC] = scaler.transform(raw_train[NUMERIC])
    scaled_test[NUMERIC] = scaler.transform(raw_test[NUMERIC])
    assert set(train_idx).isdisjoint(test_idx)
    assert np.allclose(scaled_train[NUMERIC].mean(), 0, atol=1e-12)
    for name, values in [('X_train_raw', raw_train), ('X_test_raw', raw_test),
                         ('X_train_scaled', scaled_train), ('X_test_scaled', scaled_test)]:
        values.to_csv(base / (name + '.csv'), index=False)
    df.loc[train_idx, ['Rings']].to_csv(base / 'y_train.csv', index=False)
    df.loc[test_idx, ['Rings']].to_csv(base / 'y_test.csv', index=False)
    pd.DataFrame({'row_id_0_based': np.r_[train_idx, test_idx],
                  'partition': ['train']*len(train_idx) + ['test']*len(test_idx)}).to_csv(
                  base / 'split_manifest.csv', index=False)
    audit = dict(rows=len(df), predictors=8, encoded_features=10, train_rows=len(train_idx),
                 test_rows=len(test_idx), seed=SEED, stratify=None,
                 zero_height_rows=int((df.Height == 0).sum()),
                 age_range=[float(df.Rings.min()+1.5), float(df.Rings.max()+1.5)],
                 test_sex_counts=df.loc[test_idx, 'Sex'].value_counts().to_dict(),
                 sha256=hashlib.sha256((base/'abalone.data').read_bytes()).hexdigest(),
                 numeric_scaler_mean=scaler.mean_.tolist(),
                 numeric_scaler_scale=scaler.scale_.tolist(),
                 columns=X.columns.tolist())
    (base/'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    # Read the serialized files, as in the original notebooks, to preserve CSV precision.
    return load_data(base)

def load_data(base=BASE):
    base = Path(base)
    return dict(raw_train=pd.read_csv(base/'X_train_raw.csv'),
                raw_test=pd.read_csv(base/'X_test_raw.csv'),
                scaled_train=pd.read_csv(base/'X_train_scaled.csv'),
                scaled_test=pd.read_csv(base/'X_test_scaled.csv'),
                y_train=pd.read_csv(base/'y_train.csv').Rings + 1.5,
                y_test=pd.read_csv(base/'y_test.csv').Rings + 1.5)

def metrics(y, pred):
    mse = mean_squared_error(y, pred)
    return dict(MAE=float(mean_absolute_error(y, pred)), MSE=float(mse),
                RMSE=float(np.sqrt(mse)), R2=float(r2_score(y, pred)))

def fit_method(name, data, search=True, base=BASE):
    """Fit a report method. RF tuning is five-fold CV on training rows only."""
    base = Path(base)
    scaled = name in ('Linear Regression', 'MLP')
    X_train = data['scaled_train' if scaled else 'raw_train']
    X_test = data['scaled_test' if scaled else 'raw_test']
    if name == 'Linear Regression':
        model = LinearRegression()
    elif name == 'Random Forest (default)':
        model = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1)
    elif name == 'Random Forest (tuned)':
        if search:
            cv = RandomizedSearchCV(RandomForestRegressor(random_state=SEED, n_jobs=1),
                    RF_SEARCH, n_iter=10, cv=5, scoring='neg_mean_squared_error',
                    random_state=SEED, n_jobs=1)
            cv.fit(X_train, data['y_train'])
            pd.DataFrame(cv.cv_results_).to_csv(base/'rf_cv_results.csv', index=False)
            model = cv.best_estimator_
        else:
            model = RandomForestRegressor(**ARCHIVED_RF_BEST, random_state=SEED, n_jobs=1)
    elif name == 'XGBoost':
        model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=.05,
                    subsample=.8, colsample_bytree=.8, objective='reg:squarederror',
                    random_state=SEED, n_jobs=1)
    elif name == 'MLP':
        model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                    solver='adam', max_iter=1000, random_state=SEED)
    else:
        raise ValueError(name)
    # RF search has already refitted the selected estimator; refitting identically
    # is unnecessary. The other estimators, including fixed-config RF, fit here.
    if not (name == 'Random Forest (tuned)' and search):
        model.fit(X_train, data['y_train'])
    pred = model.predict(X_test)
    result = dict(Method=name, **metrics(data['y_test'], pred))
    return model, pred, result

def subgroup_comparison(name, global_model, data, base=BASE):
    """Historical exploratory RF design and matched-design XGB/MLP subgroups.

    RF uses tuned global vs default local parameters, as in the report. This is
    confounded and must not be interpreted as a controlled causal comparison.
    XGB/MLP local models clone their corresponding global configuration.
    """
    scaled = name == 'MLP'
    Xtr, Xte = data['scaled_train' if scaled else 'raw_train'], data['scaled_test' if scaled else 'raw_test']
    rows = []
    for sex in ['F', 'I', 'M']:
        tr, te = Xtr['Sex_'+sex].astype(bool), Xte['Sex_'+sex].astype(bool)
        local = (RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1)
                 if name == 'Random Forest' else clone(global_model))
        local.fit(Xtr.loc[tr], data['y_train'].loc[tr])
        for kind, pred in [('Global', global_model.predict(Xte.loc[te])),
                           ('Local', local.predict(Xte.loc[te]))]:
            rows.append(dict(Method=name, Sex=sex, Model=kind, n_train=int(tr.sum()),
                             n_test=int(te.sum()), **metrics(data['y_test'].loc[te], pred)))
    return pd.DataFrame(rows)

def save_diagnostics(data, fitted, results, subgroup, base=BASE):
    base = Path(base)
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 120})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ['#718096', '#3182bd', '#6baed6', '#dd8452', '#55a868']
    for ax, metric in zip(axes, ['MAE', 'RMSE', 'R2']):
        ax.bar(results.Method, results[metric], color=colors)
        ax.set_title(metric + ' (verification run)')
        ax.tick_params(axis='x', rotation=55)
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(base/'verification_model_comparison.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pred = fitted['MLP'][1]; y = data['y_test']
    axes[0].scatter(y, pred, s=12, alpha=.4)
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    axes[0].set(xlabel='Actual age (years)', ylabel='Predicted age (years)', title='MLP verification fit')
    axes[1].hist(y-pred, bins=30, color='#7067a2', edgecolor='white')
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].set(xlabel='Actual minus predicted age (years)', ylabel='Count', title='MLP residual distribution')
    fig.tight_layout(); fig.savefig(base/'verification_mlp_diagnostics.png'); plt.close(fig)
    pred = fitted['Random Forest (tuned)'][1]
    bins = pd.cut(data['y_test'], [0, 5, 10, 15, 20, 25, 35])
    by_age = pd.DataFrame({'Age_bin': bins, 'Absolute_error': abs(data['y_test']-pred)})
    grouped = by_age.groupby('Age_bin', observed=False).Absolute_error.agg(['count', 'mean'])
    grouped.to_csv(base/'verification_rf_age_bins.csv')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(grouped.index.astype(str), grouped['mean'].fillna(0), color='#55a868')
    for i, row in enumerate(grouped.itertuples()):
        if row.count == 0:
            ax.text(i, .1, 'n=0', ha='center')
    ax.set(xlabel='Age interval (years)', ylabel='MAE (years)', title='Tuned RF error by age; empty intervals marked n=0')
    fig.tight_layout(); fig.savefig(base/'verification_rf_age_bins.png'); plt.close(fig)
    model = fitted['XGBoost'][0]
    importance = pd.Series(model.feature_importances_, index=data['raw_train'].columns).sort_values()
    importance.rename('Importance').to_csv(base/'verification_xgboost_importance.csv')
    fig, ax = plt.subplots(figsize=(8, 4))
    importance.plot.barh(ax=ax, color='#dd8452')
    ax.set(title='XGBoost verification feature importance', xlabel='Normalized gain importance')
    fig.tight_layout(); fig.savefig(base/'verification_xgboost_importance.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, method in zip(axes, ['Random Forest', 'XGBoost', 'MLP']):
        part = subgroup[subgroup.Method == method].pivot(index='Sex', columns='Model', values='MAE')
        part.plot.bar(ax=ax, color=['#3182bd', '#dd8452'], rot=0)
        ax.set(title=method + ': global vs local', ylabel='MAE (years)')
    fig.tight_layout(); fig.savefig(base/'verification_global_local.png'); plt.close(fig)

def run_all(base=BASE, search=True):
    base = Path(base)
    data = prepare_data(base)
    fitted, rows, groups, configurations = {}, [], [], {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for name in ['Linear Regression', 'Random Forest (default)', 'Random Forest (tuned)', 'XGBoost', 'MLP']:
            print('Training:', name, flush=True)
            model, pred, result = fit_method(name, data, search=search, base=base)
            fitted[name] = (model, pred); rows.append(result)
            configurations[name] = model.get_params()
        for name, key in [('Random Forest', 'Random Forest (tuned)'), ('XGBoost', 'XGBoost'), ('MLP', 'MLP')]:
            print('Subgroup analysis:', name, flush=True)
            groups.append(subgroup_comparison(name, fitted[key][0], data, base))
    results = pd.DataFrame(rows)
    subgroup = pd.concat(groups, ignore_index=True)
    results.to_csv(base/'verification_metrics.csv', index=False)
    subgroup.to_csv(base/'verification_subgroups.csv', index=False)
    pd.DataFrame({'actual_age': data['y_test'], **{n:p for n, (_,p) in fitted.items()}}).to_csv(
                 base/'verification_predictions.csv', index=False)
    environment = dict(python=platform.python_version(), platform=platform.system(),
        packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','matplotlib']},
        seed=SEED, rf_selection='5-fold training CV' if search else 'fixed archived best parameters',
        model_parameters=configurations,
        warnings=[str(w.message) for w in caught])
    (base/'verification_environment.json').write_text(json.dumps(environment, indent=2, default=str), encoding='utf-8')
    save_diagnostics(data, fitted, results, subgroup, base)
    # Check aggregate metrics against the sample-weighted subgroup values.
    for label, main in [('Random Forest','Random Forest (tuned)'),('XGBoost','XGBoost'),('MLP','MLP')]:
        sub = subgroup[(subgroup.Method == label) & (subgroup.Model == 'Global')]
        for metric in ['MAE','MSE']:
            weighted = np.average(sub[metric], weights=sub.n_test)
            actual = results.loc[results.Method == main, metric].iloc[0]
            assert np.isclose(weighted, actual), (label, metric)
    print(results.to_string(index=False), flush=True)
    return results
