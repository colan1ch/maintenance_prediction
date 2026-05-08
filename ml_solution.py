import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


RANDOM_STATE = 50
ALERT_THRESHOLD = 0.05
df = pd.read_csv('ai4i2020.csv')

print('------ Dataset summary ------')
print(f'Shape: {df.shape}')
print('\nColumn types:')
print(df.dtypes)
print('\nMissing values:')
print(df.isna().sum())
print('\nTarget distribution:')
print(df['Machine failure'].value_counts())
print('\nStatistics:')
print(df.describe())

df['Type'] = df['Type'].apply(lambda x: ['L', 'M', 'H'].index(x))

features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
target = 'Machine failure'
X = df[features]
y = df[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
ids_val = X_val.index

models = {
    'LogisticRegression': LogisticRegression(
        solver='lbfgs',
        penalty='l2',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000
    ),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
}

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipe.fit(X_train, y_train)

    prob = pipe.predict_proba(X_val)[:, 1]
    pred = (prob >= 0.05).astype(int)

    print(f'\n------ Model: {name} ------')
    print(f'Accuracy: {accuracy_score(y_val, pred)}')
    print(f'ROC-AUC: {roc_auc_score(y_val, prob)}')
    print('Classification report (0.05 threshold):')
    print(classification_report(y_val, pred))

    report_df = pd.DataFrame(
        {
            'sample_id': ids_val,
            'y_true': y_val.values,
            'probability': prob,
            'alert': [bool(x) for x in pred]
        }
    )
    print(report_df.head())
    report_df.to_csv(f'submission{name}.csv', index=False)
