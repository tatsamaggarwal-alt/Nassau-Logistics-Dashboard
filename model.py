import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(df):
    df = df.copy()

    # Convert categorical
    df['Ship Mode'] = df['Ship Mode'].astype('category').cat.codes
    df['Region'] = df['Region'].astype('category').cat.codes

    X = df[['Ship Mode', 'Region', 'Lead Time']]
    y = df['Delayed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model
