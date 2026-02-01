from sklearn.preprocessing import LabelEncoder

def encode_categorical(df, categorical_columns):
    encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])
    return df
