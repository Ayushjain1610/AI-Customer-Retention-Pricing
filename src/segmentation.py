from sklearn.cluster import KMeans

def segment_customers(X, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    segments = model.fit_predict(X)
    return segments, model
