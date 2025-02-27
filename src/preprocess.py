import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path="data/processed.cleveland.data"):
    """
    Loads the Cleveland Heart Disease dataset from a CSV file.
    Returns a cleaned pandas DataFrame.
    
    The dataset has 14 columns:
      1) age       2) sex       3) cp        4) trestbps
      5) chol      6) fbs       7) restecg   8) thalach
      9) exang    10) oldpeak  11) slope    12) ca
     13) thal     14) num (target)
    """
    # Define column names for clarity
    col_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope",
        "ca", "thal", "num"  # 'num' is the original target in [0..4]
    ]
    
    # Read CSV, using col_names
    # processed.cleveland.data is often space- or comma-delimited. 
    # If you see errors, adjust delimiter accordingly.
    df = pd.read_csv(file_path, names=col_names, na_values="?")
    
    # Drop any rows with missing data
    df.dropna(inplace=True)
    
    # Convert numeric columns to float (if needed)
    # We'll assume all columns except 'sex', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal' 
    # can be floats. The categories can also be made floats for easier modeling,
    # or you can one-hot encode them. For now, let's just ensure they're all numeric.
    for col in col_names:
        df[col] = pd.to_numeric(df[col])
    
    # Binarize the target: 0 => no disease, 1 => disease
    # Originally, 'num' can be [0,1,2,3,4]; we map >0 to 1
    df["target"] = (df["num"] > 0).astype(int)
    
    # Optionally, drop the original 'num' column now that we have 'target'
    df.drop(columns=["num"], inplace=True)
    
    # Split data into train and test sets
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print(f"Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
