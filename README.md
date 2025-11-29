# CyberSentinel

## AI Development

### 📌Section 1 — Environment Setup

1. Installed Anaconda
You installed Anaconda, which provides:

- Python environment management
- Jupyter Notebook support
- Pre-installed ML libraries

2. Created a Dedicated Environment for the Project
Environment name: cybersentinel
`conda create -n cybersentinel python=3.9`
`conda activate cybersentinel`

3. Installed Required Libraries
You installed ML, data processing, and notebook libraries:

`pip install numpy pandas scikit-learn matplotlib seaborn jupyter notebook`
`pip install flask`
`pip install joblib`

---

### 📌Section 2 — Project Directory Setup

You created a structured folder layout:

    CyberSentinel/
    │── data/
    │     └── CICIDS2017/
    │           └── (all CSV dataset files)
    │── notebooks/
    │     └── 01_data_loading.ipynb
    │     └── 02_preprocessing.ipynb
    │── models/
    │── saved/
    │── app/
    │     └── (Flask/FastAPI backend later)

Dataset path used:
E:/Programing/CyberSentinel/data/CICIDS2017/

---

### 📌Section 3 — Dataset Preparation (CICIDS 2017)

Downloaded CICIDS 2017 CSV files

You downloaded the CSV files from:
[Index of /CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip](http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip)

Then extracted them into:
CyberSentinel/data/CICIDS2017/

**Quick Tip** (Launch Jupyter Notebook)

In Anaconda Prompt:

This is the most flexible way to temporarily start Jupyter in any directory or on any drive:

- Close the currently open Jupyter Notebook web page and the command prompt/terminal window it is running in or press ctrl + c twice in the terminal to shut it down.
- Open the Anaconda Prompt or a standard Command Prompt (CMD).
- Navigate to the E: drive by typing the drive letter followed by a colon and pressing Enter. `E:`
- `jupyter notebook`
- A browser window will open.
- Navigate to: E:\Programing\CyberSentinel\notebooks\

---

### 📌Section 4 — Notebook 01: Loading & Merging Dataset

File: 01_data_loading.ipynb

**🔹Code Used**
    
    import pandas as pd
    import os
    
    data_path = "E:/Programing/CyberSentinel/data/CICIDS2017/"
    files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    
    dfs = []
    for file in files:
        print("Loading:", file)
        df = pd.read_csv(data_path + file, low_memory=False)
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    
    print("Dataset shape:", data.shape)
    data.head()
    
    data.to_csv("E:/Programing/CyberSentinel/data/CICIDS2017/merged.csv", index=False)
    print("Merged dataset saved successfully!")
    
**🔹Output Achieved:**

Loading: merged.csv        
Dataset shape: (2830743, 79)        
Merged dataset saved successfully!        

---

### 📌Section 5 — Notebook 02: Data Preprocessing

File: 02_preprocessing.ipynb

#### 🔹Step 1 — Load the previously merged CSV

    import pandas as pd
    import numpy as np
    
    data = pd.read_csv("E:/Programing/CyberSentinel/data/CICIDS2017/merged.csv", low_memory=False)
    print("Loaded merged CSV:", data.shape)

**🔹Output Achieved:**

Loaded merged CSV: (1989801, 79)


#### 🔹Step 2 — Replace Infinite Values & Remove NaNs

CICIDS2017 contains invalid values like inf, -inf, and many missing rows.

    # Replace infinite values with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with missing values
    data.dropna(inplace=True)
    
    print("After removing NaN:", data.shape)

**🔹Output Achieved:**

After removing NaN: (1988305, 79)

#### 🔹Step 3 — Fix Column Names (remove extra spaces)

    # This step is important because CICIDS column names have leading spaces.
    data.rename(columns=lambda x: x.strip(), inplace=True)

#### 🔹Step 4 — Encode Labels (Convert Attack Names → Numbers)

    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    data["Label"] = le.fit_transform(data["Label"])
    
    print("Classes:", le.classes_)

#### 🔹Step 5 — Split Features & Labels

    X = data.drop("Label", axis=1)
    y = data["Label"]

    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)

#### 🔹Step 6 — Normalize Features (Scaling)

    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("X_scaled shape:", X_scaled.shape)


#### 🔹Extra Step (Not Doing Now)

##### Save the Preprocessed Data

At the end of preprocessing, add:

    data.to_csv("E:/Programing/CyberSentinel/data/CICIDS2017/preprocessed.csv", index=False)
    print("Preprocessed dataset saved!")

and for scaled features:

    import numpy as np
    np.save("E:/Programing/CyberSentinel/data/CICIDS2017/X_scaled.npy", X_scaled)
    np.save("E:/Programing/CyberSentinel/data/CICIDS2017/y.npy", y)

Then next time you restart Load directly:

    import pandas as pd
    import numpy as np

    data = pd.read_csv("E:/Programing/CyberSentinel/data/CICIDS2017/preprocessed.csv")
    X_scaled = np.load("E:/Programing/CyberSentinel/data/CICIDS2017/X_scaled.npy")
    y = np.load("E:/Programing/CyberSentinel/data/CICIDS2017/y.npy")

No preprocessing needed again.
Instant resume

---

### 📌Section 6 — You Are READY for Model Training

You have completed:

✔ Environment Setup    
✔ Dataset Download    
✔ Data Loading    
✔ Data Merging    
✔ Cleaning (NaN, inf values)    
✔ Column Fixing    
✔ Label Encoding    
✔ Feature/Label Splitting    
✔ Feature Scaling    

You are now ready to train:

- Random Forest (supervised)

- Isolation Forest (unsupervised)

- Hybrid Model (combination)

🎉 You have completed the entire Data Preparation & Preprocessing Pipeline — perfectly.

---
