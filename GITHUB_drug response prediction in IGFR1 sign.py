# drug response prediction in IGFR1 signalling pathways 
# USE JUPITER NOTEBOOK or COLAB and execute in Boxe subsets #------
# Modelling and Prediction
#Modelling and Prediction_1
import pandas as pd
import numpy as np
#-------------------------------------------
df1=pd.read_csv('IGFRmaster.csv')
#--------------------------------------
# Describe desired data
df1
df1.describe().T
#---------------------------------------
# Select features and assign them to seperate variables based on their datatype (categorical/continuous/ Binary)
# This is done to avoid unneccesary normalization of correct features like (binary/ categorical data)
from sklearn.preprocessing import OneHotEncoder
selected = ['CELL_LINE', 'DRUG_NAME', 'PUTATIVE_TARGET', 'PATHWAY_NAME', 'AUC', 'LN_IC50','CLASS']
# select and drop any datset and dont change beyond this code and follow # numbers and dont touch / run any other tab
selected = ['LN_IC50','Location'] #2
df = df1.drop(selected, axis=1)
df
category = df[['NHOHCount', 'NOCount', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
                 'NumRotatableBonds', 'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
                 'NumAromaticHeterocycles', 'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles', 'RingCount',
                 'Veber_Passes', 'Ghose_Passes', 'Muegge_Passes', 'Ro3_Passes', 'Egan_Passes', 'Ro2_Passes',
                 'Lipinski_Passes', 'NHOHCount_skf', 'NumHAcceptors_skf', 'NumHDonors_skf', 'NumHeteroatoms_skf',
                 'NumRotatableBonds_skf', 'NumValenceElectrons_skf', 'NumAromaticRings_skf',
                 'NumSaturatedRings_skf', 'NumAliphaticRings_skf', 'NumAromaticHeterocycles_skf',
                 'NumSaturatedHeterocycles_skf', 'NumAliphaticHeterocycles_skf', 'RingCount_skf']]
#-------------------------------------------

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Instantiate MinMaxScaler
scaler = MinMaxScaler()

# Identify binary columns (columns with only 0 and 1 values)
binary_columns = [col for col in df.columns if set(df[col]) == {0, 1}]

# Columns to be normalized (excluding binary columns and 'CLASS')
columns_to_normalize = [col for col in df.columns if col not in binary_columns and col != 'CLASS']

# Exclude non-numeric columns and 'CLASS' column from normalization
numeric_columns = df[columns_to_normalize].select_dtypes(include=['number']).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Now, numeric columns are normalized (excluding 'CLASS'), and binary columns are identified.
#-------------------------------------------
# Display information about NaN values in the remaining columns #3
nan_info = df.isnull().sum().sort_values(ascending=False)
print(nan_info.head(50))
#-------------------------------------------
# Finalising Dataset preprocessing and Spliting for Test/ Train
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,classification_report
# change your Target below
X = df.drop('newcategory', axis=1) #5
y = df['newcategory'].astype('category')
# Extract features (excluding non-numeric columns)
X = df.select_dtypes(include=['float64', 'int64'])
label_encoder = LabelEncoder()
df['Location'] = label_encoder.fit_transform(df['Location'])
columns_to_drop = ['Location', 'CELL_LINE', 'DRUG_NAME', 'PUTATIVE_TARGET', 'PATHWAY_NAME', 'AUC', 'LN_IC50']
X = df.drop(columns_to_drop, axis=1)
from sklearn.model_selection import train_test_split #6
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,)
#-------------------------------------------

# DECISION TREES
from sklearn.tree import DecisionTreeClassifier
DTmodel=DecisionTreeClassifier() #training
DTmodel.fit(X_train,Y_train)
Y_pred_DT=DTmodel.predict(X_test) #Predict
print("Confusion Martix:\n",confusion_matrix(Y_test,Y_pred_DT))
print("f1_score:\n",f1_score(Y_test,Y_pred_DT,average='macro'))
print("Accuruacy:\n",accuracy_score (Y_test,Y_pred_DT))
print("Classification Report: \n", classification_report(Y_test,Y_pred_DT))

#-------------------------------------------

# RANDOM FOREST
#Random Forest
from sklearn.ensemble import RandomForestClassifier
RFmodel=RandomForestClassifier()
RFmodel.fit(X_train,Y_train)
Y_pred_RF=RFmodel.predict(X_test)
print("Confusion matrix:\n",confusion_matrix(Y_test,Y_pred_RF))
print("F1 Score:\n",f1_score(Y_test,Y_pred_RF,average='macro'))
print("Accuracy:\n", accuracy_score(Y_test,Y_pred_RF))
print("Classification Report:\n",classification_report(Y_test,Y_pred_RF))

#-------------------------------------------
# XAI explaianation using SHAP on Random Forest model
import shap

explainer = shap.TreeExplainer(RFmodel)
shap_values = explainer.shap_values(X_test)

#-------------------------------------------
# ADA, Gradient, HistGradient BOOSTING
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier
ABModel = AdaBoostClassifier()
ABModel.fit(X_train, Y_train)


GBModel = GradientBoostingClassifier()
GBModel.fit(X_train, Y_train)


HGBModel = HistGradientBoostingClassifier()
HGBModel.fit(X_train, Y_train)
Y_pred_AB = ABModel.predict(X_test)
Y_pred_GB = GBModel.predict(X_test)
Y_pred_HGB = HGBModel.predict(X_test)
print("Confusion matrix of Ada Boost:\n",confusion_matrix(Y_test,Y_pred_AB))
print("F1 Score of Ada Boost:\n",f1_score(Y_test,Y_pred_AB,average='macro'))
print("Accuracy of Ada Boost:\n", accuracy_score(Y_test,Y_pred_AB))
print("Classification Report of Ada Boost:\n",classification_report(Y_test,Y_pred_AB))
print("---------------------------------------------------------")
print("Confusion matrix of Gradient Boost:\n",confusion_matrix(Y_test,Y_pred_GB))
print("F1 Score of Gradient Boost:\n",f1_score(Y_test,Y_pred_GB,average='macro'))
print("Accuracy of Gradient Boost:\n", accuracy_score(Y_test,Y_pred_GB))
print("Classification Report of Gradient Boost:\n",classification_report(Y_test,Y_pred_GB))
print("---------------------------------------------------------")
print("Confusion matrix of HistGradient Boost:\n",confusion_matrix(Y_test,Y_pred_HGB))
print("F1 Score of HistGradient Boost:\n",f1_score(Y_test,Y_pred_HGB,average='macro'))
print("Accuracy of HistGradient Boost:\n", accuracy_score(Y_test,Y_pred_HGB))
print("Classification Report of HistGradient Boost:\n",classification_report(Y_test,Y_pred_HGB))
print("---------------------------------------------------------")
#-------------------------------------------

# XG BOOST
import xgboost as xgb

# Initialize and train the XGBoost classifier
XGBModel = xgb.XGBClassifier()
XGBModel.fit(X_train, Y_train)
Y_pred_XGB = XGBModel.predict(X_test)
print("Confusion matrix of Y_pred_XG Boost:\n",confusion_matrix(Y_test,Y_pred_XGB))
print("F1 Score of Y_pred_XG Boost:\n",f1_score(Y_test,Y_pred_XGB,average='macro'))
print("Accuracy of Y_pred_XG Boost:\n", accuracy_score(Y_test,Y_pred_XGB))
print("Classification Report of Y_pred_XG Boost:\n",classification_report(Y_test,Y_pred_XGB))
print("---------------------------------------------------------")
#-------------------------------------------

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

# Initialize and train the Gaussian Naive Bayes classifier
NBModel = GaussianNB()
NBModel.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred_NB = NBModel.predict(X_test)

# Evaluate the performance of the model
print("Confusion matrix of Naive Bayes:\n", confusion_matrix(Y_test, Y_pred_NB))
print("F1 Score of Naive Bayes:\n", f1_score(Y_test, Y_pred_NB, average='macro'))
print("Accuracy of Naive Bayes:\n", accuracy_score(Y_test, Y_pred_NB))
print("Classification Report of Naive Bayes:\n", classification_report(Y_test, Y_pred_NB))

#-------------------------------------------
# Clustering 
# Tsne and UMAP visualization
pip install --upgrade pip
import umap
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.base import BaseEstimator, TransformerMixin

# Assuming df1 has LN_IC50 values
df4 = df
df4['LN_IC50'] = df1['LN_IC50']

# Define different colors for t-SNE
tsne_colors = plt.cm.viridis(np.linspace(0, 1, len(df4['LN_IC50'].unique())))
tsne_color_dict = dict(zip(sorted(df4['LN_IC50'].unique()), tsne_colors))

# Separate features and target variable for t-SNE
X_tsne = df4.drop('LN_IC50', axis=1)
y_tsne = df4['LN_IC50']

# Convert y_tsne to integers for using as indices
y_tsne_int = y_tsne.astype(int)

# Use t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
X_tsne_transformed = tsne.fit_transform(X_tsne)

# Visualize the t-SNE-transformed data
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne_transformed[:, 0], X_tsne_transformed[:, 1], c=[tsne_color_dict[label] for label in y_tsne], marker='o', edgecolors='k')
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

# Define different colors for UMAP
umap_colors = plt.cm.viridis(np.linspace(0, 1, len(df4['LN_IC50'].unique())))
umap_color_dict = dict(zip(sorted(df4['LN_IC50'].unique()), umap_colors))

# Separate features and target variable for UMAP
X_umap = df4.drop('LN_IC50', axis=1)
y_umap = df4['LN_IC50']

# Use UMAP for dimensionality reduction
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap_transformed = umap_model.fit_transform(X_umap)

# Visualize the UMAP-transformed data without legend
plt.figure(figsize=(10, 6))
for class_label in sorted(y_umap.unique()):
    class_indices = (y_umap == class_label)
    plt.scatter(X_umap_transformed[class_indices, 0], X_umap_transformed[class_indices, 1],
                c=[umap_color_dict[class_label]], marker='o', edgecolors='k')

plt.title('UMAP Visualization')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.show()


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Assuming X_umap_transformed is your UMAP-transformed data
#----------------------------------------------------------------------
# Look at the end to know HOw this K values was found
#----------------------------------------------------------------------
# Set the number of clusters (K) to 4
k = 4

# Perform K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(X_umap_transformed)

# Visualize the clustered data with x, y scale legends
plt.figure(figsize=(10, 6))
for cluster_label in range(k):
    cluster_indices = (cluster_labels == cluster_label)
    plt.scatter(X_umap_transformed[cluster_indices, 0], X_umap_transformed[cluster_indices, 1],
                label=f'Cluster {cluster_label}', alpha=0.7)

df4['newcategory'] = cluster_labels
plt.title('K-Means Clustering with x, y Scale Legends (K=4)')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.legend(title='Cluster')
plt.show()

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming X_umap_transformed is your UMAP-transformed data
# Assuming optimal_k is the optimal number of clusters obtained

# Perform K-means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df4[' '] = kmeans.fit_predict(X_umap_transformed)

# Create new columns in df4 for each cluster
for cluster_label in range(optimal_k):
    cluster_indices = (df4['Cluster'] == cluster_label)
    df4[f'Cluster_{cluster_label}_UMAP_1'] = np.nan
    df4[f'Cluster_{cluster_label}_UMAP_2'] = np.nan
    df4.loc[cluster_indices, f'Cluster_{cluster_label}_UMAP_1'] = X_umap_transformed[cluster_indices, 0]
    df4.loc[cluster_indices, f'Cluster_{cluster_label}_UMAP_2'] = X_umap_transformed[cluster_indices, 1]

# Visualize the clustered data with x, y scale legends
plt.figure(figsize=(10, 6))
for cluster_label in range(optimal_k):
    cluster_indices = (df4['Cluster'] == cluster_label)
    plt.scatter(X_umap_transformed[cluster_indices, 0], X_umap_transformed[cluster_indices, 1],
                label=f'Cluster {cluster_label}', alpha=0.7)

plt.title('K-Means Clustering with x, y Scale Legends')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.legend(title='Cluster')
plt.show()

df['Location'] = cluster_labels
df.to_csv('IGFRlocated.csv',index=False)
df['Location'].unique()
# Optimal clustering value of K
!pip install yellowbrick
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#-------------------------------------------#-------------------------------------------
#-------------------------------------------#-------------------------------------------
# The Below Code is used to find the optimal value of K aand vizualize the dataset clusterd based on that

# Use the Elbow Method to find the optimal K
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2, 10))
visualizer.fit(X_scaled)
visualizer.show()

!pip install gap-stat
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from gap_statistic import OptimalK
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score

# Load a sample dataset (Iris dataset)
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Elbow Method using Yellowbrick
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2, 10))
visualizer.fit(X_scaled)
visualizer.show()
optimal_k_elbow = visualizer.elbow_value_

# Silhouette Score
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette_avg)

# Optimal K using Silhouette Score
optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2

# Gap Statistics
optimalK = OptimalK(parallel_backend="joblib")  # Use joblib for parallel processing
optimal_k_gap = optimalK(X_scaled, cluster_array=range(1, 11))

# Print the results
print(f'Optimal K using Elbow Method: {optimal_k_elbow} clusters')
print(f'Optimal K using Silhouette Score: {optimal_k_silhouette} clusters')
print(f'Optimal K using Gap Statistics: {optimal_k_gap} clusters')

# Print the results
print(f'Optimal K using Elbow Method: {optimal_k_elbow} clusters')
print(f'Optimal K using Silhouette Score: {optimal_k_silhouette} clusters')
print(f'Optimal K using Gap Statistics: {optimal_k_gap} clusters')
#-------------------------------------------
# The below are some experiments and visualizations using SHAP library
# SHAP and LIME
!pip install SHAP
import shap
import pandas as pd
import numpy as np
shap.initjs()

explainer = shap.Explainer(XGBModel)
shap_values = explainer.shap_values(X_test)
import matplotlib.pyplot as plt

# Display each plot in a separate row
plt.figure(figsize=(8, 20))

# Plot for dot plot 1
plt.subplot(4, 1, 1)
shap.summary_plot(shap_values[0], X_test, plot_type='dot', show=False)
plt.title('Dot Plot 1')

# Plot for dot plot 2
plt.subplot(4, 1, 2)
shap.summary_plot(shap_values[1], X_test, plot_type='dot', show=False)
plt.title('Dot Plot 2')

# Plot for dot plot 3
plt.subplot(4, 1, 3)
shap.summary_plot(shap_values[2], X_test, plot_type='dot', show=False)
plt.title('Dot Plot 3')

# Plot for dot plot 4
plt.subplot(4, 1, 4)
shap.summary_plot(shap_values[3], X_test, plot_type='dot', show=False)
plt.title('Dot Plot 4')

plt.tight_layout(pad=3.0)
plt.show()

shap.summary_plot(shap_values[0], X_test, plot_type='dot', show=False)
plt.title('Bee Swarm Plot')
shap.plots.force(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0, :], matplotlib = True)
X_test
shap.summary_plot(shap_values[0], X_test)
# Convert SHAP values to a DataFrame
shap_summary_df = pd.DataFrame(shap_values[0], columns=X_test.columns)

# Display the summary statistics
summary_stats = shap_summary_df.describe()
print(summary_stats)