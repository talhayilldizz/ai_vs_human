import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# ----------------- Veri ve Model -----------------
df=pd.read_csv("AI Generated Essays Dataset.csv")
# print(df.head())
df = df.rename(columns={'generated': 'label'})

nltk.download("stopwords")
stop_words=set(stopwords.words("english"))

#Metinleri Temizleme İşlemi
def clean_text(text):
    text=text.lower() #Büyük harfleri küçük harfe dönüştürdüm
    text=" ".join(text.split()) #Gereksiz boşlukları çıkardım
    text=text.translate(str.maketrans("","",string.punctuation)) #Noktalama işaretlerini kaldırdım
    text=re.sub(r"[^A-Za-z0-9\s]","",text) #Özel karakter varsa kaldırdım
    text=' '.join([word for word in text.split() if word not in stop_words]) #stopwordsleri kaldırdım

    return text

df["clean_text"]=df["text"].apply(clean_text)
print(df.head())

X=df["clean_text"]
y=df["label"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
vectorizer=TfidfVectorizer(max_features=500)
X_train_tfidf=vectorizer.fit_transform(X_train)
X_test_tfidf=vectorizer.transform(X_test)


models={
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42)
}

results={}

for name,model in models.items():
    model.fit(X_train_tfidf,y_train)
    y_pred=model.predict(X_test_tfidf)
    acc=accuracy_score(y_test, y_pred)
    print(f"--- {name} ---")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
    results[name] = acc


model_names = list(models.keys())
num_models = len(model_names)
cols = 2
rows = (num_models + 1) // cols  
fig, axes = plt.subplots(rows, cols, figsize=(12, 5*rows))
axes = axes.flatten()
for i, name in enumerate(model_names):
    y_pred = models[name].predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False, ax=axes[i])
    axes[i].set_title(f'{name}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

for j in range(num_models, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()


# ----------------- Streamlit Tasarımı -----------------
st.markdown("<h1 style='text-align: center; color: #FFFFF;'>AI vs Human</h1>", unsafe_allow_html=True)
st.image("background.jpg")
st.markdown("<p style='text-align: center; color: gray;'>Bir metnin AI tarafından mı yoksa insan tarafından mı yazıldığını tahmin edin.</p>", unsafe_allow_html=True)
sample=st.text_area("Metninizi Girin: ",height=150)

def predict(text):
    text=clean_text(text)
    vectorized=vectorizer.transform([text])
    proba=models["SVM"].predict_proba(vectorized)[0]
    pred=np.argmax(proba)
    confidence=proba[pred]*100
    return ("AI" if pred == 1 else "HUMAN"), confidence

if st.button("Gönder"):
    if sample:
        result, confidence = predict(sample)
        st.write(f"<h3>Bu metin %{confidence:.2f} ihtimalle <b>**{result}**</b> tarafından yazılmıştır</h3>",unsafe_allow_html=True)
    else:
        st.write("Lütfen metin giriniz..")
