# AI vs Human 

Bu proje, bir metnin **AI tarafından mı yoksa insan tarafından mı yazıldığını tahmin eden** bir makine öğrenmesi uygulamasıdır. Python, Scikit-learn ve Streamlit kullanılarak geliştirilmiştir.  

![background](background.jpg)

## Dataset

Projede kullanılan veri seti [Kaggle: AI Generated Essays Dataset](https://www.kaggle.com/datasets/denvermagtibay/ai-generated-essays-dataset/data) adresinden alınmıştır. Dataset, AI tarafından üretilmiş ve insan tarafından yazılmış çeşitli essay (deneme) metinlerini içerir.  

- **Sütunlar:**
  - `text`: Metin içeriği
  - `generated` (rename edilerek `label` oldu): Metnin AI tarafından üretilip üretilmediği bilgisi (0 = Human, 1 = AI)

## Özellikler

- Metin temizleme (lowercase, noktalama kaldırma, stopword temizleme)  
- TF-IDF ile metinleri sayısal vektörlere dönüştürme  
- Çeşitli sınıflandırma modelleri ile tahmin:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - SVM (Linear Kernel)  

- Confusion matrix görselleştirmesi (her model için)  
- Streamlit tabanlı kullanıcı arayüzü  

## Model Seçimi

Projede dört farklı sınıflandırma modeli denendi: Logistic Regression, Random Forest, Gradient Boosting ve SVM (Linear Kernel).  

Test sonuçlarına göre:  

- **SVM**, hem **accuracy** hem de **precision, recall ve f1-score** açısından en yüksek performansı gösterdi.  
- Özellikle, AI tarafından üretilmiş metinlerin tespitinde yüksek doğruluk sağladı ve sınıflar arasındaki dengesizlikte iyi sonuç verdi.  
- Bu nedenle, Streamlit arayüzünde tahmin yapmak için **SVM modeli** tercih edildi.  

Özetle, SVM modeli hem doğruluk hem de kararlılık açısından diğer modellere göre en uygun seçenektir.  

Aşağıdaki görselde, denediğim modellerin tahmin performansları gösterilmiştir

![performans](Model%20Performansları.png)
