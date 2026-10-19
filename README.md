# At Yarışı Sonuç Sıralaması Tahmini

Hong Kong at yarışı verisiyle (1997–2005) yarış bitiş sıralamasını tahmin eden bir learning-to-rank projesi.
Piyasa favorisi, Logistic Regression ve Random Forest temel modelleri LightGBM LambdaRank ile karşılaştırılır.

## Sonuçlar (test seti: 882 yarış)

| Model | Kazanan isabeti | Top-3 isabeti | NDCG@3 |
|---|---|---|---|
| Rastgele | %7,3 | %23,4 | 0,240 |
| Piyasa favorisi | %30,6 | %50,3 | 0,550 |
| Logistic Regression | %29,7 | %49,5 | 0,544 |
| Random Forest | %28,7 | %49,8 | 0,540 |
| LightGBM LambdaRank | %28,8 | %49,9 | 0,544 |
| LambdaRank (ganyan oranı olmadan) | %24,0 | %44,2 | 0,480 |

Ayrıntılı analiz ve yorumlar: [At_Yarisi_Siralama_Tahmini_Rapor.pdf](At_Yarisi_Siralama_Tahmini_Rapor.pdf)

## Dosyalar

- `at_yarisi_siralama.ipynb` – veri keşfi, ön işleme, modeller, değerlendirme ve hata analizi (çıktılarıyla)
- `at_yarisi_siralama.py` – notebook'un kaynak hali (`# %%` hücreleri)
- `figures/` – grafikler ve sonuç tabloları
- `rapor_olustur.py` – Word raporunu üretir
- `py_to_ipynb.py` – `.py` → `.ipynb` dönüştürücü

## Veri

Kaggle: https://www.kaggle.com/datasets/gdaley/hkracing
Veri lisans gereği depoya eklenmemiştir. `races.csv` ve `runs.csv` dosyalarını `data/` klasörüne koyun.

## Çalıştırma

```bash
pip install pandas scikit-learn lightgbm matplotlib seaborn nbformat nbconvert ipykernel python-docx
python py_to_ipynb.py at_yarisi_siralama.py at_yarisi_siralama.ipynb
jupyter nbconvert --to notebook --execute --inplace at_yarisi_siralama.ipynb
```
