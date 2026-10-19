"""Proje raporunu (Word) figures/ klasöründeki grafik ve tablolardan üretir."""
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FIG = "figures"
BLUE = RGBColor(0x1F, 0x4E, 0x79)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.2))
sec.top_margin = sec.bottom_margin = Cm(2)

st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(10.5)
st.paragraph_format.space_after = Pt(4)
st.paragraph_format.line_spacing = 1.1
for lvl, size in ((1, 14), (2, 12)):
    h = doc.styles[f"Heading {lvl}"]
    h.font.name, h.font.size, h.font.color.rgb = "Calibri", Pt(size), BLUE
    h.paragraph_format.space_before = Pt(10 if lvl == 1 else 6)
    h.paragraph_format.space_after = Pt(4)


def p(text, bold_prefix=None, italic=False, align=None, size=None):
    para = doc.add_paragraph()
    if bold_prefix:
        para.add_run(bold_prefix).bold = True
    r = para.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if align:
        para.alignment = align
    else:
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return para


def bullet(text, bold_prefix=None):
    para = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        para.add_run(bold_prefix).bold = True
    para.add_run(text)
    para.paragraph_format.space_after = Pt(2)
    return para


def figure(name, caption, width=15.5):
    doc.add_picture(f"{FIG}/{name}.png", width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = p(caption, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=9)
    c.paragraph_format.space_after = Pt(8)


def shade(cell, hex_color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def table(df, col_widths, highlight_row=None):
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(c))
        run.bold, run.font.size = True, Pt(9)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(cell, "1F4E79")
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(9)
            if highlight_row is not None and row.iloc[0] == highlight_row:
                run.bold = True
                shade(cells[i], "E2EFDA")
    for row in t.rows:
        for i, w in enumerate(col_widths):
            row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ---------------------------------------------------------------- sonuçlar
res = pd.read_csv(f"{FIG}/model_results.csv")
roi = pd.read_csv(f"{FIG}/betting_roi.csv")
R = res.set_index("Model")


def pct(x):
    return f"%{x * 100:.1f}"


# ---------------------------------------------------------------- başlık
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("At Yarışı Sonuç Sıralamasının Makine Öğrenmesi ile Tahmini")
r.bold, r.font.size, r.font.color.rgb = True, Pt(17), BLUE
s = doc.add_paragraph()
s.alignment = WD_ALIGN_PARAGRAPH.CENTER
s.add_run("Learning-to-Rank (LightGBM LambdaRank) ile Hong Kong Yarışlarında Bitiş Sırası Tahmini").italic = True
info = doc.add_paragraph()
info.alignment = WD_ALIGN_PARAGRAPH.CENTER
info.add_run("Makine ve Derin Öğrenme Uygulamaları – Proje Raporu  |  Kod: ").font.size = Pt(9.5)
link = info.add_run("https://github.com/Elif-Yildiz/at-yarisi-siralama-tahmini")
link.font.size = Pt(9.5)
link.font.color.rgb = BLUE

# ---------------------------------------------------------------- 1
doc.add_heading("1. Problem Tanımı", 1)
p("At yarışlarında bir yarışa katılan atların bitiş sırasını önceden tahmin etmek; bahis piyasası, "
  "at sahipleri ve antrenörler için yarış stratejisi, risk değerlendirmesi ve performans analizi "
  "açısından önemli bir problemdir. Bu projede girdi, bir yarışa katılan her atın yarıştan önce bilinen "
  "bilgileridir (at, jokey, antrenör, taşınan kilo, kulvar, handikap puanı, pist koşulları, ganyan oranı "
  "ve geçmiş performans). Hedef çıktı ise yarış içindeki bitiş sıralamasıdır. Problem, atların tek tek "
  "değil birbirlerine göre değerlendirildiği bir sıralama (learning-to-rank) problemi olarak ele "
  "alınmış; ayrıca kazananın tahmini ikili sınıflandırma, bitiş yeri tahmini ise regresyon bakış "
  "açısıyla da değerlendirilmiştir.")

# ---------------------------------------------------------------- 2
doc.add_heading("2. Veri Seti", 1)
p("Kaggle'da yayımlanan “Horse Racing in HK” veri seti kullanılmıştır "
  "(https://www.kaggle.com/datasets/gdaley/hkracing). Veri, Hong Kong Jokey Kulübü'nün Sha Tin (ST) ve "
  "Happy Valley (HV) hipodromlarında Haziran 1997 – Ağustos 2005 arasında koşulan yarışları içerir.", align=WD_ALIGN_PARAGRAPH.LEFT)
bullet(" 6.349 yarış (ST: 4.006, HV: 2.342), 37 sütun – tarih, hipodrom, mesafe, pist türü (çim/kum), "
       "pist durumu (going), yarış sınıfı, ödül.", "races.csv:")
bullet(" 79.447 koşu kaydı, 37 sütun – at/jokey/antrenör kimliği, yaş, menşe, handikap puanı, ekipman, "
       "at ağırlığı, taşınan kilo, kulvar (draw), ganyan oranı, bitiş yeri; 4.405 farklı at, 186 jokey.",
       "runs.csv:")
bullet(" Bitiş yeri (result). Sıralama etiketi olarak 1.→4, 2.→3, 3.→2, 4.→1, diğerleri→0 "
       "önem puanı (relevance) kullanılmıştır.", "Hedef değişken:")
bullet(" Bir yarışta ortalama ~12,5 at koştuğundan atların yalnızca %8,0'i kazanır; “kazanır” "
       "sınıfı belirgin şekilde azınlıktadır.", "Sınıf dengesizliği:")

# ---------------------------------------------------------------- 3
doc.add_heading("3. Veri Analizi ve Ön İşleme", 1)
doc.add_heading("3.1 Veri keşfi", 2)
p("Ganyan oranı ile gerçek kazanma oranı karşılaştırıldığında (Şekil 1) piyasanın atları çok tutarlı "
  "sıraladığı görülür: oran arttıkça kazanma oranı monoton biçimde düşer. Oranın ima ettiği olasılık "
  "(1/oran) her aralıkta gerçek orandan biraz yüksektir; bu fark bahis şirketinin komisyonudur (~%18). "
  "En düşük oranlı at (favori) yarışların %29,8'ini kazanmıştır; bu değer modeller için asıl çıtadır.")
figure("eda_odds_calibration", "Şekil 1 – Ganyan oranı aralığına göre gerçek ve ima edilen kazanma oranı", 13)
p("Kulvar etkisi (Şekil 2) özellikle dar virajlı Happy Valley pistinde belirgindir: iç kulvardan (1) "
  "çıkan atların kazanma oranı %11,9 iken 12. kulvarda %5,4'e düşer. Sha Tin'de etki daha zayıftır. "
  "Spearman korelasyonlarına göre bitiş yeriyle en güçlü ilişki ganyan oranındadır (ρ = 0,51); kulvar "
  "ρ = 0,12 ile ikinci sıradadır. Dikkat çekici bir bulgu, handikap puanının bitiş yeriyle neredeyse "
  "ilişkisiz olmasıdır (ρ ≈ 0). Bunun nedeni handikap sisteminin yüksek puanlı atlara daha fazla kilo "
  "yükleyerek yarışı dengelemesidir. Bu nedenle puanın tek başına değil, aynı yarıştaki rakiplere göre "
  "(göreli) kullanılması gerektiği sonucuna varılmıştır.")
figure("eda_draw", "Şekil 2 – Kulvar numarasına göre kazanma oranı (HV: Happy Valley, ST: Sha Tin)", 12)

doc.add_heading("3.2 Ön işleme ve özellik mühendisliği", 2)
bullet(" Yarış sırasında oluşan bilgiler (ara sıralamalar, ara zamanlar, bitiş zamanı, "
       "fark boyu) yarış öncesinde bilinmediği için modelden çıkarılmıştır.", "Veri sızıntısının önlenmesi:")
bullet(" Eksik kategorik değerler “Unknown” ile, eksik ödül değerleri (%7,4) medyan ile "
       "doldurulmuştur. Ganyan oranları 1–99 aralığına kırpılmıştır.", "Eksik ve aykırı değerler:")
bullet(" Her koşu için yalnızca önceki yarışlardan hesaplanan at/jokey/antrenör kazanma "
       "oranı, ilk-3 oranı, ortalama normalize bitiş yeri, son iki yarış sonucu, son yarıştan bu yana "
       "geçen gün, puan ve kilo değişimi. Az yarışı olanlar için Bayesyen düzeltme uygulanmıştır.",
       "Geçmiş performans özellikleri:")
bullet(" Sıralama bir yarış içi karşılaştırma olduğundan puan, kilo ve geçmiş "
       "performansın yarış ortalamasından farkı, oran sırası, piyasa olasılık payı ve göreli kulvar "
       "hesaplanmıştır. Ekipman alanından ekipman sayısı çıkarılmıştır. Toplam 45 özellik elde edilmiştir.",
       "Yarış içi göreli özellikler:")
bullet(" Logistic Regression ve Random Forest için sayısal değişkenlere StandardScaler, "
       "kategoriklere one-hot encoding uygulanmıştır. LightGBM kategorikleri doğrudan işler.",
       "Encoding ve ölçekleme:")
bullet(" Rastgele bölme geleceği sızdıracağı için yarışlar tarihe göre sıralanmıştır. Geçmiş "
       "bilgisinin oluşması için ilk 6 ay atılmış, kalan yarışların %70'i eğitim (4.114 yarış, "
       "1997–2003), %15'i doğrulama (882 yarış, 2003–2004) ve %15'i test (882 yarış, Tem 2004 – Ağu 2005) "
       "olarak ayrılmıştır.", "Zamana dayalı bölme:")

# ---------------------------------------------------------------- 4
doc.add_heading("4. Model / Yöntem", 1)
p("Dört yaklaşım aynı test setinde karşılaştırılmıştır:")
bullet(" Her yarışta en düşük oranlı atı 1. tahmin eder. Piyasanın kolektif bilgisini temsil "
       "eden güçlü bir temel modeldir.", "Piyasa favorisi (baseline):")
bullet(" Kazanma olasılığını (won = 1/0) tahmin eden doğrusal ve yorumlanabilir model. "
       "Sınıf dengesizliği için class_weight = “balanced”, C = 0,5 kullanılmıştır.", "Logistic Regression:")
bullet(" Normalize bitiş yerini (0 = birinci, 1 = sonuncu) tahmin eden, doğrusal "
       "olmayan ilişkileri yakalayabilen topluluk modeli. 300 ağaç, max_depth = 12, "
       "min_samples_leaf = 20 değerleri kullanılmıştır.", "Random Forest Regressor:")
bullet(" Asıl model. Her yarış bir grup olarak verilir ve gradient boosting "
       "ağaçları doğrudan NDCG sıralama metriğini optimize edecek şekilde eğitilir. Problem atların "
       "mutlak değil göreli başarısıyla ilgili olduğundan bu yaklaşım probleme en uygun olanıdır.",
       "LightGBM LambdaRank:")
p("Ayrıca ganyan oranı özellikleri çıkarılarak eğitilen bir LambdaRank modeli (ablasyon) ile modelin "
  "piyasa bilgisi olmadan, yalnızca at/jokey/antrenör geçmişi ve yarış koşullarından ne kadar "
  "öğrenebildiği ölçülmüştür.")

# ---------------------------------------------------------------- 5
doc.add_heading("5. Eğitim Süreci", 1)
p("LambdaRank modeli learning_rate = 0,03, num_leaves = 31, min_child_samples = 50, "
  "feature_fraction = bagging_fraction = 0,8 ve L2 = 1,0 ile eğitilmiştir. Ağaç sayısı en fazla 2.000 "
  "olarak belirlenmiş, doğrulama setinde NDCG@3 100 tur boyunca iyileşmediğinde erken durdurma "
  "uygulanmıştır. Model en iyi doğrulama skoruna 61. iterasyonda ulaşmıştır (doğrulama NDCG@1 = 0,445, "
  "NDCG@3 = 0,528). Şekil 3'te eğitim skoru artmaya devam ederken doğrulama skorunun erken dönemde "
  "plato yaptığı görülmektedir. Bu durum verideki gürültünün yüksek olduğunu ve modelin kolayca aşırı "
  "öğrenebileceğini gösterir. Hiperparametre seçimi yalnızca doğrulama seti kullanılarak yapılmış, test "
  "seti yalnızca son değerlendirmede kullanılmıştır.")
figure("lgbm_learning_curve", "Şekil 3 – LambdaRank öğrenme eğrileri (kesikli çizgi: erken durdurma noktası)", 14)

# ---------------------------------------------------------------- 6
doc.add_heading("6. Değerlendirme Sonuçları", 1)
p("Her test yarışında atlar model skoruna göre sıralanmış ve gerçek sıralamayla karşılaştırılmıştır. "
  "Kullanılan metrikler: Kazanan isabeti (top-1 accuracy), Top-3 isabeti (tahmin edilen ve gerçek ilk "
  "üçün kesişimi), NDCG@3, Spearman ρ ve tahmin edilen sıra ile gerçek sıra arasındaki MAE/RMSE. "
  "Sınıflandırma açısından her yarışta modelin 1. sıraya koyduğu at “Kazanır” olarak kabul edilmiştir.")
show = res[["Model", "Kazanan isabeti", "Top-3 isabeti", "Exacta", "NDCG@3", "Spearman ρ",
            "MAE (sıra)", "ROC-AUC"]].copy()
for c in ["Kazanan isabeti", "Top-3 isabeti", "Exacta"]:
    show[c] = show[c].map(pct)
for c in ["NDCG@3", "Spearman ρ", "MAE (sıra)", "ROC-AUC"]:
    show[c] = show[c].map(lambda v: f"{v:.3f}")
table(show, [4.2, 2.0, 1.9, 1.6, 1.6, 1.8, 1.7, 1.7], highlight_row="LightGBM LambdaRank")
p("Tablo 1 – Test seti sonuçları (882 yarış, 11.062 koşu).", italic=True,
  align=WD_ALIGN_PARAGRAPH.CENTER, size=9)
figure("model_comparison", "Şekil 4 – Modellerin test setindeki sıralama başarısı", 14.5)

lg, mk, no = R.loc["LightGBM LambdaRank"], R.loc["Piyasa favorisi"], R.loc["LambdaRank (oransız)"]
rnd = R.loc["Rastgele"]
p(f"Ganyan oranını kullanan üç model de piyasa favorisiyle hemen hemen aynı düzeydedir (kazanan "
  f"isabeti {pct(R['Kazanan isabeti'].iloc[2:5].min())}–{pct(R['Kazanan isabeti'].iloc[1:5].max())}, "
  f"NDCG@3 ≈ {lg['NDCG@3']:.2f}, ROC-AUC ≈ {lg['ROC-AUC']:.2f}). Rastgele tahminin kazanan isabeti "
  f"{pct(rnd['Kazanan isabeti'])} olduğundan, bu modeller kazananı rastgele tahmine göre yaklaşık "
  f"{lg['Kazanan isabeti'] / rnd['Kazanan isabeti']:.0f} kat daha sık bulur. Top-3 isabeti yaklaşık "
  f"%50'dir: tahmin edilen ilk üçün yarısı gerçekten ilk üçe girer. Oransız LambdaRank modeli ise "
  f"hiçbir piyasa bilgisi kullanmadan {pct(no['Kazanan isabeti'])} kazanan isabeti ve "
  f"{no['Spearman ρ']:.2f} Spearman korelasyonu elde etmiştir. Bu, at/jokey/antrenör geçmişi ve yarış "
  f"koşullarından anlamlı bir sinyal öğrenildiğini gösterir (rastgeleye göre ≈"
  f"{no['Kazanan isabeti'] / rnd['Kazanan isabeti']:.1f} kat).")
figure("confusion_matrices", "Şekil 5 – “Kazanır / Kazanmaz” karışıklık matrisleri (test seti)", 12.5)
p(f"Karışıklık matrisinde her yarışta tam olarak bir at “Kazanır” olarak işaretlendiği için precision, "
  f"recall ve F1 değerleri kazanan isabetine eşittir (LambdaRank: {lg['F1 (kazanır)']:.3f}). At bazındaki "
  f"accuracy (%{lg['Accuracy (at bazında)'] * 100:.1f}) ise “Kazanmaz” sınıfının baskınlığından dolayı "
  f"yüksek görünür. Rastgele modelin bile accuracy değeri %{rnd['Accuracy (at bazında)'] * 100:.1f}'dir. "
  f"Bu nedenle bu problemde accuracy tek başına yanıltıcıdır ve sıralama metrikleri esas alınmalıdır.")

# ---------------------------------------------------------------- 7
doc.add_heading("7. Hata Analizi", 1)
figure("feature_importance", "Şekil 6 – LambdaRank modelinde en önemli 20 özellik (bilgi kazancı)", 11)
p("Özellik önemleri (Şekil 6), modelin kararlarının büyük ölçüde piyasa olasılığı ve ganyan oranına "
  "dayandığını gösterir. Geçmiş performans özellikleri (göreli ortalama bitiş yeri, jokey deneyimi, "
  "antrenör kazanma oranı), kulvar ve göreli kilo ise ikincil katkı sağlar. Ganyan oranı, kamuya açık "
  "bilgilerin büyük kısmını zaten içerdiğinden diğer özelliklerin ek bilgi değeri sınırlı kalmaktadır. "
  "Modelin piyasayı geçememesinin temel nedeni budur.")
figure("error_analysis", "Şekil 7 – LambdaRank kazanan isabetinin kırılımları (n = yarış sayısı)", 16)
p("Şekil 7'de görüldüğü gibi, model gerçek kazananın oranı 3'ün altında olduğu yarışlarda kazananı "
  "%95 oranında doğru bulurken, oranı 6'nın üzerinde olan atların kazandığı yarışlarda neredeyse hiç "
  "isabet edememektedir. Test yarışlarının %25,3'ünde kazanan oranı 10'un üzerinde olan bir “sürpriz” "
  "attır. Bu tür sonuçlar yarış içi şans faktörlerinden kaynaklanır (start, sıkışma, tempo, atın o "
  "günkü formu) ve yarış öncesi verilerle öngörülmeleri çok zordur. Kalabalık yarışlarda (14 at) "
  "isabet %24'e düşer. Yarış sınıfına göre güçlü bir eğilim yoktur; 0, 6 ve 11–13 sınıflarındaki "
  "yüksek/düşük değerler az sayıda yarıştan geldiği için güvenilir değildir.")
r0 = roi.set_index("Model")
p(f"Ek bir analiz olarak, her test yarışında modelin 1. sıraya koyduğu ata 1 birim ganyan bahsi "
  f"yapıldığı simüle edilmiştir. Tüm stratejiler zarar etmiştir: piyasa favorisinin getirisi "
  f"%{r0.loc['Piyasa favorisi', 'ROI'] * 100:.1f}, LambdaRank modelinin getirisi "
  f"%{r0.loc['LightGBM LambdaRank', 'ROI'] * 100:.1f}. Bu sonuç, bahis piyasasının verimli olduğunu ve "
  f"komisyon (~%18) aşılmadan kâr elde edilemeyeceğini göstermektedir.")

# ---------------------------------------------------------------- 8
doc.add_heading("8. Sonuç ve Geliştirme Önerileri", 1)
p("Bu projede at yarışı bitiş sıralaması bir learning-to-rank problemi olarak modellenmiş; veri "
  "sızıntısını önleyen geçmiş performans özellikleri ve zamana dayalı doğrulama ile LightGBM LambdaRank "
  "modeli geliştirilmiştir. Modeller rastgele tahmine göre kazananı yaklaşık 4 kat daha sık bulmuş ve "
  "ilk üçün yaklaşık yarısını doğru tahmin etmiştir. Ancak ganyan oranı kullanıldığında piyasa "
  "favorisinin önüne geçilememiştir.")
p("Güçlü yönler: ", italic=False).runs[0].bold = True
doc.paragraphs[-1].add_run("sızıntısız ve zamana dayalı tasarım, sıralamaya özgü kayıp fonksiyonu, "
                           "yorumlanabilir özellik önemleri ve piyasa bilgisi olmadan da anlamlı "
                           "öğrenme (oransız modelde %24 kazanan isabeti).")
p("Zayıf yönler: ").runs[0].bold = True
doc.paragraphs[-1].add_run("modelin ganyan oranına aşırı bağımlı olması, sürpriz kazananları "
                           "öngörememesi ve yarış içi olayların (start, tempo, sıkışma) verisinin "
                           "bulunmaması.")
p("Geliştirme önerileri:").runs[0].bold = True
bullet(" Ara sıralama verilerinden atın koşu stilini (önde/geride koşan) çıkarıp yarışın "
       "beklenen temposunu modellemek.", "Koşu stili ve tempo:")
bullet(" Elo/TrueSkill benzeri dinamik güç puanları ile at-jokey ve at-mesafe-pist "
       "etkileşimlerini eklemek.", "Dinamik güç puanları:")
bullet(" Kazanma olasılığını conditional logit / Plackett-Luce ile modelleyip piyasa "
       "olasılığıyla birleştirmek (Benter yaklaşımı).", "Olasılık modeli:")
bullet(" Optuna ile hiperparametre araması yapmak ve zaman serisi çapraz doğrulama (walk-forward) "
       "ile sonuçların kararlılığını ölçmek.", "Hiperparametre ve doğrulama:")
bullet(" Daha güncel yarışları (ör. TJK verisi) ekleyerek modelin farklı pistlere "
       "genellenip genellenmediğini test etmek.", "Güncel veri:")

cp = doc.core_properties
cp.author = cp.last_modified_by = cp.comments = ""
cp.title = "At Yarışı Sonuç Sıralamasının Makine Öğrenmesi ile Tahmini"
doc.save("At_Yarisi_Siralama_Tahmini_Rapor.docx")
print("Rapor kaydedildi.")
