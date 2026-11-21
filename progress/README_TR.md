# CS 415 — Progress report çalışma paketi

Hazırlanma tarihi: 16 Kasım 2026. Ana rapor, öneriyle aynı dilde (İngilizce), gerçek `IEEEtran` conference LaTeX sınıfıyla çift sütunlu olarak hazırlanmıştır. Özgün öneri değiştirilmemiştir.

## Dosyalar

- `progress_report.pdf`: değerlendirme için hazırlanmış ilerleme raporu.
- `progress_report.tex`: düzenlenebilir ana LaTeX kaynağı.
- `references.bib`: doğrulanmış yayın kayıtları; 13 çalışma ve 1 veri kümesi kaynağı.
- `progress_overleaf.zip`: Overleaf'e aktarılabilen küçük LaTeX paketi. Ham veri ve Python ortamını içermez.
- `IEEEtran.cls`: CTAN'dan alınan IEEE conference sınıfı.
- `figures/`: vektörel PDF ve PNG grafikler.
- `results/`: ham deney sonuçları, yapılandırma, sayısal doğrulamalar ve otomatik üretilen LaTeX tabloları.
- `scripts/`: deney, rapor görselleri ve kontrol kodları.
- `data/`: indirilen Fine Foods arşivi, temizlenmiş veri ve satır düzeyindeki bölme manifesti.
- `sources/`: verilen önerinin kopyası, önceki küçük denetimler ve yöntem kontrolünde kullanılan kaynak dosyası.
- `build/`: derleme kayıtları ve sayfa görselleri.

## Gerçekleşen çalışma ve sınırlar

Bu çalışma sırasında 568.454 kaydın tamamı indirildi ve denetlendi. 119 normalize metin grubunda farklı puanlar vardı; bu gruplardaki 428 kayıt çıkarıldı. Sonra 175.025 aynı puanlı tekrar çıkarıldı. Kalan 393.001 kayıt ürün bazında ayrıldı:

| Bölüm | Yorum | Ürün |
|---|---:|---:|
| Eğitim | 276.844 | 47.394 |
| Doğrulama | 56.791 | 9.892 |
| Ayrılmış test | 59.366 | 10.199 |

İki TF–IDF doğrusal sınıflandırıcı, 17/29/43 tohumlarıyla gerçekten eğitildi. Ağırlıksız ve sınıf ağırlıklı modellerin her biri için üç sonuç ve tahmin dosyaları saklandı. Veriler yapay üretilmiş sonuçlar değildir. Ancak bunlar **TextCNN veya DistilBERT sonuçları değildir**. Bu modeller ve CE/OLL/MLL karşılaştırması henüz çalıştırılmadı. Rapor bunu açıkça belirtir. Kayıp fonksiyonu örnek tablosu, seçilmiş olasılık vektörleriyle hesaplanan teorik bir gösterimdir; eğitilmiş model deneyi değildir.

Rapor doğrulama kümesi performansını verir. Ayrılmış test kümesi için model tahminleri veya performans sonuçları hesaplanmadı. Testin yalnızca sayımları ve sabit veri temizliği kontrol edildi. Bu ayrım son deneylerde korunmalıdır.

Sınıf ağırlıklı temel modelin üç koşu ortalaması: accuracy %73,54; macro-F1 0,5376; MAE 0,3956; en az iki yıldız hata oranı %7,75. Bu kazanımlar sınıf ağırlıklandırmasıyla ilgilidir ve önerideki ordinal kayıp hipotezinin doğrulandığı anlamına gelmez.

## PDF'yi yeniden derleme

Bu bilgisayarda Tectonic kuruldu:

```bash
cd '/Users/efehankeles/Desktop/2026-2027 FALL/CS 415/project folders/progress'
bash build_report.sh
```

Grafik ve tablolar mevcutsa Python deneylerini tekrar çalıştırmak gerekmez. Overleaf'te `progress_overleaf.zip` dosyasını yeni proje olarak içe aktarın ve `progress_report.tex` dosyasını ana belge seçin. Standart pdfLaTeX/BibTeX derlemesi desteklenir; yerel çıktı Tectonic ile derlenmiştir.

## Deneyleri yeniden üretme

`.venv` bu klasörde hazırlanmış yerel Python ortamıdır. Mevcut ortamla:

```bash
.venv/bin/python scripts/run_experiments.py > results/run.log 2>&1
.venv/bin/python scripts/verify_evidence.py
.venv/bin/python scripts/make_report_assets.py
bash build_report.sh
```

Bu komutlar önceki sonuç dosyalarını yeniden üretir/üzerine yazar. Süreler donanıma ve o anda çalışan işlere göre değişir. Paket sürümleri `requirements.txt` içinde, gerçek koşu ortamı `results/experiment_config.json` içinde bulunur. Başka bilgisayarda Python sanal ortamı oluşturup requirements dosyasından kurulum yapın. `run_experiments.py` zamanlama bilgisinde macOS `sysctl` kullanır; başka işletim sisteminde CPU tanımlama satırı uyarlanmalıdır.

Veri kaynağı: <https://snap.stanford.edu/data/finefoods.txt.gz>

Sıkıştırılmış kaynak SHA-256:

```text
03bbbe147814be08a945501ef867795a8ad535385341e71f0d56a0c7d9daf1b4
```

Temizlik: HTML entity açma, etiket kaldırma, boşluk birleştirme, küçük harfe dönüştürme. Stopword silme veya stemming yoktur. Birden fazla puanlı normalize metinler tamamen elenir; kalan aynı metinlerden kaynak sırasındaki ilk kayıt tutulur. Bölme `SHA256('cs415-v1:' + product_id)` ilk 16 onaltılık hanesine dayanır. Eşikler 0,70 ve 0,85'tir. İstatistiksel olarak dengeli bir bölme zorlanmadığı için yorum oranları tam %70/%15/%15 değildir.

## Teslim öncesi grubun incelemesi gerekenler

1. Metni ve sonuçların yorumunu okuyup grubun bu ilerlemeyi açıklayabildiğinden emin olun. Yazar adları öneriden alındı; okul numarası veya kurum bilgisi uydurulmadı.
2. TextCNN/DistilBERT eğitimleri henüz yapılmadığı için bunları tamamlanmış gibi sunmayın. Sonraki eğitimlerden sonuç gelirse tablo ve durum bölümlerini gerçek çıktılarla güncelleyin.
3. İngilizce dil denetimi, yakın tekrar kontrolü ve Transformer token uzunlukları sonraki iş olarak açık bırakıldı. Veri kapsamı değişirse manifest sürümünü değiştirip temelleri yeniden çalıştırın.
4. Öğretim elemanının gerçek son teslim tarihini ders sisteminden doğrulayın; verilen metinde tarih yoktu. Grup adına tek PDF gönderin.

Sayfa sayısı sınırı olmadığı için literatür, yöntem gerekçeleri ve deney ayrıntıları kısaltılmadan verildi. Kaynaklarda konferans, dergi, workshop ve preprint ayrımı korunmuştur.
