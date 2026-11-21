# Product Review Rating Prediction — Türkçe Literatür ve Yöntem Rehberi

**Ders:** CS 415 / Deep Learning  
**Hazırlanma tarihi:** 22 Ekim 2026  
**Amaç:** Makaleleri anlayarak proje sorusunu ve deneyleri seçmek; ardından proposal’ı birlikte, paragraf paragraf yazmak.

> Bu belge tamamlanmış bir proposal değildir. Makale bulguları, öğretici örnekler ve bizim proje için önerilen deneyler birbirinden ayrılmıştır. Burada kendi projemize ait ölçülmüş bir sonuç yoktur. Örnek sayılar ayrıca belirtilmiştir.

## İçindekiler

1. [Bizim problemimiz ve literatürü okuma biçimimiz](#1-bizim-problemimiz-ve-literatürü-okuma-biçimimiz)
2. [Teknik kavramları birbirinden ayırmak](#2-teknik-kavramları-birbirinden-ayırmak)
3. [On temel makale](#3-on-temel-makale)
4. [Tamamlayıcı kaynaklar ve veri kaynakları](#4-tamamlayıcı-kaynaklar-ve-veri-kaynakları)
5. [Literatürden bizim deneylerimize geçiş](#5-literatürden-bizim-deneylerimize-geçiş)
6. [Performansı nasıl değerlendireceğiz?](#6-performansı-nasıl-değerlendireceğiz)
7. [OOD ve veri sızıntısı konusunda deney tasarımı](#7-ood-ve-veri-sızıntısı-konusunda-deney-tasarımı)
8. [Proposal’a nasıl aktaracağız?](#8-proposala-nasıl-aktaracağız)
9. [Kaynakça](#9-kaynakça)

## 1. Bizim problemimiz ve literatürü okuma biçimimiz

### 1.1 Girdi ve hedef

Bizim görevimiz, bir ürün yorumunun metninden o yoruma ait **1–5 yıldız puanını** tahmin etmek. Ürünün bütün kullanıcılar üzerinden ortalama puanını hesaplamak veya kullanıcıya yeni ürün önermek ayrı görevlerdir.

Matematiksel olarak eğitim verisi `D = {(x_i, y_i)}` biçimindedir. `x_i` yorum metni, `y_i ∈ {1,2,3,4,5}` gözlenen kullanıcı puanıdır. Model, metinden öğrendiği temsil üzerinden bir yıldız tahmini üretir. Yıldızlar sıralıdır; fakat kullanıcı memnuniyetinin 1–2 ile 4–5 arasında psikolojik olarak eşit miktarda değiştiğini varsaymak zorunda değiliz.

Etiket, kullanıcının gerçekten verdiği puandır. Bu puan yalnızca kelimelerdeki olumlu/olumsuz ifadeleri değil; fiyatı, teslimatı, beklentiyi ve kişisel puanlama alışkanlığını da yansıtabilir. Bu nedenle metinden puan tahmininde kaçınılmaz belirsizlik olabilir. Modelin farklı tahmin ettiği her yorumu otomatik olarak “yanlış etiketlenmiş” saymamalıyız.

### 1.2 Veri setimiz

Önerilen [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), [orijinal SNAP sayfasına](https://snap.stanford.edu/data/web-FineFoods.html) göre 568.454 yorum, 256.059 kullanıcı ve 74.258 ürün içerir; zaman aralığı Ekim 1999–Ekim 2012’dir. Metin, özet, puan, kullanıcı ve ürün bilgileri bulunur. Temizlikten sonraki örnek sayısını ayrıca raporlamamız gerekir.

Başlangıçta yalnızca yorum metnini kullanmak, görevi açık tutar. Özeti de eklemek istersek bunu ayrı bir girdi tercihi olarak belirtiriz. Ürün kimliğini veri bölmek için kullanabiliriz; bu, kimliği model girdisi yapmakla aynı şey değildir. Hedef puanı doğrudan veya onu açığa çıkaran türetilmiş alanları girdiye eklemeyiz.

### 1.3 Her makalede soracağımız sorular

- Tam olarak hangi görev çözülüyor: ikili sentiment, beş sınıflı sentiment, yıldız tahmini veya recommendation?
- Veri seti bizimkiyle aynı mı? “Amazon” sözcüğü tek başına yeterli değil.
- Değişen şey metin temsili mi, ağ mimarisi mi, loss mu, veri kullanımı mı?
- Karşılaştırmalar aynı veri bölünmesinde ve benzer eğitim bütçesinde mi?
- Sonuç hangi metrikle ölçülmüş? F1 macro mu, weighted mı?
- Bu çalışma bizim hangi kararımızı gerekçelendiriyor?

## 2. Teknik kavramları birbirinden ayırmak

### 2.1 Temsil, encoder, çıkış katmanı ve loss

Bir metin sınıflandırıcısını dört ayrı karar olarak düşünmek yararlıdır:

```text
Ham yorum
   ↓
Tokenlaştırma ve sayısal temsil
   ↓
Encoder: metinden özellik çıkaran ağ
   ↓
Çıkış katmanı: sınıf skorları veya sıralı eşikler
   ↓
Tahmin

Eğitim sırasında: model çıktısı + gerçek etiket → loss → backpropagation
```

**Tokenlaştırma**, metni işlenecek birimlere ayırır. Bir token her zaman tam kelime değildir; bazı yöntemler kelime parçaları kullanır. **Embedding**, bu birimin sayısal temsilidir. **Encoder**, temsilleri işleyerek görev için yararlı özellikler üretir. **Loss**, modelin eğitimde hangi hataları azaltmaya çalışacağını belirler.

Bu ayrım proje için önemlidir: CNN ile LSTM’yi karşılaştırmak encoder karşılaştırmasıdır. Aynı CNN’i Cross-Entropy ve ordinal loss ile eğitmek ise öğrenme hedefini karşılaştırır. İkisini aynı anda değiştirirsek sonuç farkını hangi değişikliğin oluşturduğunu ayırmak zorlaşır.

### 2.2 Eğitimde ağırlıkların değişmesi ile girdiye göre çıktının değişmesi

Eğitim sırasında optimizer, loss’un gradyanını kullanarak öğrenilebilir parametreleri günceller. Test sırasında ağırlıklar sabit tutulur; farklı yorumlar aynı ağdan farklı çıktılar alır. Dolayısıyla “çıktı vektörü değişti” demek, “model test sırasında yeniden eğitildi” demek değildir.

Embedding katmanı dondurulmuşsa yalnızca sonraki öğrenilebilir katmanlar güncellenir. Dondurulmamışsa embedding parametreleri de eğitim sinyalinden etkilenebilir. Bu seçenekleri karşılaştırırken diğer koşulların aynı tutulması gerekir.

### 2.3 Model çıktısı ve nihai karar

Standart beş sınıflı model, `z ∈ R^5` şeklinde beş logit üretir. Softmax bunları toplamı 1 olan olasılıklara dönüştürür:

`p_k = exp(z_k) / Σ_j exp(z_j)`

En büyük olasılığa sahip yıldızı seçmek, `argmax` kararıdır. Olasılıklardan beklenen yıldız değerini hesaplamak da mümkündür; ancak bu farklı bir karar kuralıdır. Bir loss’u argmax ile, diğerini yuvarlanmış beklenen değerle değerlendirip bütün farkı loss’a bağlamamalıyız. Karar kuralını önceden belirler veya ayrı bir deney olarak inceleriz.

### 2.4 Cross-Entropy hakkında önemli ayrım

Tek örnek için standart kayıp `L_CE = -log(p_y)` olur. Gerçek sınıfa verilen olasılık yükseldikçe kayıp azalır. CE yanlış sınıfları eğitimden tamamen dışlamaz: softmax üzerinden bütün logitler gradyan alır. Ancak yıldızlar arasındaki mesafeyi açıkça kullanmaz.

Örneğin gerçek yıldız 1 iken iki model de `p_1 = 0,4` veriyorsa, kalan olasılığın ağırlıklı olarak 2’ye veya 5’e gitmesi bu örneğin CE değerini değiştirmez. Ordinal yöntemleri motive eden ayrım budur. Bu örnek öğretici amaçla kurulmuştur; makaleden alınmış deney sonucu değildir.

## 3. On temel makale

### 3.1 Kim (2014) — TextCNN

**Başlık:** Convolutional Neural Networks for Sentence Classification  
**Bağlantılar:** [Makale](https://aclanthology.org/D14-1181/) · [PDF](https://aclanthology.org/D14-1181.pdf)

**Problem ve yöntem:** Önceden eğitilmiş kelime vektörleriyle basit bir CNN’in cümle sınıflandırmasındaki etkinliği araştırılıyor. `n` token, `d` boyutlu embedding ile `X ∈ R^(n×d)` oluşturur. Filtre, `h` ardışık tokenın bütün embedding boyutlarını kapsar. `c_i = ReLU(w·vec(X[i:i+h]) + b)` işlemi konumlar boyunca uygulanır. Her filtrenin çıktısına max-over-time pooling uygulanarak tek değer alınır; değerler birleştirilip sınıflandırıcıya verilir.

**Teknik seçimler:** Deneylerde 300 boyutlu word2vec, 3/4/5 token genişlikleri ve her genişlikte 100 filtre kullanılır. Rastgele embedding, sabit pretrained embedding, güncellenen pretrained embedding ve iki kanallı yapı karşılaştırılır. İki kanalın biri sabit, diğeri öğrenilebilirdir.

**Bulgular:** SST-1’de rastgele başlangıç %45,0, güncellenen pretrained temsil %48,0 accuracy verir. İki kanal her görevde üstün değildir. SST-1 bizim yıldız veri setimiz değildir.

**Bizim kullanımımız:** Temel encoder’ımız olabilir. Aynı CNN’i farklı loss’larla eğitmek, yıldız sırasının etkisini incelemeyi kolaylaştırır. Pooling konum bilgisinin bir kısmını kaybettiğinden uzun yorumlarda uzak ifadelerin ilişkileri sınırlı temsil edilebilir. Bu, her uzun yorumda başarısız olacağı anlamına gelmez.

**Proposal’daki yeri:** Method ve baseline gerekçesi. Önce Figure 1, sonra model çeşitleri ve Table 2 okunmalı. Kendimize soracağımız soru: “Bizim kazanımımız embedding’den mi, loss’tan mı geliyor?”

### 3.2 Dang, Moreno-García ve De la Prieta (2021) — BERT ile hibrit modeller

**Başlık:** An Approach to Integrating Sentiment Analysis into Recommender Systems  
**Bağlantılar:** [Makale](https://www.mdpi.com/1424-8220/21/16/5666) · [DOI](https://doi.org/10.3390/s21165666)

**Problem ve yöntem:** Yorumlardan çıkarılan duygu bilgisini öneri sistemlerine eklemeyi amaçlıyor. İlk aşamada BERT temsilleri CNN→LSTM veya LSTM→CNN sırasıyla işleniyor. Yorumlar yıldızlara karşılık gelen beş duygu seviyesine sınıflandırılıyor. Sonra bu bilgi, kullanıcı–ürün puanlarından yararlanan collaborative filtering yöntemlerine katılıyor.

**Bizimle bağlantı:** Amazon Fine Foods üzerinde deney yapılması doğrudan ilgili. Table 2’de LSTM→CNN accuracy’si %80,04, CNN→LSTM %79,95. Raporlanan F-score weighted ortalamadır; macro-F1 değildir. Aradaki 0,09 yüzde puan fark, tek başına sıralamaların güvenilir üstünlüğünü kanıtlamaz.

**Teknik okuma:** Bir katmanın çıktısı diğerine aktarılır; bağımsız modellerin oylaması yapılmaz. BERT, temsil üretme aşamasıdır. CNN ve LSTM, bu temsillerden sınıflandırma özellikleri çıkarır. Makalenin çıktı aktivasyonu anlatımı ReLU içerir; bunu doğrudan normalize sınıf olasılığı üreten softmax gibi yorumlamamalıyız.

**Bizim kullanımımız:** Daha karmaşık bir mimari adayı ve aynı veri setinde önceki çalışma örneği. Önerdiğimiz ablation, BERT+sınıflandırıcı ile ek CNN/LSTM katmanlarını ayrı ayrı karşılaştırmaktır; bu bizim deney önerimizdir.

**Sınır ve proposal:** Sonraki MAE tabloları öneri sistemi bileşimini değerlendirir; saf metin sınıflandırıcımızla doğrudan kıyaslanamaz. Related Work’te kapsamı açıklanmalı; %80 sonucu bizim garanti hedefimiz yapılmamalı. Öncelikli okuma: Bölüm 3.2 ve Table 2.

### 3.3 Yang ve diğerleri (2016) — Hierarchical Attention Networks

**Başlık:** Hierarchical Attention Networks for Document Classification  
**Bağlantılar:** [Makale](https://aclanthology.org/N16-1174/) · [PDF](https://aclanthology.org/N16-1174.pdf)

**Problem:** Uzun belgelerde kelimelerin ve cümlelerin sınıflandırmaya katkısı eşit değildir. Metni tek bir düz dizi olarak işlemek yerine doğal hiyerarşisinden yararlanılır.

**Teknik yöntem:** Her cümledeki kelime temsilleri çift yönlü GRU’dan geçer. Öğrenilen attention skorları softmax ile ağırlıklara dönüştürülür. Ağırlıklı toplam cümle vektörünü üretir: `s_i = Σ_t α_it h_it`. Ardından cümle vektörleri ikinci bir çift yönlü GRU ve attention katmanından geçerek belge vektörüne dönüşür. Son katman sınıfı tahmin eder. Kelime attention’ı ve cümle attention’ı ayrı seviyelerdir.

**Bulgular:** Çalışmanın beş sınıflı Amazon deneyinde CNN-word %57,6, LSTM %59,4, HAN %63,6 accuracy raporlanır. Bu Amazon koleksiyonu Fine Foods değildir; Dang’ın sonucuyla doğrudan sıralama yapılamaz.

**Bizim kullanımımız:** Çok cümleli, karışık görüş içeren yorumlarda alternatif encoder olabilir. Örneğin teslimat olumlu, lezzet olumsuz olduğunda hangi bölümlerin etkili olduğu incelenebilir. Bizim önerdiğimiz analiz, kısa ve uzun yorum gruplarında performansı ayrı ölçmektir.

**Sınır:** Cümle bölme ve iki seviyeli padding ek uygulama yükü getirir. Attention ağırlıkları modelin hangi temsilleri ağırlıklandırdığını gösterir; tek başına insan düzeyinde nedensel açıklama kanıtı sayılmamalıdır.

**Proposal’daki yeri:** Uzun yorum motivasyonu ve alternatif mimari. Önce mimari şema, ardından attention denklemleri okunmalı. Ana soru: “Hiyerarşi eklemek, bizim yorum uzunluklarımızda maliyetini karşılıyor mu?”

### 3.4 Pang ve Lee (2005) — Seeing Stars

**Başlık:** Seeing Stars: Exploiting Class Relationships for Sentiment Categorization with Respect to Rating Scales  
**Bağlantılar:** [Makale](https://aclanthology.org/P05-1015/) · [PDF](https://aclanthology.org/P05-1015.pdf)

**Problem:** Olumlu/olumsuz kararının ötesinde, derecelendirme seviyelerinin ilişkileri nasıl kullanılabilir? Etiketlerin sıralı olması, bağımsız kategoriler gibi ele alınmalarından farklı bilgi sağlar.

**Yöntem:** Film yorumlarında puan seviyelerini ele alır. İncelenen metric labeling yaklaşımı, tek örneğin sınıf tercihiyle benzer örneklerin etiketlerinin uyumunu birlikte değerlendirir. Metinsel olarak yakın örneklere çok uzak puanlar verilmesini maliyetli hale getiren bir yapı düşünülür. Benzerlik ölçüsü bu nedenle yöntemin önemli bir parçasıdır.

**Bulguların anlamı:** Çalışma, sınıf ilişkilerinden yararlanmanın deneysel faydasını araştırır. Bizim için asıl katkısı bir CNN mimarisi sağlaması değil, puan tahmininin ordinal yapısını açıkça gündeme getirmesidir. Kullanılan yöntem deep learning değildir.

**Bizim kullanımımız:** Problem Definition’da 1→2 hatasıyla 1→5 hatasının aynı olmadığını gerekçelendirir. Bu düşünceyi neural modelimize ordinal loss veya eşik tabanlı çıkış olarak aktarabiliriz. Bu aktarım, orijinal algoritmayı aynen uyguladığımız anlamına gelmez.

**Sınır:** Yakın metinlerin daima aynı puanı taşıyacağı varsayılamaz; kullanıcı beklentileri farklı olabilir. Ayrıca ordinal sıra, puan aralıklarının öznel memnuniyet açısından eşit olduğu anlamına gelmez.

**Proposal’daki yeri:** Motivation ve ordinal problem tanımı. Okuma sorusu: “Bizim hedefimiz yalnızca doğru sınıf mı, yoksa uzak hataları da azaltmak mı?”

### 3.5 Chen ve diğerleri (2021) — Hata büyüklüğüne duyarlı değerlendirme

**Başlık:** Error-Sensitive Evaluation for Ordinal Target Variables  
**Bağlantılar:** [Makale](https://aclanthology.org/2021.eval4nlp-1.19/) · [PDF](https://aclanthology.org/2021.eval4nlp-1.19.pdf)

**Problem:** Accuracy ve F1, yanlış etiketin gerçek etiketten ne kadar uzakta olduğunu doğrudan açıklamaz. Sıralı hedeflerde yalnızca bu metriklerle model seçmek, önemli hata farklarını gizleyebilir.

**Yöntem:** Aynı temel model mimarisi üzerinde çok sınıflı sınıflandırma ile ordinal regression karşılaştırılır. Mimariyi aynı tutmak, karşılaştırmanın hedef kodlaması ve öğrenme yaklaşımına odaklanmasını sağlar. CORAL tabanlı ordinal yapı da incelenir.

**Bulgular:** Çok sınıflı yaklaşım bazı veri setlerinde accuracy/F1 bakımından daha iyi olsa da ordinal yaklaşım incelenen benchmark’larda MSE ve sıralama ilişkisini değerlendiren ölçümlerde daha iyi sonuçlar verir. Bu, her ordinal yöntemin her veri setinde üstün olduğu şeklinde genellenemez.

**Bizim kullanımımız:** Performance Metrics bölümünün temel kaynağıdır. MAE, Macro-F1 ve büyük hata oranını birlikte incelemek için gerekçe sağlar. Aynı encoder ile farklı loss karşılaştırması, bizim kontrollü deney tasarımımıza da örnektir.

**Kendi deney sorumuz:** Ordinal model MAE’yi düşürürken düşük yıldız sınıflarını ihmal ediyor olabilir mi? Bu nedenle yalnızca genel ortalamaya değil, sınıf bazında F1 ve hata dağılımına da bakarız.

**Okuma notu:** Önce abstract, sonra değerlendirme yaklaşımı ve sonuçların tartışılması okunmalı. Bibliyografik yıl için ACL kayıt bilgisindeki 2021 kullanılır. Makale sonuçları, bizim Fine Foods testimiz yapılmadan doğrudan proje sonucu olarak yazılmaz.

### 3.6 Castagnos, Mihelich ve Dognin (2022) — Ordinal Log-Loss

**Başlık:** A Simple Log-based Loss Function for Ordinal Text Classification  
**Bağlantılar:** [Makale](https://aclanthology.org/2022.coling-1.407/) · [PDF](https://aclanthology.org/2022.coling-1.407.pdf) · [Yazarların kodu](https://github.com/glanceable-io/ordinal-log-loss)

**Problem:** Standart CE, yıldızlar arasındaki uzaklığı açıkça kullanmaz. OLL, uzak yanlış sınıflara yüksek olasılık verilmesini daha güçlü cezalandırmayı amaçlar.

**Teknik yöntem:** `p_k` sınıf olasılığı, `y` gerçek sınıf ve `d(y,k)` etiket mesafesi olmak üzere:

`L_OLL = -Σ_k d(y,k)^α log(1-p_k)`

`α > 0` uzak hataların ağırlığını kontrol eder. Biz yıldızlar için `d(y,k)=|y-k|` seçebiliriz. Gerçek etiketin mesafesi sıfırdır; yanlış sınıfların olasılıkları mesafelerine göre cezalandırılır. Softmax olasılıkları birbirine bağlı olduğundan yanlış sınıfların baskılanması doğru sınıfa da alan açar.

**Bulgular:** Amazon Reviews Corpus, Yelp, SST-5 ve SNLI üzerinde farklı kayıplar karşılaştırılır. Ordinal ölçütlerde güçlü sonuçlar raporlanır; uygun α seçimi göreve bağlıdır. Amazon koleksiyonu bizim Fine Foods değildir.

**Bizim kullanımımız:** CNN mimarisini değiştirmeden eğitim hedefini değiştirebiliriz. Önerilen karşılaştırma aynı veri bölünmesiyle CNN+CE ve CNN+OLL’dir. α validation üzerinden seçilir.

**Sınır:** Uygulamada `log(1-p)` sayısal kararlılık gerektirir; eğitim loss’u değerlendirme metriğiyle aynı şey değildir. Genel MAE iyileşmesi bütün yıldız sınıflarının iyileştiğini garanti etmez.

**Proposal’daki yeri:** Proposed Method ve hipotez. Önce denklem, ardından deney metrikleri okunmalı. Kullanılan loss bize ait yeni yöntem gibi sunulmamalı.

### 3.7 Kasa ve diğerleri (2024) — Ordinal yöntemlerin karşılaştırılması

**Başlık:** Exploring Ordinality in Text Classification: A Comparative Study of Explicit and Implicit Techniques  
**Bağlantılar:** [Makale](https://aclanthology.org/2024.findings-acl.320/) · [PDF](https://aclanthology.org/2024.findings-acl.320.pdf)

**Problem ve yöntem:** Sıra bilgisini loss’a açıkça koymakla, dil modellerinin etiket açıklamalarındaki anlamdan yararlanması karşılaştırılır. CE, OLL, CORAL ve başka kayıpların yanında entailment ve üretici model yaklaşımları incelenir. Önerilen MLL, iki hedefi birleştirir:

`L_MLL = λ L_CE + (1-λ) L_OLL`, `0 ≤ λ ≤ 1`.

**Bulgular:** İngilizce MARC Amazon verisinde BERT-base için CE: weighted-F1 0,586 / MAE 0,485; OLL: 0,586 / 0,477; MLL: 0,589 / 0,476. İyileşmeler küçüktür; ortalamalar tek başına kesin istatistiksel üstünlük göstermez. CORAL bazı deneylerde belirgin geride kalır.

**Bizim kullanımımız:** Aynı CNN’de CE, OLL ve MLL karşılaştırabiliriz. Bu, hem mesafeyi önemsemenin hem CE ile birleştirmenin etkisini ayırır. λ validation üzerinden seçilmelidir; testte en iyi λ’yı aramak veri sızıntısıdır.

**Sınır:** Veri Fine Foods değildir. Weighted-F1, bizim önereceğimiz Macro-F1 ile karıştırılmamalı. Büyük dil modellerinin sonuçları, daha küçük bir CNN’in beklenen performansı sayılamaz.

**Proposal’daki yeri:** Güncel Related Work ve loss seçimi. Ordinal yaklaşımın zaten araştırıldığını gösterir; “yıldız sırasını ilk kullanan proje” iddiasını engeller. Öncelikle Tables 2–3 ve yöntemin avantaj/sınır tartışması okunmalı.

### 3.8 Cao, Mirjalili ve Raschka (2020) — CORAL

**Başlık:** Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation  
**Bağlantılar:** [PDF](https://sebastianraschka.com/pdf/papers/2020-coral.pdf) · [DOI](https://doi.org/10.1016/j.patrec.2020.11.008)

**Problem:** Sıralı bir hedefi bağımsız ikili sorulara bölmek tutarsız kararlar oluşturabilir. Model “puan 4’ten büyük” derken “puan 2’den büyük değil” diyebilir.

**Teknik yöntem:** CORAL, `K` sınıfı `K−1` eşik görevine dönüştürür. Beş yıldız için sorular `y>1`, `y>2`, `y>3`, `y>4` olur. Dört yıldızın hedef kodu `[1,1,1,0]` şeklindedir. Paylaşılan ağırlık vektörü ve eşiğe özgü bias’larla rank tutarlılığı sağlayan yapı kurulur. Dört bağımsız sigmoid eklemek tek başına CORAL değildir.

**Bulgular:** Orijinal deneyler yüz görüntülerinden yaş tahmini üzerinedir. Çalışma ordinal ağlara tutarlı çıkış kazandırır; Fine Foods başarısını doğrudan ölçmez.

**Bizim kullanımımız:** Encoder’ı koruyup çıkış başlığını değiştiren üçüncü bir yöntem ailesidir. CE/OLL/MLL aynı sınıf olasılıkları üzerinde çalışabilirken, CORAL hedef kodlamasını ve çıkış yorumunu da değiştirir.

**Sınır:** Eğitim ve tahmin kodu standart softmax sınıflandırıcıdan farklıdır. Bu ek değişikliği deney raporunda açıkça belirtmeliyiz. Kasa’nın sonuçları nedeniyle CORAL’ın mutlaka üstün çıkacağını varsaymayız.

**Proposal’daki yeri:** Alternatif yöntem veya genişletme. Önce eşik mantığı, sonra mimari ve tutarlılık açıklaması okunmalı. Ana soru: “Loss değişikliği yeterli mi, yoksa sıralı çıkış tasarımı ek fayda sağlıyor mu?”

### 3.9 Blitzer, Dredze ve Pereira (2007) — Kategoriler arası aktarım

**Başlık:** Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification  
**Bağlantılar:** [Makale](https://aclanthology.org/P07-1056/) · [PDF](https://aclanthology.org/P07-1056.pdf)

**Problem:** Kitap, DVD, elektronik ve mutfak yorumlarında duygu farklı kelimelerle ifade edilir. Bir alanda öğrenilen ilişkiler başka alanda zayıflayabilir.

**Teknik yöntem:** Structural Correspondence Learning, iki alanda sık görülen pivot özellikleri seçer. Etiketli kaynak veriyle birlikte kaynak ve hedefin etiketsiz verileri kullanılır. Diğer özelliklerden pivotların varlığını tahmin eden doğrusal modeller öğrenilir. Ağırlık matrisinin SVD ile düşük boyutlu yapısı çıkarılarak alanlar arasındaki ortak ilişkileri temsil eden ek özellikler oluşturulur. SCL-MI, pivot seçiminde etiketle ilişkili bilgiyi de dikkate alır.

**Bulgular:** Çalışma alanlar arası uyarlamanın faydasını ve alan benzerliğinin aktarım başarısıyla ilişkisini inceler. Görev ikili sentiment’tir; beş yıldız tahmini değildir.

**Bizim kullanımımız:** Görüntü projesindeki OOD sorununun metinde de karşılığı olduğunu açıklayan temel kaynaktır. Yemek yorumlarından elektronik yorumlarına geçiş deneyini motive eder. SCL’yi ana model yapmak zorunda değiliz; dersimizde deep learning merkezde kalır.

**Sınır:** Hedef alanın etiketsiz verisini kullanmak, hedefi hiç görmeden test etmekten farklıdır. Yöntemin sonuçlarını doğrudan bizim zero-adaptation testimizle kıyaslayamayız.

**Proposal’daki yeri:** Motivation, Generalization ve Limitations. Okurken özellikle kullanılan verilerin etiketli/etiketsiz ayrımı izlenmeli. Ana soru: “Model duygu bilgisini mi öğreniyor, yoksa kategoriye özgü kolay ipuçlarına mı dayanıyor?”

### 3.10 Rostami ve diğerleri (2023) — Daha dayanıklı iç temsiller

**Başlık:** Domain Adaptation for Sentiment Analysis Using Robust Internal Representations  
**Bağlantılar:** [Makale](https://aclanthology.org/2023.findings-emnlp.769/) · [PDF](https://aclanthology.org/2023.findings-emnlp.769.pdf)

**Problem:** Alan değiştiğinde aynı duyguya ait metin temsilleri farklı bölgelere kayabilir; sınıflandırma sınırı hedef alanda daha çok hata yapar.

**Teknik yöntem:** Kaynak alandaki farklı sınıfların temsillerini geniş aralıklarla ayırmak ve alanların dağılımlarını hizalamak birlikte amaçlanır. Kaynak temsillerin dağılımı Gaussian mixture ile modellenir. Sliced Wasserstein distance, dağılım farkını farklı doğrultulardaki tek boyutlu izdüşümler üzerinden ölçmede kullanılır. Eğitim, sınıf ayrımını korurken hedef örneklerin uygun temsil bölgelerine yaklaşmasını hedefler.

**Bulgular:** Amazon’un farklı ürün alanları arasındaki ikili duygu aktarımında yöntem değerlendirilir; çalışma uyarlamanın etkinliğine ilişkin teorik ve deneysel sonuçlar sunar. Bu, beş yıldızlı Fine Foods sonucu değildir.

**Bizim kullanımımız:** Harici kategoride ciddi performans düşüşü görürsek daha ileri bir araştırma yönü sağlar. Ana projede önce düşüşü doğru ölçmek, sonra gerekiyorsa uyarlama denemek daha yönetilebilir olur.

**Sınır:** Hedef alanın etiketsiz eğitim verisine ihtiyaç vardır. Hedef test örneklerini uyarlamada kullanmak farklı, transductive bir protokoldür; önerimiz ayrı target-train ve target-test kullanmaktır. Temsil hizalama, ordinal loss’tan farklı bir sorunu hedefler.

**Proposal’daki yeri:** OOD Related Work ve olası genişletme. Okuma sorusu: “Yeni kategorideki başarısızlık yıldızların sırasından mı, öğrenilen metin temsillerinin alan bağımlılığından mı kaynaklanıyor?”

## 4. Tamamlayıcı kaynaklar ve veri kaynakları

### 4.1 BERT — Temsili anlamak için

[Devlin ve diğerleri, 2019: BERT](https://aclanthology.org/N19-1423/), token temsillerini hem sol hem sağ bağlamı kullanarak üreten pretrained Transformer yaklaşımını açıklar. Word2vec lookup’ında aynı token aynı başlangıç vektörünü alırken, BERT’in bağlamsal çıktısı bulunduğu cümleye göre değişir. Tokenlar kelime parçaları da olabilir.

Kendi örneğimizde “good” ile “not good” içindeki token temsilleri, çevredeki bilgi nedeniyle farklılaşabilir. Bu, modelin bütün olumsuzlukları doğru yorumlayacağını garanti etmez. BERT’i dondurulmuş feature extractor olarak kullanmak ile uçtan uca fine-tune etmek farklı deneylerdir. Dang’ın yaklaşımını anlamak ve pretrained encoder seçimini açıklamak için bu kaynak kullanılabilir.

### 4.2 CheckList — Modelin belirli davranışlarını sınamak için

[Ribeiro ve diğerleri, 2020: Beyond Accuracy](https://aclanthology.org/2020.acl-main.442/), yalnızca tek bir test skoruna bakmak yerine belirli dilsel yetenekleri kontrollü örneklerle sınamayı önerir. Temel işlev testleri, anlamı koruyan değişikliklerde tutarlılık ve beklenen yöndeki değişimler ayrı test türleridir.

Bizim ek deneyimiz, insan tarafından kontrol edilmiş küçük bir tanı kümesi olabilir. Anlamı koruyan bir yazım değişikliğinde tahminin ne kadar değiştiğini ölçebiliriz. Olumlu ifadeyi olumsuzlaştırınca tam olarak hangi yıldızın doğru olduğunu her zaman bilemeyiz; bu yüzden otomatik olarak kesin yıldız etiketi üretmeyiz. Böyle bir tanı testi harici kategori OOD testinin yerine geçmez.

### 4.3 Aynı veri setinde ek çalışma

[Developing a Product Rating Prediction Model Based on Customer Text Feedback Using Machine Learning Algorithms](https://vujs.vn/all-issues/copy/A.0109-2025), Fine Foods üzerinde klasik modellerle CNN/LSTM gibi modelleri karşılaştırır. Yayıncı özetinde accuracy ile ordinal ölçütler arasında farklı model sıralamaları raporlanır.

**Doğrulama durumu:** Bu araştırmada tam metindeki veri bölünmesi ve deney ayrıntıları doğrulanamadı. Bu nedenle kaynak ek okuma olarak tutulmuştur; kesin sayısal benchmark veya mimari tercihi için kullanılmamıştır.

### 4.4 Harici veri için Amazon Reviews 2023

[Resmî veri sayfası](https://amazon-reviews-2023.github.io/) farklı kategorilere ait yorum, puan ve ürün bilgileri sağlar. İlgili veri yayınının kaynağı Hou ve arkadaşlarının *Bridging Language and Items for Retrieval and Recommendation* çalışmasıdır; makalenin ana görevi bizim metinden yıldız sınıflandırmamızla aynı değildir.

Harici test için bütün koleksiyonu indirmek gerekmez. İngilizce, 1–5 puanlı ve uygun metin alanına sahip kontrollü bir örneklem seçilebilir. Veri yılları da değişebileceğinden gözlenen fark, kategoriyle birlikte zaman ve sınıf dağılımı değişimini içerebilir.

## 5. Literatürden bizim deneylerimize geçiş

Bu bölüm makalelerin ortak sonucu değildir; kaynaklardan hareketle **bizim proje için önerilen deney tasarımıdır**. Henüz yöntem seçimi kesinleşmemiştir.

### 5.1 Üç farklı araştırma yönü

| Yön | Araştırma sorusu | Olası karşılaştırma | Temel risk |
|---|---|---|---|
| Temsil/mimari | Hangi encoder yorum bilgisini daha iyi kullanıyor? | TextCNN, HAN veya BERT tabanı | Çok fazla mimariyle yüzeysel deney yapmak |
| Ordinal öğrenme | Yıldız sırası büyük hataları azaltıyor mu? | Aynı encoder + CE/OLL/MLL | Bir metriği iyileştirirken diğerini düşürmek |
| Genelleme | Yeni ürün/kategoride kazanım korunuyor mu? | Product-disjoint ve harici test | Farklı dağılım etkilerini birbirine karıştırmak |

Üç yönü birden en geniş haliyle gerçekleştirmek gerekmez. Ders projesinde tek bir ana soru seçip diğerlerini onu açıklayan deneyler olarak kullanmak daha tutarlı bir çalışma oluşturur.

### 5.2 Önerilen çekirdek kapsam

Ana soru: **Aynı TextCNN’de ordinal eğitim, görülmemiş ürünlerde büyük yıldız hatalarını azaltıyor mu?**

Temel modeller `TextCNN+CE`, `TextCNN+OLL` ve `TextCNN+MLL` olabilir. Tokenlaştırma, embedding başlangıcı, veri bölünmesi, maksimum uzunluk ve eğitim bütçesi ortak tutulur. Her loss için uygun öğrenme oranı gerekiyorsa eşit büyüklükte validation arama bütçesi tanınır. Tek bir modele yoğun ayar yapıp diğerini varsayılan ayarda bırakmayız.

Bu kapsam deep learning gereğini karşılar: encoder, backpropagation ile görev üzerinde eğitilir. Literatürdeki mimariyi kullanmak bilimsel katkıyı ortadan kaldırmaz; kaynağı belirtmek ve deneyde neyi yeni ölçtüğümüzü açıklamak gerekir. Dersin özel yeniden-uygulama kuralları varsa ayrıca uygulanmalıdır.

### 5.3 Hipotez ile sonuç farklıdır

Önerilen hipotez: “Ordinal loss, standart CE’ye göre MAE ve en az iki yıldızlık hata oranını azaltabilir.” Bu henüz test edilmemiş bir beklentidir. “Ordinal model daha iyi olacak” diye kesin sonuç yazmayız.

İkinci soru, bu olası kazanımın harici kategoride korunup korunmadığıdır. Ordinal loss, kelime dağarcığı veya kategori değişimini doğrudan çözmek için tasarlanmış değildir. OOD başarısını artırmaması, ilk hipotezin anlamsız olduğu anlamına gelmez; etki alanını açıklamış oluruz.

### 5.4 Genişletmeleri sıraya koymak

1. Veri temizliği, bölme ve CE baseline’ı çalışır hale getir.
2. OLL ve MLL karşılaştırmasını tamamla.
3. Aynı test örneklerinde hata türlerini incele.
4. Zaman varsa bir harici kategori testi ekle.
5. Kaynak yeterliyse aynı loss karşılaştırmasını ikinci bir encoder üzerinde tekrarla.

İkinci encoder, sonucun yalnızca CNN’e özgü olup olmadığını araştırır. Domain adaptation ise hedef alan verisi, ek loss ve ayrı veri bölünmesi gerektirdiğinden daha büyük bir genişletmedir. Bunların hepsini proposal’da zorunlu teslim gibi vaat etmek yerine çekirdek kapsamı netleştiririz.

### 5.5 Ablation neden gerekli?

Bir modelin daha karmaşık olması hangi parçanın faydalı olduğunu göstermez. Kendi deneyimizde embedding, loss ve encoder’ı aynı anda değiştirirsek sonuç farkının nedenini ayıramayız. Ablation, belirli bir bileşeni çıkararak veya tek bir tercihi değiştirerek katkısını ölçer.

Örneğin MLL’yi yalnızca CE ile kıyaslamak, ordinal bileşenin faydasını araştırır. MLL’yi OLL ile de kıyaslamak, CE bileşeninin ek faydasını araştırır. Bu iki karşılaştırma aynı soru değildir. Bütün modelleri aynı test yorumlarında değerlendirmek, hata örneklerini eşleştirerek karşılaştırmayı da sağlar.

## 6. Performansı nasıl değerlendireceğiz?

### 6.1 Birlikte raporlanacak ölçümler

| Metrik | Hesap/anlam | Neyi kaçırabilir? |
|---|---|---|
| Accuracy | Doğru yıldız sayısı / bütün örnekler | Hatanın kaç yıldız olduğunu ve azınlık sınıfları |
| Macro-F1 | Beş sınıfın F1 değerlerinin eşit ortalaması | Yıldızlar arasındaki mesafeyi |
| MAE | `mean(abs(y_hat-y))` | Hatanın hangi sınıfta yoğunlaştığını |
| Büyük hata oranı | `mean(abs(y_hat-y) >= 2)` | Bir yıldızlık hatalar arasındaki farkları |
| Sınıf bazlı F1 | Her yıldız için ayrı precision/recall dengesi | Tek başına hata mesafesini |

MAE yıldız ölçeğinde yorumlanabilir: 0,5 değeri, örnek başına ortalama mutlak farkın yarım yıldız olmasıdır; her örnekte yarım yıldız tahmin yapıldığı anlamına gelmez. Weighted-F1 sınıfları örnek sayılarıyla ağırlıklandırır. Macro-F1 ise nadir yıldızlara da eşit sınıf ağırlığı verir. İkisini aynı isim altında sunmayız.

### 6.2 Aynı başarı kelimesi farklı şeyler anlatabilir

Aşağıdaki örnek bizim tarafımızdan oluşturulmuştur:

```text
Gerçek:  1 2 3 4 5
Model A: 1 2 3 4 1
Model B: 2 3 3 4 5
```

A’nın accuracy’si %80, MAE’si 0,8’dir. B’nin accuracy’si %60, MAE’si 0,4’tür. A daha çok doğru tahmin yaparken, B’nin toplam yıldız hatası daha küçüktür. Dolayısıyla model seçiminin hangi kullanım hedefiyle yapıldığı açık olmalıdır.

Bir başka risk, modelin düşük MAE uğruna orta puanları fazla tercih etmesidir. Bunun olup olmadığını confusion matrix ve sınıf bazlı ölçümlerle kontrol ederiz. Genel ortalamanın iyileşmesi, 1 yıldızlı kullanıcıların şikâyetlerinin daha iyi yakalandığını tek başına göstermez.

### 6.3 Sayısal başarı kriteri nasıl konabilir?

Aşağıdakiler **öneri niteliğinde, henüz kabul edilmemiş proje hedefleridir**; literatürden alınmış başarı garantileri değildir:

- Ortak product-disjoint testte CE baseline’ına göre MAE’de en az %3 göreli azalma.
- Aynı testte büyük hata oranında en az %5 göreli azalma.
- Macro-F1’da en fazla 0,01 mutlak düşüş; tercihen düşüş olmaması.
- Ana karşılaştırmaların üç farklı random seed ile tamamlanması ve ortalama/standart sapmanın raporlanması.

Örneğin baseline MAE=0,60 ise %3 göreli azalma için hedef en fazla 0,582 olur. Büyük hata oranı %10 ise %5 göreli azalma, %9,5’e inmek demektir; beş yüzde puan azalma değildir. Bu örnek değerler hesap öğretmek içindir.

Hedefleri küçük bir eğitim/validation pilotundan sonra, nihai test sonuçlarına bakmadan kesinleştiririz. Hedefe ulaşılamaması raporda gizlenmez. Tamamlanmış ve kontrollü bir deney, hipotezi desteklemese bile değerli bir bulgudur; performans hedefinin karşılanıp karşılanmadığı ayrıca dürüstçe belirtilir.

### 6.4 Küçük farkları nasıl yorumlarız?

Tek bir seed’de 0,002 F1 farkı görmek, yöntemin güvenilir üstünlüğünü göstermeyebilir. Eğitim rastgeleliğini tekrarlarla, test örneklemesi belirsizliğini gerekirse paired bootstrap ile inceleriz. Ürün başına çok yorum varsa ürün gruplarını koruyan bootstrap düşünülmelidir. Üç seed tek başına bütün belirsizliği çözmez; ancak tek çalıştırmadan daha açıklayıcıdır.

## 7. OOD ve veri sızıntısı konusunda deney tasarımı

### 7.1 Üç farklı test sorusu

**Rastgele yorum ayrımı:** Yorumlar train/validation/test’e ayrılır. Aynı ürünün farklı yorumları birden fazla bölümde bulunabilir. Bu her görev için otomatik veri sızıntısı değildir; fakat görülmemiş ürüne genelleme iddiasını desteklemez.

**Ürün bazlı ayrım:** Aynı `ProductId` yalnızca bir bölümde bulunur. Bu, eğitimde görülmeyen ürünlerin yorumlarını tahmin etmeyi ölçer. Yemek kategorisinin içinde kalıyorsak buna doğrudan kategoriler arası OOD demeyiz. Aynı kullanıcının farklı ürünlerde görünmesi hâlâ mümkündür; unseen-user iddiası yapmayız.

**Harici kategori testi:** Yemek yorumlarında eğitilmiş model başka kategoride, yeniden eğitilmeden değerlendirilir. Burada kelimeler, yorum uzunluğu, yıldız dağılımı, ürün özellikleri ve veri yılları değişebilir. Düşüşü tek bir nedene bağlamak için ek kontroller gerekir.

Bu üç düzen için sayıların farklı çıkması normaldir. Her deneyin hedeflediği kullanım senaryosunu ayrı yazmak, tek bir yüksek accuracy’den daha açıklayıcıdır.

### 7.2 Temizlik ve eğitim sınırları

Tekrarlanan veya neredeyse aynı yorumların train ve testte bulunması, modelin genelleme yerine metin benzerliğinden yararlanmasına neden olabilir. Deduplication kurallarını önceden belirleyip raporlarız. Farklı ürün kimliklerinde aynı metin bulunabileceğinden yalnızca ürün bazlı bölme bu riski tamamen ortadan kaldırmaz.

Vocabulary gibi veri üzerinden öğrenilen dönüşümler train üzerinde kurulmalıdır. Sabit pretrained tokenizer kullanımı ayrı bir durumdur. Sınıf ağırlıkları train’den hesaplanır. Oversampling yapılacaksa yalnızca train’de yapılır. Testi yapay olarak dengelemek, gerçek dağılımdaki performans sorusunu değiştirir; dengeli alt küme gerekirse ayrı tanı deneyi olarak raporlanır.

Metin temizliğinde “not” gibi olumsuzlukları veya anlamlı noktalama işaretlerini otomatik silmek zararlı olabilir. Temizlik, bütün yöntemlere körlemesine aynı işlemleri uygulamak değildir; tokenizer ve encoder’ın beklentileri dikkate alınmalıdır. Kullanılan kurallar açık olmalıdır.

### 7.3 OOD testi ile domain adaptation farklıdır

Doğrudan OOD testinde model, hedef kategori verisini eğitimde kullanmadan sınanır. Unsupervised domain adaptation’da hedef kategorinin etiketsiz eğitim verileri kullanılır. Hedef etiketleriyle fine-tuning ise supervised adaptation’dır. Bu koşulları tek tabloda karşılaştırabiliriz, fakat veri erişimini açıkça belirtiriz.

Önerilen ilk adım doğrudan OOD testidir. Sonra adaptation eklersek hedefin eğitim ve test örneklerini ayırırız. Hedef test performansına tekrar tekrar bakıp yöntem seçmek de test bilgisini geliştirme sürecine taşır. Test kümesi son değerlendirme için korunmalıdır.

### 7.4 Bizim hata analizimiz

Her modelden yalnızca başarı skoru almak yerine ortak örnekler üzerinde şu grupları inceleyebiliriz:

- Gerçek 1–2 yıldız olup 4–5 tahmin edilen ciddi hatalar.
- Gerçek 4–5 yıldız olup 1–2 tahmin edilen ters hatalar.
- 3 yıldız gibi karışık görüş içerebilen orta sınıflar.
- Kısa ve uzun yorumlar.
- Olumsuzluk, birden fazla ürün özelliği veya karşıt görüş içeren yorumlar.

Kısa/uzun sınırlarını ve örnek seçimini nihai test sonuçlarına göre avantaj sağlayacak şekilde değiştirmeyiz. “Model lezzeti öğrendi” gibi yorumlar için birkaç örnek yeterli kanıt değildir; örnekleri gözlem olarak sunarız.

## 8. Proposal’a nasıl aktaracağız?

### 8.1 Kaynakların görev dağılımı

| Proposal kısmı | Yararlı kaynaklar | Anlatılacak bağlantı |
|---|---|---|
| Problem ve motivasyon | Pang–Lee, Blitzer | Sıralı hedef ve kategori değişimi |
| Related Work | Kim, Dang, Yang, Kasa | Temsil, mimari ve ordinal yöntemlerin gelişimi |
| Yöntem | Kim, Castagnos, Kasa; isteğe bağlı CORAL | Seçilen encoder ve eğitim hedefi |
| Değerlendirme | Chen; isteğe bağlı CheckList | Doğru sınıf ile hata büyüklüğünün ayrılması |
| Genelleme ve sınırlar | Blitzer, Rostami | Hedef veriye erişim ve aktarım koşulları |
| Veri seti | SNAP ve gerekirse Amazon Reviews 2023 | Veri kaynağı ve deney kapsamı |

Related Work, kaynakları tek tek art arda özetlemekten ibaret olmamalı. Mantıksal akış kurmalıyız: metin temsilleri ve encoder’lar → yıldız sırasının önemi → değerlendirme sorunu → genelleme → bizim dar araştırma sorumuz. Bu belge ayrıntılı öğrenme notudur; proposal’da aynı ayrıntıların tamamına yer vermek gerekmez.

### 8.2 Yenilik iddiasını doğru kurmak

CNN ile yıldız tahmini, ordinal loss ve domain adaptation önceden çalışılmıştır. “İlk kez yıldız sırasını kullanıyoruz” veya “OOD’yi çözüyoruz” demeyiz. Bizim olası katkımız, belirli veri temizliği ve ürün bazlı test koşullarında loss’ların büyük hatalara ve genellemeye etkisini kontrollü incelemektir.

Bu bir ders projesi düzeyinde savunulabilir deneysel katkı olabilir; literatürde küresel olarak ilk olduğuna dair iddia değildir. Yenilik, çok sayıda katman eklemekle ölçülmez. İyi tanımlanmış bir soru, adil deney ve açık sınırlamalar daha ikna edicidir.

### 8.3 Değerlendirme rubriğiyle bağlantı

**Writing quality:** Tek ana soru etrafında, sade B1 İngilizceyle neden–yöntem–ölçüm ilişkisi kuracağız. Teknik terimleri ilk kullanımda açıklayacağız.

**Comprehensiveness:** Veri, temizleme, bölme, model, loss, karşılaştırma, metrik, kaynak ihtiyacı ve sınırlamaları kapsayacağız. Kapsamlı olmak bütün yöntemleri uygulamak değildir.

**Novelty:** Hangi bilinen yöntemi hangi açık soruyu araştırmak için kullandığımızı belirteceğiz. Kaynakların katkısıyla kendi katkımızı ayıracağız.

**Challenge level:** Ordinal eğitim, kontrollü ablation ve görülmemiş ürün testi teknik zorluk sağlar. Harici kategori veya ikinci encoder eklenirse bu ek kapsamın zaman ve hesaplama maliyeti belirtilir.

### 8.4 Sen okuduktan sonra paragraf paragraf ilerleme sırası

1. Bir cümlede görevi ve ana araştırma sorusunu kesinleştirelim.
2. Problem Definition paragrafını yazalım: girdi, hedef, zorluk ve motivasyon.
3. Related Work’ü iki veya üç bağlantılı paragrafa indirelim.
4. Dataset ve Method bölümünde seçtiğimiz kapsamı anlatalım.
5. Hipotezi, sayısal hedefleri ve değerlendirme protokolünü yazalım.
6. Grup üyelerini, sorumlulukları ve uygulanabilirlik planını ekleyelim.
7. Kaynakça ve biçim kontrolünü yapalım: Arial veya eşdeğeri, 9 punto. Markdown kendi başına font/puntoyu sabitlemez; teslim belgesinde uygulanır.

### 8.5 Okuma sonrası karar listesi

- [ ] Ana amaç: doğru yıldız oranı, hata büyüklüğü veya genellemeden hangisi öncelikli?
- [ ] Çekirdek encoder TextCNN olacak mı?
- [ ] CE/OLL/MLL karşılaştırması kapsam için yeterli mi?
- [ ] Ürün bazlı ayrım temel test olacak mı?
- [ ] Harici kategori testi zorunlu mu, genişletme mi?
- [ ] GPU, zaman ve ekip büyüklüğü ikinci encoder’a izin veriyor mu?
- [ ] Nihai testten önce sayısal başarı kriterlerini sabitleyecek miyiz?

## 9. Kaynakça

1. Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*. EMNLP, 1746–1751. [Kaynak](https://aclanthology.org/D14-1181/).
2. Dang, C. N., Moreno-García, M. N., & De la Prieta, F. (2021). *An Approach to Integrating Sentiment Analysis into Recommender Systems*. Sensors, 21(16), 5666. [DOI](https://doi.org/10.3390/s21165666).
3. Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016). *Hierarchical Attention Networks for Document Classification*. NAACL-HLT, 1480–1489. [Kaynak](https://aclanthology.org/N16-1174/).
4. Pang, B., & Lee, L. (2005). *Seeing Stars: Exploiting Class Relationships for Sentiment Categorization with Respect to Rating Scales*. ACL, 115–124. [Kaynak](https://aclanthology.org/P05-1015/).
5. Chen, D. Z., Courtland, M., Faulkner, A., & Ezen-Can, A. (2021). *Error-Sensitive Evaluation for Ordinal Target Variables*. Eval4NLP, 189–199. [Kaynak](https://aclanthology.org/2021.eval4nlp-1.19/).
6. Castagnos, F., Mihelich, M., & Dognin, C. (2022). *A Simple Log-based Loss Function for Ordinal Text Classification*. COLING, 4604–4609. [Kaynak](https://aclanthology.org/2022.coling-1.407/).
7. Kasa, S. R., ve diğerleri (2024). *Exploring Ordinality in Text Classification: A Comparative Study of Explicit and Implicit Techniques*. Findings of ACL, 5390–5404. [Kaynak](https://aclanthology.org/2024.findings-acl.320/).
8. Cao, W., Mirjalili, V., & Raschka, S. (2020). *Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation*. Pattern Recognition Letters, 140, 325–331. [DOI](https://doi.org/10.1016/j.patrec.2020.11.008).
9. Blitzer, J., Dredze, M., & Pereira, F. (2007). *Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification*. ACL, 440–447. [Kaynak](https://aclanthology.org/P07-1056/).
10. Rostami, M., Bose, D., Narayanan, S., & Galstyan, A. (2023). *Domain Adaptation for Sentiment Analysis Using Robust Internal Representations*. Findings of EMNLP, 11484–11498. [Kaynak](https://aclanthology.org/2023.findings-emnlp.769/).
11. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT, 4171–4186. [Kaynak](https://aclanthology.org/N19-1423/).
12. Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*. ACL, 4902–4912. [Kaynak](https://aclanthology.org/2020.acl-main.442/).
13. McAuley, J., & Leskovec, J. (2013). *From Amateurs to Connoisseurs: Modeling the Evolution of User Expertise through Online Reviews*. WWW. [Fine Foods veri ve atıf sayfası](https://snap.stanford.edu/data/web-FineFoods.html).
14. Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). *Bridging Language and Items for Retrieval and Recommendation*. arXiv:2403.03952. [Veri sayfası](https://amazon-reviews-2023.github.io/) · [Makale](https://arxiv.org/abs/2403.03952).
15. Ha Dien Thi Hong. *Developing a Product Rating Prediction Model Based on Customer Text Feedback Using Machine Learning Algorithms*. Vinh University Journal of Science. [Yayıncı kaydı](https://vujs.vn/all-issues/copy/A.0109-2025). Ek okuma; bu rehber için tam metin deney ayrıntıları doğrulanmamıştır.

**Not:** Kaynakların sonuçları kendi veri, eğitim ve değerlendirme koşullarına aittir. Bu rehberde önerilen deneyler ve sayısal hedefler bizim proje tasarımı seçeneklerimizdir; yayımlanmış bulgu veya elde edilmiş proje sonucu değildir.
