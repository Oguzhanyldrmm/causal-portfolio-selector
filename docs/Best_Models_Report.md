# Best Oracle ve Overlap Model Raporu

Hazırlanma tarihi: 2026-05-06

Bu rapor, synthetic causal portfolio verisiyle eğitilen iki final aday modeli dokümante eder:

- En iyi oracle/regret modeli: oracle4_overlap1_regret025
- En iyi top3-overlap modeli: overlap2_oracle1_regret010

İki model de aynı 8 algoritmalı portföyden top3 algoritma seti seçer.

## Yönetici Özeti

| model | objective | odak | top1_hit | oracle_in_top3 | top3_overlap>=2 | avg_top3_overlap | avg_regret_at_3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle4_overlap1_regret025 | top3 combination reward regression | oracle coverage ve düşük regret | 2/14 | 11/14 | 4/14 | 1.143 | 1.357 |
| overlap2_oracle1_regret010 | top3 combination reward regression | ground-truth top3 overlap | 4/14 | 8/14 | 8/14 | 1.500 | 3.357 |
| balanced_985_ranking | ranking | n/a | 5/14 | 8/14 | 6/14 | 1.357 | 3.786 |

Kısa yorum:

- oracle4_overlap1_regret025: öncelik top3 içinde oracle seviyesinde bir algoritma bulmak ve regret düşürmekse en iyi aday.
- overlap2_oracle1_regret010: öncelik ground-truth top3 listesinden en az 2 algoritmayı yakalamaksa en iyi aday.
- balanced_985_ranking: önceki güçlü ranking baseline olarak tutuldu.

## Model Kayıt Durumu

Üç model de ileride tekrar kullanılabilecek şekilde kaydedildi. Final değerlendirmede ana karşılaştırma iki yeni combination selector üzerinden yapıldı; balanced_985_ranking modeli ise önceki güçlü baseline olarak saklandı.

| model | rol | kayıt durumu | yaklaşık boyut |
| --- | --- | --- | --- |
| oracle4_overlap1_regret025 | en iyi oracle/regret modeli | kaydedildi | 518 MB |
| overlap2_oracle1_regret010 | en iyi top3-overlap modeli | kaydedildi | 619 MB |
| balanced_985_ranking | baseline ranking modeli | kaydedildi | 282 MB |

## Portföy ve Veri

Kullanılan algoritmalar: PC_discrete, FCI, GES, HC, Tabu, K2, BOSS, GRaSP

MMHC bu final modellerde yok. Temiz synthetic ve exact koşularında başarıyla tamamlanan 8 algoritmalı portföy kullanıldı.

| kalem | değer |
| --- | --- |
| feature rows | 1336 |
| target rows | 10688 |
| synthetic_train dataset | 985 |
| synthetic_val dataset | 175 |
| synthetic_test dataset | 176 |
| dataset başına algoritma row | 8 |

Önemli training detayı: validation sonrası kaydedilen final selector synthetic_train ve synthetic_val birleşimi üzerinde yeniden eğitilir. 8 algoritmada 56 olası top3 kombinasyonu olduğu için final combination model yaklaşık 64.960 kombinasyon row ile eğitilir. synthetic_test splitindeki 176 dataset held-out kalır.

Exact evaluation set 14 dataset ve 112 target row içerir. Bu set sadece evaluation için kullanıldı; traininge dahil edilmedi.

## Feature Set

İki final model de 29 adet handcrafted dataset-level feature kullanır. Bu iki final modelde encoder veya learned graph fingerprint kullanılmadı.

| grup | featurelar |
| --- | --- |
| Boyut | n_samples, n_features, sample_to_feature_ratio |
| Değişken dağılımı | continuous_ratio, categorical_ratio, missing_ratio, avg_variance, avg_skewness, avg_kurtosis |
| Cardinality ve sparsity | avg_cardinality, max_cardinality, cardinality_entropy, rare_category_ratio, singleton_category_ratio, feature_sparsity_ratio |
| Pairwise bağımlılık | mean_nmi, max_nmi, std_nmi, mean_cramers_v, max_cramers_v, std_cramers_v, mean_chi2_pvalue, ci_rejection_rate |
| Proxy graph yapısı | proxy_graph_density, proxy_avg_degree, proxy_degree_gini, proxy_avg_clustering, proxy_modularity, proxy_num_components |

Model fit sırasında selector ayrıca top3 kombinasyon encoding featureları ekler:

- 8 adet binary algoritma-var/yok göstergesi.
- 3 adet grup var/yok göstergesi: constraint_based, score_based, search_based.
- Aynı 3 grup için count featureları.

Dolayısıyla her regression row, 29 dataset feature + 14 kombinasyon encoding feature içerir.

## Model Mimarisi ve Objective

İki final model de aynı implementationı kullanır:

Top3CombinationSelector

Selector, aday top3 kombinasyonları üzerinde supervised reward regression yapar. Her dataset için 56 olası top3 kombinasyonu üretilir. Her kombinasyona, gerçek algoritma run sonuçlarından hesaplanan bir reward target atanır.

| model | estimator | n_estimators | min_samples_leaf | random_state | imputer | feature_set |
| --- | --- | --- | --- | --- | --- | --- |
| oracle4_overlap1_regret025 | sklearn RandomForestRegressor | 300 | 2 | 42 | SimpleImputer(strategy=median) | handcrafted_all |
| overlap2_oracle1_regret010 | sklearn RandomForestRegressor | 300 | 2 | 42 | SimpleImputer(strategy=median) | handcrafted_all |

Loss/objective notu:

- Neural-network epoch loop, GRPO veya RL training yoktur.
- Target, her aday top3 kombinasyonu için hesaplanan supervised regression reward değeridir.
- RandomForestRegressor, tree splitlerinde default squared-error kriterini kullanır ve ağaçların ortalama reward tahminini döndürür.
- Inference sırasında 56 kombinasyonun tamamı skorlanır ve en yüksek predicted-reward kombinasyonu model top3 olur.

Reward bileşenleri:

- oracle_in_top3: candidate top3 içinde o dataset için minimum SHD alan algoritmalardan biri varsa 1.
- top3_overlap: candidate top3 ile ground-truth top3 arasındaki ortak algoritma sayısı. 0 ile 3 arasıdır.
- top3_overlap_at_least_2: top3_overlap en az 2 ise 1.
- regret_at_3: candidate top3 içindeki en iyi SHD ile oracle SHD arasındaki fark. Düşük olması iyidir.

Reward ağırlıkları:

| model | oracle ağırlığı | top3 overlap ağırlığı | en az 2 overlap bonusu | regret cezası |
| --- | --- | --- | --- | --- |
| oracle4_overlap1_regret025 | 4.0 | 1.0 | yok | 0.25 |
| overlap2_oracle1_regret010 | 1.0 | 2.0 | 2.0 | 0.10 |

## Training Süreci

İki model de aynı synthetic balanced training tablosundan eğitildi. Önce synthetic_train splitinde model fit edildi, synthetic_val üzerinde model seçimi yapıldı, ardından final model synthetic_train ve synthetic_val birleşimiyle tekrar eğitilip kaydedildi. Synthetic_test spliti eğitimde kullanılmadı ve held-out test olarak saklandı.

Training sırasında her dataset için 56 olası top3 kombinasyonu oluşturuldu. Her kombinasyonun gerçek algoritma run sonuçlarına göre reward değeri hesaplandı. Modelin görevi, dataset featureları ve kombinasyon encoding featurelarını kullanarak bu reward değerini tahmin etmekti. Inference sırasında en yüksek tahmini reward alan top3 kombinasyonu model çıktısı oldu.

## Synthetic Validation ve Test Sonuçları

| model | split | top1_hit | oracle_in_top3 | top3_overlap>=2 | avg_top3_overlap | regret_at_3 | rank_spearman | rank_kendall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle4_overlap1_regret025 | synthetic_val | 0.680 | 0.909 | 0.726 | 1.800 | 0.189 | 0.512 | 0.419 |
| oracle4_overlap1_regret025 | synthetic_test | 0.750 | 0.920 | 0.750 | 1.801 | 0.318 | 0.548 | 0.459 |
| overlap2_oracle1_regret010 | synthetic_val | 0.646 | 0.909 | 0.749 | 1.863 | 0.171 | 0.565 | 0.465 |
| overlap2_oracle1_regret010 | synthetic_test | 0.676 | 0.898 | 0.795 | 1.949 | 0.233 | 0.616 | 0.513 |

Synthetic test yorumu:

- oracle4_overlap1_regret025 synthetic testte oracle coverage tarafında güçlü: oracle_in_top3=0.920, regret_at_3=0.318.
- overlap2_oracle1_regret010 synthetic top3 benzerliği tarafında güçlü: avg_top3_overlap=1.949, top3_overlap>=2=0.795; regret_at_3=0.233 seviyesinde kalıyor.

## Exact Evaluation Dataset Profili

| dataset | node | sample | sample/node | avg_cardinality | max_cardinality | proxy_density | oracle_best | ground_truth_top3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alarm | 37 | 10000 | 270.270 | 2.838 | 4 | 0.174 | GRaSP, PC_discrete | PC_discrete, GRaSP, GES |
| asia | 8 | 5000 | 625.000 | 2.000 | 2 | 0.250 | HC, Tabu | HC, Tabu, GES |
| barley | 48 | 4000 | 83.333 | 8.458 | 67 | 0.074 | PC_discrete | PC_discrete, FCI, BOSS |
| cancer | 5 | 5000 | 1000.000 | 2.000 | 2 | 0.000 | BOSS | BOSS, HC, Tabu |
| cat_chain | 3 | 8000 | 2666.667 | 2.667 | 3 | 1.000 | HC, Tabu | HC, Tabu, BOSS |
| cat_collider | 3 | 8000 | 2666.667 | 2.667 | 3 | 0.667 | PC_discrete | PC_discrete, BOSS, GRaSP |
| child | 20 | 5000 | 250.000 | 3.000 | 6 | 0.137 | K2 | K2, BOSS, FCI |
| earthquake | 5 | 5000 | 1000.000 | 2.000 | 2 | 0.800 | BOSS, GES, GRaSP, HC, K2, PC_discrete, Tabu | BOSS, GES, GRaSP |
| hailfinder | 56 | 5000 | 89.286 | 3.982 | 11 | 0.036 | PC_discrete | PC_discrete, HC, Tabu |
| hepar2 | 70 | 3000 | 42.857 | 2.314 | 4 | 0.006 | HC, Tabu | HC, Tabu, GES |
| insurance | 27 | 5000 | 185.185 | 3.259 | 5 | 0.242 | PC_discrete | PC_discrete, FCI, BOSS |
| mildew | 35 | 4000 | 114.286 | 15.057 | 88 | 0.234 | PC_discrete | PC_discrete, GES, HC |
| survey | 6 | 5000 | 833.333 | 2.333 | 3 | 0.000 | PC_discrete | PC_discrete, K2, HC |
| water | 32 | 5000 | 156.250 | 2.469 | 4 | 0.052 | BOSS | BOSS, GRaSP, K2 |

## Exact Evaluation Özeti

| model | objective | odak | top1_hit | oracle_in_top3 | top3_overlap>=2 | avg_top3_overlap | avg_regret_at_3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle4_overlap1_regret025 | top3 combination reward regression | oracle coverage ve düşük regret | 2/14 | 11/14 | 4/14 | 1.143 | 1.357 |
| overlap2_oracle1_regret010 | top3 combination reward regression | ground-truth top3 overlap | 4/14 | 8/14 | 8/14 | 1.500 | 3.357 |
| balanced_985_ranking | ranking | n/a | 5/14 | 8/14 | 6/14 | 1.357 | 3.786 |

Exact tablolarda kullanılan metrik açıklamaları:

- top1_hit: model top1 algoritması oracle algoritmalardan biri mi?
- oracle_in_top3: model top3 içinde oracle algoritmalardan biri var mı?
- top3_overlap: ground-truth top3 ile model top3 arasındaki ortak algoritma sayısı.
- top3_overlap>=2: model ground-truth top3 algoritmalarından en az 2 tanesini seçti mi?
- regret_at_3: model top3 içindeki en iyi algoritmanın SHD değeri ile oracle SHD arasındaki fark. Düşük daha iyi; 0 oracle top3 içinde demektir.

## Exact Evaluation Tablosu: oracle4_overlap1_regret025

| dataset | node | sample | ground_truth_top3 | oracle_best | model_top3 | pred_top1 | top1_hit | oracle_in_top3 | top3_overlap | top3_overlap>=2 | regret_at_3 | oracle_shd | best_top3_shd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alarm | 37 | 10000 | PC_discrete, GRaSP, GES | GRaSP, PC_discrete | K2, BOSS, PC_discrete | K2 | hayır | evet | 1 | hayır | 0.0 | 4.0 | 4.0 |
| asia | 8 | 5000 | HC, Tabu, GES | HC, Tabu | K2, Tabu, GES | K2 | hayır | evet | 2 | evet | 0.0 | 2.0 | 2.0 |
| barley | 48 | 4000 | PC_discrete, FCI, BOSS | PC_discrete | HC, BOSS, K2 | HC | hayır | hayır | 1 | hayır | 15.0 | 60.0 | 75.0 |
| cancer | 5 | 5000 | BOSS, HC, Tabu | BOSS | K2, PC_discrete, Tabu | K2 | hayır | hayır | 1 | hayır | 2.0 | 0.0 | 2.0 |
| cat_chain | 3 | 8000 | HC, Tabu, BOSS | HC, Tabu | K2, Tabu, GES | K2 | hayır | evet | 1 | hayır | 0.0 | 1.0 | 1.0 |
| cat_collider | 3 | 8000 | PC_discrete, BOSS, GRaSP | PC_discrete | K2, HC, Tabu | K2 | hayır | hayır | 0 | hayır | 2.0 | 0.0 | 2.0 |
| child | 20 | 5000 | K2, BOSS, FCI | K2 | K2, HC, BOSS | K2 | evet | evet | 2 | evet | 0.0 | 39.0 | 39.0 |
| earthquake | 5 | 5000 | BOSS, GES, GRaSP | BOSS, GES, GRaSP, HC, K2, PC_discrete, Tabu | K2, Tabu, PC_discrete | K2 | evet | evet | 0 | hayır | 0.0 | 0.0 | 0.0 |
| hailfinder | 56 | 5000 | PC_discrete, HC, Tabu | PC_discrete | K2, BOSS, PC_discrete | K2 | hayır | evet | 1 | hayır | 0.0 | 44.0 | 44.0 |
| hepar2 | 70 | 3000 | HC, Tabu, GES | HC, Tabu | K2, HC, PC_discrete | K2 | hayır | evet | 1 | hayır | 0.0 | 72.0 | 72.0 |
| insurance | 27 | 5000 | PC_discrete, FCI, BOSS | PC_discrete | GRaSP, PC_discrete, HC | GRaSP | hayır | evet | 1 | hayır | 0.0 | 25.0 | 25.0 |
| mildew | 35 | 4000 | PC_discrete, GES, HC | PC_discrete | K2, PC_discrete, BOSS | K2 | hayır | evet | 1 | hayır | 0.0 | 17.0 | 17.0 |
| survey | 6 | 5000 | PC_discrete, K2, HC | PC_discrete | K2, PC_discrete, BOSS | K2 | hayır | evet | 2 | evet | 0.0 | 3.0 | 3.0 |
| water | 32 | 5000 | BOSS, GRaSP, K2 | BOSS | K2, BOSS, Tabu | K2 | hayır | evet | 2 | evet | 0.0 | 40.0 | 40.0 |

### Dataset Bazlı Metrik Notları: oracle4_overlap1_regret025

- alarm (37 node): GT top3 PC_discrete, GRaSP, GES; model top3 K2, BOSS, PC_discrete. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=1: ground-truth top3'ten 1 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- asia (8 node): GT top3 HC, Tabu, GES; model top3 K2, Tabu, GES. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=2: ground-truth top3'ten 2 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- barley (48 node): GT top3 PC_discrete, FCI, BOSS; model top3 HC, BOSS, K2. oracle_in_top3=hayır: seçilen top3 içinde gerçek en iyi algoritma yok. top3_overlap=1: ground-truth top3'ten 1 algoritma yakalandı. regret_at_3=15.0: seçilen top3 içindeki en iyi algoritma oracle'dan 15.0 SHD uzak.
- cancer (5 node): GT top3 BOSS, HC, Tabu; model top3 K2, PC_discrete, Tabu. oracle_in_top3=hayır: seçilen top3 içinde gerçek en iyi algoritma yok. top3_overlap=1: ground-truth top3'ten 1 algoritma yakalandı. regret_at_3=2.0: seçilen top3 içindeki en iyi algoritma oracle'dan 2.0 SHD uzak.
- cat_chain (3 node): GT top3 HC, Tabu, BOSS; model top3 K2, Tabu, GES. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=1: ground-truth top3'ten 1 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- cat_collider (3 node): GT top3 PC_discrete, BOSS, GRaSP; model top3 K2, HC, Tabu. oracle_in_top3=hayır: seçilen top3 içinde gerçek en iyi algoritma yok. top3_overlap=0: ground-truth top3'ten 0 algoritma yakalandı. regret_at_3=2.0: seçilen top3 içindeki en iyi algoritma oracle'dan 2.0 SHD uzak.
- child (20 node): GT top3 K2, BOSS, FCI; model top3 K2, HC, BOSS. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=2: ground-truth top3'ten 2 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- earthquake (5 node): GT top3 BOSS, GES, GRaSP; model top3 K2, Tabu, PC_discrete. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=0: ground-truth top3'ten 0 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- hailfinder (56 node): GT top3 PC_discrete, HC, Tabu; model top3 K2, BOSS, PC_discrete. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=1: ground-truth top3'ten 1 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- hepar2 (70 node): GT top3 HC, Tabu, GES; model top3 K2, HC, PC_discrete. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=1: ground-truth top3'ten 1 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- insurance (27 node): GT top3 PC_discrete, FCI, BOSS; model top3 GRaSP, PC_discrete, HC. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=1: ground-truth top3'ten 1 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- mildew (35 node): GT top3 PC_discrete, GES, HC; model top3 K2, PC_discrete, BOSS. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=1: ground-truth top3'ten 1 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- survey (6 node): GT top3 PC_discrete, K2, HC; model top3 K2, PC_discrete, BOSS. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=2: ground-truth top3'ten 2 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- water (32 node): GT top3 BOSS, GRaSP, K2; model top3 K2, BOSS, Tabu. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=2: ground-truth top3'ten 2 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.

## Exact Evaluation Tablosu: overlap2_oracle1_regret010

| dataset | node | sample | ground_truth_top3 | oracle_best | model_top3 | pred_top1 | top1_hit | oracle_in_top3 | top3_overlap | top3_overlap>=2 | regret_at_3 | oracle_shd | best_top3_shd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alarm | 37 | 10000 | PC_discrete, GRaSP, GES | GRaSP, PC_discrete | BOSS, K2, GES | BOSS | hayır | hayır | 1 | hayır | 14.0 | 4.0 | 18.0 |
| asia | 8 | 5000 | HC, Tabu, GES | HC, Tabu | BOSS, K2, HC | BOSS | hayır | evet | 1 | hayır | 0.0 | 2.0 | 2.0 |
| barley | 48 | 4000 | PC_discrete, FCI, BOSS | PC_discrete | HC, Tabu, BOSS | HC | hayır | hayır | 1 | hayır | 15.0 | 60.0 | 75.0 |
| cancer | 5 | 5000 | BOSS, HC, Tabu | BOSS | K2, BOSS, HC | K2 | hayır | evet | 2 | evet | 0.0 | 0.0 | 0.0 |
| cat_chain | 3 | 8000 | HC, Tabu, BOSS | HC, Tabu | K2, HC, Tabu | K2 | hayır | evet | 2 | evet | 0.0 | 1.0 | 1.0 |
| cat_collider | 3 | 8000 | PC_discrete, BOSS, GRaSP | PC_discrete | K2, HC, Tabu | K2 | hayır | hayır | 0 | hayır | 2.0 | 0.0 | 2.0 |
| child | 20 | 5000 | K2, BOSS, FCI | K2 | K2, BOSS, Tabu | K2 | evet | evet | 2 | evet | 0.0 | 39.0 | 39.0 |
| earthquake | 5 | 5000 | BOSS, GES, GRaSP | BOSS, GES, GRaSP, HC, K2, PC_discrete, Tabu | K2, BOSS, GRaSP | K2 | evet | evet | 2 | evet | 0.0 | 0.0 | 0.0 |
| hailfinder | 56 | 5000 | PC_discrete, HC, Tabu | PC_discrete | K2, HC, Tabu | K2 | hayır | hayır | 2 | evet | 9.0 | 44.0 | 53.0 |
| hepar2 | 70 | 3000 | HC, Tabu, GES | HC, Tabu | HC, K2, Tabu | HC | evet | evet | 2 | evet | 0.0 | 72.0 | 72.0 |
| insurance | 27 | 5000 | PC_discrete, FCI, BOSS | PC_discrete | HC, K2, GRaSP | HC | hayır | hayır | 0 | hayır | 6.0 | 25.0 | 31.0 |
| mildew | 35 | 4000 | PC_discrete, GES, HC | PC_discrete | K2, HC, PC_discrete | K2 | hayır | evet | 2 | evet | 0.0 | 17.0 | 17.0 |
| survey | 6 | 5000 | PC_discrete, K2, HC | PC_discrete | K2, BOSS, GES | K2 | hayır | hayır | 1 | hayır | 1.0 | 3.0 | 4.0 |
| water | 32 | 5000 | BOSS, GRaSP, K2 | BOSS | BOSS, K2, GRaSP | BOSS | evet | evet | 3 | evet | 0.0 | 40.0 | 40.0 |

### Dataset Bazlı Metrik Notları: overlap2_oracle1_regret010

- alarm (37 node): GT top3 PC_discrete, GRaSP, GES; model top3 BOSS, K2, GES. oracle_in_top3=hayır: seçilen top3 içinde gerçek en iyi algoritma yok. top3_overlap=1: ground-truth top3'ten 1 algoritma yakalandı. regret_at_3=14.0: seçilen top3 içindeki en iyi algoritma oracle'dan 14.0 SHD uzak.
- asia (8 node): GT top3 HC, Tabu, GES; model top3 BOSS, K2, HC. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=1: ground-truth top3'ten 1 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- barley (48 node): GT top3 PC_discrete, FCI, BOSS; model top3 HC, Tabu, BOSS. oracle_in_top3=hayır: seçilen top3 içinde gerçek en iyi algoritma yok. top3_overlap=1: ground-truth top3'ten 1 algoritma yakalandı. regret_at_3=15.0: seçilen top3 içindeki en iyi algoritma oracle'dan 15.0 SHD uzak.
- cancer (5 node): GT top3 BOSS, HC, Tabu; model top3 K2, BOSS, HC. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=2: ground-truth top3'ten 2 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- cat_chain (3 node): GT top3 HC, Tabu, BOSS; model top3 K2, HC, Tabu. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=2: ground-truth top3'ten 2 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- cat_collider (3 node): GT top3 PC_discrete, BOSS, GRaSP; model top3 K2, HC, Tabu. oracle_in_top3=hayır: seçilen top3 içinde gerçek en iyi algoritma yok. top3_overlap=0: ground-truth top3'ten 0 algoritma yakalandı. regret_at_3=2.0: seçilen top3 içindeki en iyi algoritma oracle'dan 2.0 SHD uzak.
- child (20 node): GT top3 K2, BOSS, FCI; model top3 K2, BOSS, Tabu. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=2: ground-truth top3'ten 2 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- earthquake (5 node): GT top3 BOSS, GES, GRaSP; model top3 K2, BOSS, GRaSP. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=2: ground-truth top3'ten 2 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- hailfinder (56 node): GT top3 PC_discrete, HC, Tabu; model top3 K2, HC, Tabu. oracle_in_top3=hayır: seçilen top3 içinde gerçek en iyi algoritma yok. top3_overlap=2: ground-truth top3'ten 2 algoritma yakalandı. regret_at_3=9.0: seçilen top3 içindeki en iyi algoritma oracle'dan 9.0 SHD uzak.
- hepar2 (70 node): GT top3 HC, Tabu, GES; model top3 HC, K2, Tabu. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=2: ground-truth top3'ten 2 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- insurance (27 node): GT top3 PC_discrete, FCI, BOSS; model top3 HC, K2, GRaSP. oracle_in_top3=hayır: seçilen top3 içinde gerçek en iyi algoritma yok. top3_overlap=0: ground-truth top3'ten 0 algoritma yakalandı. regret_at_3=6.0: seçilen top3 içindeki en iyi algoritma oracle'dan 6.0 SHD uzak.
- mildew (35 node): GT top3 PC_discrete, GES, HC; model top3 K2, HC, PC_discrete. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=2: ground-truth top3'ten 2 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.
- survey (6 node): GT top3 PC_discrete, K2, HC; model top3 K2, BOSS, GES. oracle_in_top3=hayır: seçilen top3 içinde gerçek en iyi algoritma yok. top3_overlap=1: ground-truth top3'ten 1 algoritma yakalandı. regret_at_3=1.0: seçilen top3 içindeki en iyi algoritma oracle'dan 1.0 SHD uzak.
- water (32 node): GT top3 BOSS, GRaSP, K2; model top3 BOSS, K2, GRaSP. oracle_in_top3=evet: seçilen top3 içinde gerçek en iyi algoritma var. top3_overlap=3: ground-truth top3'ten 3 algoritma yakalandı. regret_at_3=0.0: seçilen top3 içindeki en iyi algoritma oracle'dan 0.0 SHD uzak.

## Model Seçim Rehberi

oracle4_overlap1_regret025 şu durumda kullanılmalı:

- önerilen top3 içinde oracle seviyesinde en az bir algoritma olsun istiyorsak;
- top3 içindeki en iyi algoritmanın oracle SHD değerine yakın olmasını istiyorsak;
- full ground-truth top3 benzerliğinde düşüşü kabul edebiliyorsak.

overlap2_oracle1_regret010 şu durumda kullanılmalı:

- ground-truth top3 listesinden en az 2 algoritmayı yakalama öncelikliyse;
- top3 listesinin genel benzerliği önemliyse;
- oracle/regret modeline göre daha yüksek regret kabul edilebiliyorsa.

## Limitasyonlar ve Riskler

- Exact evaluation set sadece 14 dataset içeriyor. Bu metrikler yüksek sinyal taşıyor ama istatistiksel olarak küçük bir setten geliyor.
- Training verisi synthetic BN datasıdır. Synthetic-exact domain mismatch en büyük bilinen limitasyon olmaya devam ediyor.
- Bu iki final model learned encoder featurelarını kullanmaz. Encoder çalışması ayrıydı; burada final selectorlar handcrafted feature kullanır.
- Model dosyaları büyüktür: oracle/regret modeli yaklaşık 518 MB, overlap modeli yaklaşık 620 MB.
- Top3 combination yaklaşımı inference sırasında deterministiktir; training tarafında sabit random seed kullanılmıştır.
- Exact datasetler yalnızca evaluation için kullanıldı, model fit aşamasına dahil edilmedi.
