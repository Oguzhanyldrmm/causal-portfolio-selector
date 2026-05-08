# Synthetic Generation Parametreleri ve Ground Truth Algoritma Analizi


Bu rapor, sentetik Bayesian-network datasetleri oluşturulurken kullanılan parametreler ile algoritmaların gerçek performansı arasındaki ilişkiyi inceler. Amaç, generation metadata ile ground-truth algoritma sonuçları arasında sistematik bir ilişki olup olmadığını görmek ve bu parametrelerin selector traininginde feature olarak kullanılıp kullanılamayacağını değerlendirmektir.

## Kısa Sonuç

- Evet, generation parametreleri ile ground-truth algoritma başarısı arasında belirgin ilişki var. Graph family, node sayısı, sample sayısı, density, edge count ve alpha profile algoritma dağılımlarını etkiliyor.
- En güçlü sinyal, mevcut sentetik sette K2 algoritmasının çok baskın olması. Bu, synthetic train dağılımının bazı algoritmaları doğal olarak avantajlı hale getirdiğini gösteriyor.
- Generation parametrelerini doğrudan final selector featureı yapmak çoğu durumda doğru değil. Çünkü graph_family, alpha_profile, true DAG density, edge_count ve seed gibi bilgiler gerçek/evaluation datasetlerde gözlemlenebilir değil ya da ground-truth graph bilgisi gerektiriyor.
- Ancak bu parametreler training seti dengelemek, stratified evaluation yapmak, sample weighting uygulamak ve synthetic-exact mismatch teşhisi için çok faydalı.
- Feature olarak kullanılabilecek kısım, sadece ham datasetten çıkarılabilen gözlenebilir karşılıklar: n_samples, n_features, cardinality istatistikleri ve proxy graph density gibi mevcut handcrafted featurelar. Bunların çoğu zaten selector feature setinde var.

## Kullanılan Veri

- Complete synthetic dataset sayısı: 1753
- Target row sayısı: 14024
- Algoritma sayısı: 8
- Balanced train subset sayısı: 985

Analiz, 8 algoritmanın tamamında başarılı run sonucu olan complete synthetic set üzerinde yapıldı. Balanced train subset ayrıca karşılaştırma için kullanıldı.

## Üretimde Kullanılan Parametreler

| Parametre | Açıklama | Feature olarak doğrudan kullanılabilir mi? |
| --- | --- | --- |
| graph_family | DAG üretim ailesi: erdos_renyi_sparse, erdos_renyi_dense, scale_free, layered_dag, chain_heavy, collider_heavy, hub_spoke | Hayır. Synthetic-only label; gerçek datasetlerde bilinmez. |
| n_features | Node/değişken sayısı | Evet. Zaten handcrafted feature olarak var. |
| n_samples | Gözlem/satır sayısı | Evet. Zaten handcrafted feature olarak var. |
| density | Gerçek DAG edge yoğunluğu | Doğrudan hayır. Ground-truth graph gerektirir. Proxy karşılığı kullanılabilir. |
| edge_count | Gerçek DAG edge sayısı | Doğrudan hayır. Ground-truth graph gerektirir. |
| max_indegree | Üretimde izin verilen maksimum indegree | Hayır. Bu sette sabit 4; ayrıca gerçek graph bilinmeden ölçülemez. |
| cardinalities | Node kategori sayıları | Evet. Ham veriden ölçülebilir; avg/max/cardinality_entropy zaten var. |
| alpha_profile | CPT keskinliği: sharp, medium, smooth | Doğrudan hayır. Synthetic-only latent parametre. |
| seed | Dataset üretim random seed’i | Hayır. Anlamsal feature değil. |

## Synthetic Set Dağılımı

### Graph Family
| graph_family | dataset_count |
| --- | --- |
| chain_heavy | 235 |
| collider_heavy | 249 |
| erdos_renyi_dense | 256 |
| erdos_renyi_sparse | 259 |
| hub_spoke | 240 |
| layered_dag | 261 |
| scale_free | 253 |

### Node Sayısı
| n_features | dataset_count |
| --- | --- |
| 4 | 200 |
| 6 | 200 |
| 8 | 250 |
| 10 | 300 |
| 15 | 280 |
| 20 | 255 |
| 30 | 169 |
| 40 | 99 |

### Sample Sayısı
| n_samples | dataset_count |
| --- | --- |
| 500 | 492 |
| 1000 | 470 |
| 3000 | 413 |
| 5000 | 378 |

### Alpha Profile
| alpha_profile | dataset_count |
| --- | --- |
| medium | 600 |
| sharp | 497 |
| smooth | 656 |

## Genel Algoritma Performansı

oracle_rate, ilgili algoritmanın datasetlerde oracle seti içinde bulunma oranıdır. top3_rate, ilgili algoritmanın ground-truth top3 içinde bulunma oranıdır. Tielar nedeniyle oracle_rate toplamı 100% olmak zorunda değildir.

| algorithm_name | oracle_rate | top3_rate | mean_rank | mean_relative_regret | mean_shd | median_shd |
| --- | --- | --- | --- | --- | --- | --- |
| PC_discrete | 0.1426 | 0.2288 | 5.2504 | 0.0739 | 16.4638 | 8.0000 |
| FCI | 0.0257 | 0.0913 | 6.4912 | 0.1175 | 19.7570 | 10.0000 |
| GES | 0.0947 | 0.2276 | 4.8939 | 0.0840 | 16.7074 | 8.0000 |
| HC | 0.3035 | 0.6811 | 2.9276 | 0.0509 | 14.7912 | 7.0000 |
| Tabu | 0.3035 | 0.5151 | 4.2881 | 0.0509 | 14.7912 | 7.0000 |
| K2 | 0.7239 | 0.7867 | 2.3691 | 0.0191 | 12.3172 | 5.0000 |
| BOSS | 0.1038 | 0.3366 | 3.9977 | 0.0875 | 16.9646 | 8.0000 |
| GRaSP | 0.0936 | 0.1329 | 5.7821 | 0.0877 | 17.1010 | 8.0000 |

### Primary Oracle Dağılımı

Primary oracle, tie durumlarında quality_rank sıralamasına göre tek bir algoritma seçilerek oluşturuldu. Bu tablo association analizini kolaylaştırmak için kullanıldı; gerçek oracle seti tie içerebilir.

| algorithm | primary_oracle_count | rate |
| --- | --- | --- |
| PC_discrete | 68 | 0.0388 |
| FCI | 2 | 0.0011 |
| GES | 51 | 0.0291 |
| HC | 443 | 0.2527 |
| Tabu | 0 | 0.0000 |
| K2 | 980 | 0.5590 |
| BOSS | 182 | 0.1038 |
| GRaSP | 27 | 0.0154 |

## Parametreler ile Primary Oracle İlişkisi

Aşağıdaki tablo Cramer’s V ile generation parametresi ve primary_oracle arasındaki ilişki gücünü gösterir. 0 ilişki yok demektir; 1 mükemmel ilişki demektir. Bu değer nedensellik değil, dağılımsal association ölçer.

| parameter | cramers_v_vs_primary_oracle |
| --- | --- |
| alpha_profile | 0.195 |
| edge_count_bin | 0.188 |
| graph_family | 0.132 |
| n_features | 0.127 |
| sample_node_ratio_bin | 0.122 |
| density_bin | 0.097 |
| avg_cardinality_bin | 0.096 |
| n_samples | 0.077 |

Yorum: Bu sette en yüksek association alpha_profile, edge_count, graph_family ve node sayısı tarafında görülüyor. Bu beklenen bir durum: algoritmaların performansı conditional dağılım keskinliği, graph yapısı ve graph boyutundan etkileniyor.

## Graph Family Bazında Oracle Dağılımı

| graph_family | n | oracle_rate_leader | oracle_rate_top3 |
| --- | --- | --- | --- |
| chain_heavy | 235 | K2 (57.0%) | K2 57.0%, HC 39.1%, Tabu 39.1% |
| collider_heavy | 249 | K2 (65.9%) | K2 65.9%, PC_discrete 23.3%, HC 21.3% |
| erdos_renyi_dense | 256 | K2 (80.1%) | K2 80.1%, HC 23.8%, Tabu 23.8% |
| erdos_renyi_sparse | 259 | K2 (69.5%) | K2 69.5%, HC 41.7%, Tabu 41.7% |
| hub_spoke | 240 | K2 (82.1%) | K2 82.1%, HC 40.0%, Tabu 40.0% |
| layered_dag | 261 | K2 (78.2%) | K2 78.2%, HC 24.9%, Tabu 24.9% |
| scale_free | 253 | K2 (73.1%) | K2 73.1%, HC 22.5%, Tabu 22.5% |

## Graph Family Bazında En Düşük Ortalama Regret

| graph_family | lowest_mean_regret | top3_low_regret |
| --- | --- | --- |
| chain_heavy | K2 (0.040) | K2 0.040, HC 0.053, Tabu 0.053 |
| collider_heavy | K2 (0.024) | K2 0.024, HC 0.066, Tabu 0.066 |
| erdos_renyi_dense | K2 (0.012) | K2 0.012, HC 0.052, Tabu 0.052 |
| erdos_renyi_sparse | K2 (0.013) | K2 0.013, HC 0.030, Tabu 0.030 |
| hub_spoke | K2 (0.010) | K2 0.010, HC 0.030, Tabu 0.030 |
| layered_dag | K2 (0.012) | K2 0.012, HC 0.046, Tabu 0.046 |
| scale_free | K2 (0.024) | K2 0.024, HC 0.080, Tabu 0.080 |

## Node Sayısı Bazında Oracle Dağılımı

| n_features | n | oracle_rate_leader | oracle_rate_top3 |
| --- | --- | --- | --- |
| 4.0 | 200 | K2 (85.0%) | K2 85.0%, HC 54.5%, Tabu 54.5% |
| 6.0 | 200 | K2 (71.5%) | K2 71.5%, HC 43.0%, Tabu 43.0% |
| 8.0 | 250 | K2 (71.2%) | K2 71.2%, HC 35.6%, Tabu 35.6% |
| 10.0 | 300 | K2 (65.7%) | K2 65.7%, HC 29.3%, Tabu 29.3% |
| 15.0 | 280 | K2 (68.2%) | K2 68.2%, HC 24.6%, Tabu 24.6% |
| 20.0 | 255 | K2 (67.8%) | K2 67.8%, HC 20.4%, Tabu 20.4% |
| 30.0 | 169 | K2 (76.9%) | K2 76.9%, HC 17.8%, Tabu 17.8% |
| 40.0 | 99 | K2 (87.9%) | K2 87.9%, HC 9.1%, Tabu 9.1% |

## Node Sayısı Bazında En Düşük Ortalama Regret

| n_features | lowest_mean_regret | top3_low_regret |
| --- | --- | --- |
| 4.0 | K2 (0.034) | K2 0.034, HC 0.105, Tabu 0.105 |
| 6.0 | K2 (0.041) | K2 0.041, HC 0.081, Tabu 0.081 |
| 8.0 | K2 (0.023) | K2 0.023, HC 0.066, Tabu 0.066 |
| 10.0 | K2 (0.024) | K2 0.024, HC 0.049, Tabu 0.049 |
| 15.0 | K2 (0.011) | K2 0.011, HC 0.035, Tabu 0.035 |
| 20.0 | K2 (0.008) | K2 0.008, HC 0.027, Tabu 0.027 |
| 30.0 | K2 (0.003) | K2 0.003, HC 0.016, Tabu 0.016 |
| 40.0 | K2 (0.001) | K2 0.001, HC 0.012, Tabu 0.012 |

## Sample Sayısı Bazında Oracle Dağılımı

| n_samples | n | oracle_rate_leader | oracle_rate_top3 |
| --- | --- | --- | --- |
| 500.0 | 492 | K2 (69.5%) | K2 69.5%, HC 31.9%, Tabu 31.9% |
| 1000.0 | 470 | K2 (75.7%) | K2 75.7%, HC 28.3%, Tabu 28.3% |
| 3000.0 | 413 | K2 (73.8%) | K2 73.8%, HC 28.8%, Tabu 28.8% |
| 5000.0 | 378 | K2 (70.4%) | K2 70.4%, HC 32.5%, Tabu 32.5% |

## Sample Sayısı Bazında En Düşük Ortalama Regret

| n_samples | lowest_mean_regret | top3_low_regret |
| --- | --- | --- |
| 500.0 | K2 (0.018) | K2 0.018, HC 0.044, Tabu 0.044 |
| 1000.0 | K2 (0.018) | K2 0.018, HC 0.050, Tabu 0.050 |
| 3000.0 | K2 (0.019) | K2 0.019, HC 0.054, Tabu 0.054 |
| 5000.0 | K2 (0.023) | K2 0.023, HC 0.058, Tabu 0.058 |

## Alpha Profile Bazında Oracle Dağılımı

| alpha_profile | n | oracle_rate_leader | oracle_rate_top3 |
| --- | --- | --- | --- |
| medium | 600 | K2 (77.5%) | K2 77.5%, HC 27.7%, Tabu 27.7% |
| sharp | 497 | K2 (56.7%) | K2 56.7%, HC 39.2%, Tabu 39.2% |
| smooth | 656 | K2 (79.6%) | K2 79.6%, HC 26.1%, Tabu 26.1% |

## Alpha Profile Bazında En Düşük Ortalama Regret

| alpha_profile | lowest_mean_regret | top3_low_regret |
| --- | --- | --- |
| medium | K2 (0.015) | K2 0.015, HC 0.057, Tabu 0.057 |
| sharp | K2 (0.034) | K2 0.034, HC 0.053, Tabu 0.053 |
| smooth | K2 (0.012) | K2 0.012, HC 0.043, Tabu 0.043 |

## Density ve Cardinality Kırılımları

Density gerçek DAG yoğunluğudur. Final inference sırasında doğrudan kullanılamaz, çünkü gerçek graph bilinmez. Cardinality ise ham veriden ölçülebilir.

### Density Bin Bazında Oracle Dağılımı
| density_bin_label | n | oracle_rate_leader | oracle_rate_top3 |
| --- | --- | --- | --- |
| (0.0212, 0.114] | 442 | K2 (74.7%) | K2 74.7%, HC 35.7%, Tabu 35.7% |
| (0.114, 0.167] | 491 | K2 (76.8%) | K2 76.8%, HC 31.8%, Tabu 31.8% |
| (0.167, 0.253] | 384 | K2 (68.0%) | K2 68.0%, HC 22.9%, Tabu 22.9% |
| (0.253, 1.0] | 436 | K2 (69.0%) | K2 69.0%, HC 29.8%, Tabu 29.8% |

### Cardinality Bin Bazında Oracle Dağılımı
| avg_cardinality_bin_label | n | oracle_rate_leader | oracle_rate_top3 |
| --- | --- | --- | --- |
| (2.249, 3.85] | 595 | K2 (70.3%) | K2 70.3%, HC 36.8%, Tabu 36.8% |
| (3.85, 4.3] | 613 | K2 (72.9%) | K2 72.9%, HC 23.8%, Tabu 23.8% |
| (4.3, 6.333] | 545 | K2 (74.1%) | K2 74.1%, HC 30.6%, Tabu 30.6% |

## Numeric Parametreler ile Relative Regret Korelasyonu

Aşağıdaki tablo en güçlü 20 Spearman korelasyonunu gösterir. Pozitif değer parametre arttıkça ilgili algoritmanın relative regret değerinin arttığını, yani ortalama olarak kötüleştiğini ima eder. Negatif değer parametre arttıkça regretin azaldığını ima eder.

| algorithm | parameter | spearman_vs_relative_regret |
| --- | --- | --- |
| FCI | density | 0.618 |
| FCI | n_features | -0.616 |
| FCI | sample_node_ratio | 0.525 |
| GES | n_features | -0.523 |
| GRaSP | n_features | -0.518 |
| BOSS | n_features | -0.516 |
| GRaSP | density | 0.480 |
| BOSS | density | 0.478 |
| GES | density | 0.459 |
| FCI | edge_count | -0.432 |
| PC_discrete | n_features | -0.415 |
| GES | edge_count | -0.386 |
| GRaSP | edge_count | -0.370 |
| BOSS | edge_count | -0.369 |
| BOSS | sample_node_ratio | 0.336 |
| GRaSP | sample_node_ratio | 0.331 |
| HC | density | 0.328 |
| Tabu | density | 0.328 |
| PC_discrete | sample_node_ratio | 0.325 |
| PC_discrete | edge_count | -0.323 |

## Balanced 985 Subset ve Complete 1753 Karşılaştırması

Balanced subset oluşturulurken 1753 complete setin tamamı korunmadı; 985 datasetlik daha dengeli train subset seçildi. Aşağıdaki tablo primary oracle dağılımının complete set ve balanced train subset arasındaki farkını gösterir.

| algorithm | complete_primary_oracle_count | complete_rate | balanced_primary_oracle_count | balanced_rate |
| --- | --- | --- | --- | --- |
| PC_discrete | 68 | 0.0388 | 59 | 0.0599 |
| FCI | 2 | 0.0011 | 2 | 0.0020 |
| GES | 51 | 0.0291 | 39 | 0.0396 |
| HC | 443 | 0.2527 | 135 | 0.1371 |
| Tabu | 0 | 0.0000 | 0 | 0.0000 |
| K2 | 980 | 0.5590 | 580 | 0.5888 |
| BOSS | 182 | 0.1038 | 150 | 0.1523 |
| GRaSP | 27 | 0.0154 | 20 | 0.0203 |

## Bu Parametreleri Training Feature Olarak Kullanabilir Miyiz?

Kısa cevap: doğrudan çoğunu kullanmamalıyız. Bazılarını zaten kullanıyoruz.

### Doğrudan Kullanılması Mantıklı Olanlar

- n_samples ve n_features: ham veriden doğrudan bilinir, zaten feature setinde var.
- Cardinality istatistikleri: avg_cardinality, max_cardinality, cardinality_entropy gibi değerler ham veriden çıkarılabilir, zaten feature setinde var.
- Proxy graph yoğunluğu: gerçek DAG density değil ama verideki pairwise ilişkilerden çıkarılan yaklaşık density kullanılabilir, zaten proxy_graph_density olarak var.

### Doğrudan Kullanılması Mantıksız veya Riskli Olanlar

- graph_family: gerçek datasetlerde hangi synthetic graph family’ye ait olduğu bilinmez. Bunu feature yapmak train-test mismatch yaratır.
- alpha_profile: CPT üretim parametresidir; gerçek datasetlerde yoktur. Doğrudan feature olursa synthetic-only shortcut olur.
- true density ve edge_count: ground-truth graph bilinmeden hesaplanamaz. Inference aşamasında kullanmak leakage olur.
- max_indegree: bu synthetic sette sabittir ve gerçek graph bilinmeden doğrulanamaz.
- seed: model açısından anlamsal bilgi taşımaz.

## Daha Mantıklı Kullanım Yolları

1. Stratified training/evaluation: modelleri graph_family, node, sample, density, alpha_profile kırılımlarında ayrı ayrı raporlamak.
2. Sample weighting: sentetik train setinde aşırı baskın algoritma/graph bölgelerini ağırlıklandırarak dengelemek.
3. Dataset pruning: exact sete bakmadan, sadece synthetic parametre uzayındaki aşırı dominant bölgeleri budamak.
4. Auxiliary observable features: graph_family veya alpha_profile yerine, bunları ham veriden tahmin etmeye çalışan gözlenebilir proxy featurelar tasarlamak.
5. Model failure diagnosis: exact datasetlerde başarısız olduğumuz örnekleri, synthetic parametre uzayında hangi bölgeye benzediğiyle karşılaştırmak.

## Sonuç

Generation parametreleri ile ground-truth algoritma başarısı arasında anlamlı kantitatif ilişki var. Fakat bu parametrelerin çoğu gerçek inference senaryosunda bilinmez. Bu yüzden doğrudan model featureı yapmak yerine, training data tasarımı, dengeleme, stratified raporlama ve synthetic bias teşhisi için kullanılmaları daha doğru olur. Final selector feature setinde ise yalnızca ham datasetten gözlenebilir olan karşılıklar kullanılmalı: sample/node bilgisi, cardinality yapısı, pairwise dependency ölçümleri ve proxy graph istatistikleri.
