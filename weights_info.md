# İndirilen Model Ağırlıkları (Weights) Değerlendirmesi

Kısa cevap: **Yanlış değiller fakat ARTrackV2'nin kendi ağırlıkları değiller, OSTrack ağırlıkları.** 

`Weights` klasörünüzün içine baktım. Şu dosyalar mevcut:
- `OSTrack_ep0030.pth.tar`
- `OSTrack_ep0040.pth.tar`
- `largegot_OSTrack_ep0030.pth.tar`

## Durum Analizi:
ARTrack reposu, OSTrack isimli başka bir modelin altyapısı üzerine kurulmuştur (bunu README'nin sonunda "Acknowledgement" bölümünde belirtiyorlar). Bu nedenle kodun büyük kısmı OSTrack ağırlıklarını da yükleyebilecek veya ondan ön-eğitim (pre-train) alabilecek şekilde tasarlanmıştır.

Ancak **saf ARTrackV2 performansını görmek istiyorsanız**, bu ağırlıklar "ARTrackV2" modeline ait değildir (sadece onun temel aldığı eski OSTrack versiyonudur). 

Gerçek ARTrackV2 ağırlıklarının adları şöyledir (README'de Google Drive linkleri var):
- `ARTrack_ep0240.pth.tar` tarzında (veya `ARTrackV2-B-256`, `ARTrack-L-384` gibi).

## Bu ağırlıklarla demoyu çalıştırabilir miyiz?
Evet, çalıştırabiliriz. Ancak komutu verirken `artrack_seq` yerine `ostrack` (ve uygun parametresini) seçmeniz veya kodun içindeki yaml ayarlarını bu dosyalara yönlendirmeniz gerekir.

Eğer elinizdeki bu `OSTrack` ağırlıklarıyla test etmek istiyorsanız, komutu ona göre ayarlayabiliriz. Veya dilerseniz README'de yer alan gerçek `ARTrack-256` modelini indirip öyle deneyebilirsiniz. Karar sizin!
