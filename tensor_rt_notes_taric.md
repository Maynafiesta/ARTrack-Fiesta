# ARTrackV2 - TensorRT Optimizasyon ve Dağıtım Notları

Bu belge, sıfırdan klonlanan ARTrackV2 modelinin bir bilgisayarda çalıştırılması, ONNX formatına dönüştürülmesi ve TensorRT motoruyla yüksek FPS değerlerine ulaştırılması sürecindeki tüm adımları, kod değişikliklerini ve karşılaşılan hataların çözümlerini kaydeder.

---

## 1. Sistemin Kurulması ve İlk Hataların Giderilmesi (PyTorch Baseline)

Dümdüz repoyu klonlayıp çalıştırdığımızda karşılaştığımız hatalar ve çözüm yolları:

### A) `torch._six` Hatası
PyTorch 2.0 ve sonrasında `torch._six` kütüphanesi tamamen kaldırıldığı için `string_classes` importu hata veriyordu.
*   **Değiştirilen Dosya:** `lib/train/data/processing_utils.py`
*   **Değişim (Diff):**
```python
- from torch._six import string_classes
+ string_classes = str
```

### B) `weights_only=True` Güvenlik Engeli Neden Gerekliydi? Ne Yaptık?
PyTorch 2.6 ve sonrasında, internetten indirilen `.pth` veya `.tar` uzantılı model ağırlık dosyalarının içerisine kötü niyetli Pickle kodları gizlenebildiği için, PyTorch varsayılan olarak **sadece veriyi (tensörleri) yükleme** kuralı getirdi (`weights_only=True`). 
ARTrack reposunda yazılımcılar, modeli kaydederken konfigürasyon dict'lerini ve özel objeleri de ağırlık dosyasına gömmüşler. Bu yüzden PyTorch yüklemeyi reddediyordu.
*   **Değiştirilen Kritik Dosyalar:** 
    *   `lib/test/evaluation/tracker.py`
    *   `lib/models/artrack_seq/artrack_seq.py`
    *   `tracking/video_demo.py`
*   **Değişim (Diff):**
`torch.load` fonksiyonlarını bularak yanlarına `, weights_only=False` parametresini açıkça ekledik ve tehlikeyi göze aldık (çünkü modeli doğrudan orijinal kaynaktan indirdik). Örnek:
```python
- checkpoint = torch.load(network_path, map_location='cpu')
+ checkpoint = torch.load(network_path, map_location='cpu', weights_only=False)
```

### C) Blackwell (RTX 5070 Ti) CUDA Uyumsuzluğu
Repodaki eski PyTorch sürümü `sm_120` (Ada Lovelace/Blackwell mimarileri) desteklemediği için gece yarısı derlemesine (nightly) ve CUDA 12.8'e yükseltmemiz gerekti.
*   **Terminal Komutu:**
```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## 2. ONNX Dönüşümü

### ONNX Boyut Sınırı ve `--no-large-tensor` Komutu
*   **2GB Sınırı Nedir?** ONNX formatı, modellerin yapısını "Protobuf" isimli veri standartında saklar. Protobuf'un çekirdek mimarisinde hiçbir verinin veya grafın boyutu 2 Gigabyte'ı aşamaz.
*   **Neden Hata Aldık?** Bu bilgisayarın (32GB RAM vb.) donanım yetersizliğinden değil, Protobuf'un yazılımsal kuralından kaynaklanır. Large modelin ağırlıkları kendi başına ~2GB tuttuğu için dönüştürürken bu limiti delip çöküyordu.
*   **Çözüm ve `--no-large-tensor` Etkisi:** `onnxsim` aracı çalışırken büyük tensör verilerini ONNX ağacının içine gömmez (RAM patlamasını engeller), sadece dışarıdan işaret eder. Modelin doğruluğunda hiçbir negatif yan etkisi yoktur.

```bash
onnxsim artrack_seq_large_384_full.onnx artrack_seq_large_384_full_sim.onnx --no-large-tensor
```

### ONNX Doğrulama İşlemi
Modelin dönüşüm sırasında bozulmadığını matematiksel olarak ispatlamak için `verify_onnx.py` yazdık.
*   **Terminal Komutu:**
```bash
~/.artrack_venv/bin/python verify_onnx.py
```
*   **Beklediğimiz Çıktı:** PyTorch modeli ile ONNX modeline aynı sahte tensör (image) beslendiğinde her iki çıkış matrisini karşılaştırıp `Max Absolute Difference: 0.0` sonucuna (sıfır hata) varmasını bekledik, ve başarılı oldu.

---

## 3. TensorRT (TRT) Dönüşümü ve Derleme Aşaması

TensorRT motorlarını (`.engine`) TRT'nin kendi sağladığı C++ derleyicisi olan **`trtexec`** terminal komutuyla derledik.

### 🔴 FP16 Numerik Taşma (Overflow) Sorunu (Bozuk Kod)
Başlarda hızı 3 kat artırmak için modelleri Half-Precision (`--fp16`) yetkisiyle derledik:
```bash
trtexec --onnx=artrack_seq_256_full_sim.onnx --saveEngine=artrack_seq_256_full.engine --fp16
```
*   **Sonuç:** Model mükemmel derlendi, 200 FPS hızlara fırladı ancak videoyu açtığımızda kutu **ekranın sağ alt köşesine** kaçıyordu.
*   **Sebep:** FP16 kapasite gereği ~65,504 değerine kadar sayabilir. ViT mimarisi ve Transformer Attention katmanları çok yüksek sayılar üretince limit aşıldı (`NaN`). Matematik çökünce argmax komutu rastgele 637 indeksini verip sağ altta kilitlendi.

### ✅ NİHAİ DOĞRU ÇÖZÜM: FP32 Motor Derlemeleri
Taşmayı önlemek için `--fp16` bayrağını sildik ve 32-bit (FP32) motorlar derleyerek isabet oranını tamamen stabil kıldık:

**Base (256) Modeli İçin (157 FPS):**
```bash
trtexec --onnx=artrack_seq_256_full_sim.onnx --saveEngine=artrack_seq_256_full_fp32.engine --timingCacheFile=timing.cache
```

**Large (384) Modeli İçin (42 FPS):**
```bash
trtexec --onnx=artrack_seq_large_384_full_sim.onnx --saveEngine=artrack_seq_large_384_full_fp32.engine --timingCacheFile=timing_large.cache
```

### Python Tarafında TRT Çalıştırma
TRT motorunu C++ mimarisinden alıp Python'a bağlayabilmek için iki betik (`trt_wrapper.py` ve `trt_demo.py`) tasarladık. Motoru çağırmak için final komutumuz:
```bash
~/.artrack_venv/bin/python trt_demo.py ~/Videos/officeRecords110624/record9.mkv
```
