# Jetson AGX Deployment Guide: ARTrack-Fiesta

Bu doküman, ARTrack-Fiesta projesinin Jetson AGX üzerinde Docker ve TensorRT kullanılarak devreye alınması için gereken adımları içerir.

## Hazırlık ve Kurulum

1. **Projenin Klonlanması:**
   ```bash
   git clone <repo_url>
   cd ARTrack-Fiesta
   ```

2. **ONNX Dosyalarının Hazırlanması:**
   Daha önce oluşturulan `_sim.onnx` dosyaları `Weights/ONNX/` dizinine kopyalanmalıdır.

3. **Docker İmajının Oluşturulması:**
   Jetson AGX (ARM64) mimarisine uygun Docker yapılandırması kullanılır:
   ```bash
   docker compose -f docker-compose.jetson.yml build
   ```

---

## Çalıştırma Adımları

### 1. Konteynırın Başlatılması
```bash
# X11 yetkilendirmesi (Görsel çıktı için)
xhost +local:docker

# Servislerin başlatılması
docker compose -f docker-compose.jetson.yml up -d
```

### 2. Konteynır Erişimi
```bash
docker exec -it artrack_jetson bash
```

### 3. TensorRT Engine Oluşturma
Hassasiyet seçimine göre aşağıdaki komutları kullanabilirsiniz. FP32 en kararlı sonucu verirken, FP16 daha yüksek performans sunar.

**256 Full Model (FP32 - Önerilen):**
```bash
/usr/src/tensorrt/bin/trtexec --onnx=Weights/ONNX/artrack_seq_256_full_sim.onnx \
    --saveEngine=Weights/TensorRT/artrack_seq_256_full_fp32.engine \
    --workspace=4096
```

**256 Full Model (FP16):**
```bash
/usr/src/tensorrt/bin/trtexec --onnx=Weights/ONNX/artrack_seq_256_full_sim.onnx \
    --saveEngine=Weights/TensorRT/artrack_seq_256_full_fp16.engine \
    --fp16 --workspace=4096
```

**384 Large Model (FP32 - Önerilen):**
```bash
/usr/src/tensorrt/bin/trtexec --onnx=Weights/ONNX/artrack_seq_large_384_full_sim.onnx \
    --saveEngine=Weights/TensorRT/artrack_seq_large_384_full_fp32.engine \
    --workspace=4096
```

**384 Large Model (FP16):**
```bash
/usr/src/tensorrt/bin/trtexec --onnx=Weights/ONNX/artrack_seq_large_384_full_sim.onnx \
    --saveEngine=Weights/TensorRT/artrack_seq_large_384_full_fp16.engine \
    --fp16 --workspace=4096
```

### 4. Uygulamanın Çalıştırılması
```bash
python3 trt_demo.py record9.mkv
```

---

## Olası Sorunlar ve Çözümleri

### "No such file or directory: '.../local.py'" Hatası
Konteynır içerisindeki proje yollarını güncellemek için:
```bash
python3 tracking/create_default_local_file.py --workspace_dir /app --data_dir /app/data --save_dir /app/output
```

### "unknown or invalid runtime name: nvidia" Hatası
NVIDIA Container Toolkit kurulu değilse host sistemde şu komutlar çalıştırılmalıdır:
```bash
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### "Illegal instruction (core dumped)" Hatası
Numpy sürüm uyumsuzluğu durumunda konteynır içinde sürüm düşürülmelidir:
```bash
pip3 install numpy==1.23.5
```

### "Cannot open display :0" Hatası
Görsel çıktı hatası alınıyorsa host sistemde `xhost +local:docker` komutunun çalıştırıldığı teyit edilmelidir.
