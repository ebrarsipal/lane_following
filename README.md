🚗 Pekiştirmeli Öğrenme ile Şerit Takibi (RL Lane Following)
Bu proje, bir aracı dairesel bir pistte şeridini takip etmesi için eğitmeyi amaçlayan bir Pekiştirmeli Öğrenme (Reinforcement Learning - RL) simülasyonudur. Şerit takibi problemi, Derin Q Ağı (Deep Q-Network - DQN) algoritması kullanılarak çözülmüştür.

🌟 Temel Özellikler
Ortam: Özel olarak tasarlanmış, sürekli durum (Continuous State) ve ayrık eylem (Discrete Action) alanına sahip dairesel şerit takip ortamı (LaneFollowingCircleEnv).

Algoritma: Model tabanlı olmayan, değer tabanlı öğrenme algoritması olan DQN kullanılmıştır.

Görselleştirme: Eğitilmiş ajanın performansını Streamlit tabanlı bir arayüz ile görsel olarak izleme imkanı.

Teknolojiler: Python, PyTorch, Streamlit ve Matplotlib.

⚙️ Kurulum ve Çalıştırma
1. Ön Koşullar

Bu projeyi yerel ortamınızda çalıştırmak için aşağıdaki yazılımlara ihtiyacınız vardır:

Python 3.8+

Git

2. Depoyu Klonlama

Proje dosyalarını yerel makinenize indirin:

Bash
git clone https://github.com/ebrarsipal/lane-following-rl.git
cd rl_lane_following_v1
3. Sanal Ortam Oluşturma ve Etkinleştirme

Proje bağımlılıklarını izole etmek için bir sanal ortam oluşturun ve etkinleştirin:

Bash
# Sanal ortam oluşturma
python -m venv venv

# Windows için etkinleştirme
.\venv\Scripts\activate

# Linux/macOS için etkinleştirme
# source venv/bin/activate 
4. Bağımlılıkları Yükleme

Gerekli Python kütüphanelerini yükleyin (bir requirements.txt dosyasının mevcut olduğunu varsayar):

Bash
pip install -r requirements.txt
5. Model Dosyası

Eğitilmiş DQN modelinin (dqn_model.pth) projenin ana dizininde bulunduğundan emin olun.

🖥️ Simülasyonu Başlatma
Ajanın performansını görselleştirmek için Streamlit uygulamasını çalıştırın:

Bash
streamlit run streamlit_app_v2.py
Tarayıcınız otomatik olarak açılacak ve görselleştirme arayüzünü göreceksiniz.

Arayüz Kullanımı

Kontrol Alanı	Açıklama
Episodes (Bölümler)	Kaç simülasyon bölümü çalıştırmak istediğinizi ayarlar.
Show Trail (İzi Göster)	Aracın gittiği yolu gösteren izi açıp kapar.
Start Simulation	Eğitilmiş ajanı ortamda çalıştırmaya başlar.
Sağ Panel	Anlık adım, toplam ödül (reward) ve aracın konum/başlık (heading) bilgilerini gösterir.
🧠 Algoritma ve Ortam Detayları
Ortam: LaneFollowingCircleEnv

Durum Alanı (State): Aracın şeritten uzaklığı, yola göre açısı gibi 4 boyutlu sürekli vektör.

Eylem Alanı (Action): 3 adet ayrık eylem: Sola dön, Düz Git, Sağa dön.

Ödül (Reward): Şeridin merkezine yakın kalmak için pozitif ödül, pistten sapmak için negatif ödül ve ceza.

Ajan: DQNAgent

Bu projede kullanılan DQN ajanı, Q-tablosunun yerini alan bir sinir ağı (agent.model) kullanır. Eğitilmiş model, verilen duruma göre hangi eylemin en yüksek Q değerine sahip olduğunu belirler ve bu eylemi gerçekleştirir.

🤝 Katkıda Bulunma
Projenin geliştirilmesine katkıda bulunmaktan memnuniyet duyarız. Lütfen bir sorun (Issue) açmaktan veya bir Çekme İsteği (Pull Request) göndermekten çekinmeyin.

📄 Lisans
Bu proje MIT Lisansı altında yayımlanmıştır. (Lisans dosyanız mevcutsa, daha fazla ayrıntı için LICENSE dosyasına bakınız.)
