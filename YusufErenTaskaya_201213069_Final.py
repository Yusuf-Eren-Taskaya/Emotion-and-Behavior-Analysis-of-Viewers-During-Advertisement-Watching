import cv2
from keras.models import model_from_json
import numpy as np
import dlib
import matplotlib.pyplot as plt
import datetime
import tkinter as tk
from tkinter import messagebox, filedialog
import csv
from PIL import Image, ImageTk
import vlc
import keyboard

class EmotionDetectorApp:
    def __init__(self, master):
        self.master = master

        self.video_path = "video.mp4"  # Varsayılan video yoluq

        self.master.title("REKLAM FİLMLERİNİN İZLENMESİ SIRASINDA İZLEYİCİNİN DUYGU VE DAVRANIŞ ANALİZİ YusufErenTaşkaya")
        self.start_button = tk.Button(self.master, text="Reklam filmini başlatma", command=self.start_emotion_detector) #Start Emotion Detector
        self.start_button.pack()
        
        self.image = Image.open("resim.jpg")  # Burada kendi resim dosyanızın yolunu kullanın
        self.photo = ImageTk.PhotoImage(self.image)
        # Resmi içeren bir Label oluşturun
        self.image_label = tk.Label(root, image=self.photo)
        self.image_label.pack()

        self.extra_button = tk.Button(master, text="Anketi Başlat", command=self.anket)
        self.extra_button.pack()

        self.select_video_button = tk.Button(master, text="Video Seç", command=self.select_video)
        self.select_video_button.pack(pady=20)

        self.track_id_counter = 0  # Takipçi ID'si
        self.tut_dongu = 0 # Dongunun ne kadar dondugu

    def start_emotion_detector(self):
        # Video akışının hızını belirleme
        zaman_araligi = 0.5
        self.tut_dongu = 0 # Dongunun ne kadar dondugu

        # Önceden eğitilmiş modeli yükleme
        json_file = open("emotiondetectorr11.json", "r")
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights("emotiondetectorr11.h5")

        # Yüz tanıma için OpenCV'den modeli yükleme
        haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(haar_file)

        # dlib'in korelasyon takipçisini kullanarak takip
        trackers = []
        track_ids = []
        labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        emotion_counters = {}
        predicted_outputs = {}

        # En son görülen zamanları takip etmek için bir sözlük oluşturun
        last_seen = {}
        last_s_re = {}
        lseen = {}

        # Kamera yakalama
        webcam = cv2.VideoCapture(0)
       
        # VLC Media Player örneği oluştur
        instance = vlc.Instance('--no-xlib')
        # Media Player oluştur
        player = instance.media_player_new()
        # Videoyu yükle
        media = instance.media_new(self.video_path)
        # Media Player'a videoyu ata
        player.set_media(media)

        def add_tracker(frame, bbox):
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            tracker.start_track(frame, rect)
            trackers.append(tracker)
            track_ids.append(self.track_id_counter)
            # Her yeni takipçi için bir emotion_counter oluşturun
            emotion_counters[self.track_id_counter] = {label: 0 for label in labels.values()}
            predicted_outputs[self.track_id_counter] = [''] * self.tut_dongu  # tut_dongu  kadar boş eleman ekle
            # İzleyicinin en son görüldüğü zamanı kaydet
            last_seen[self.track_id_counter] = datetime.datetime.now()  # baslangic zamanı tutan dizi 
            last_s_re[self.track_id_counter] = datetime.datetime.now()  # son görülen zamanı tutan dizi
            lseen[self.track_id_counter] = datetime.datetime.now()  # Elemanların silinip silineceğine karar verilen dizi
            self.track_id_counter += 1

        def extract_features(image):
            feature = np.array(image)
            feature = feature.reshape(1, 48, 48, 1)
            return feature / 255.0

        def is_overlapping(x1, y1, w1, h1, x2, y2, w2, h2):
            # İki dikdörtgenin çakışıp çakışmadığını kontrol etme
            if (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2):
                return True
            return False

        def is_overlapping_tolerance(x1, y1, w1, h1, x2, y2, w2, h2, tolerance=10):
            # İki dikdörtgenin ±tolerance piksel hata payı ile örtüşüp örtüşmediğini kontrol eder.
            # Tolerans dahilinde genişlik ve yükseklik hesaplama
            left1 = x1 - tolerance
            right1 = x1 + w1 + tolerance
            top1 = y1 - tolerance
            bottom1 = y1 + h1 + tolerance

            left2 = x2 - tolerance
            right2 = x2 + w2 + tolerance
            top2 = y2 - tolerance
            bottom2 = y2 + h2 + tolerance

            # Örtüşme kontrolü
            if left1 < right2 and right1 > left2 and top1 < bottom2 and bottom1 > top2:
                return True
            return False

        while True:
            ret, frame = webcam.read()
            # Videoyu oynat
            player.play()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Görüntüyü gri tona dönüştürme
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Yüzleri algılama

            #current_time = datetime.datetime.now()

            tracker_sayisi = len(trackers) # Trackers listesindeki eleman sayısı:
            
            if len(faces) >= tracker_sayisi:
                # Yüz algılandığında geçen süreyi sıfırla
                #last_face_detection_time = time.time()

                for (x, y, w, h) in faces:
                    overlap_found = False
                    for tracker in trackers:
                        pos = tracker.get_position()
                        startX = int(pos.left())
                        startY = int(pos.top())
                        endX = int(pos.right())
                        endY = int(pos.bottom())
                        if is_overlapping(x, y, w, h, startX, startY, endX - startX, endY - startY):
                            overlap_found = True
                            # İzleyicinin en son görüldüğü zamanı güncelle
                            lseen[track_ids[trackers.index(tracker)]] = datetime.datetime.now()
                            break
                    if not overlap_found:
                        add_tracker(frame, (x, y, w, h))
            else:
                # Yüzler güncellendikten sonra mevcut yüzlerle eşleşmeyen yüzleri sil
                current_faces = [(x, y, w, h) for (x, y, w, h) in faces]
                keys_to_delete = []
                for track_id, last_seen_time in lseen.items():
                    tracker = trackers[track_ids.index(track_id)]
                    pos = tracker.get_position()
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                    face_still_present = False
                    for (x, y, w, h) in current_faces:
                        if is_overlapping_tolerance(x, y, w, h, startX, startY, endX - startX, endY - startY,10):
                            face_still_present = True
                            break
                    if not face_still_present:
                        time_since_last_seen = (datetime.datetime.now() - last_seen_time).total_seconds()
                        if time_since_last_seen >= 5:
                            keys_to_delete.append(track_id)
                for key in keys_to_delete:
                    del emotion_counters[key]
                    del predicted_outputs[key]
                    del lseen[key]
                    track_index = track_ids.index(key)
                    del trackers[track_index]
                    del track_ids[track_index]
                if not track_ids:
                    trackers = []
                    track_ids = []
                    emotion_counters = {}
                    predicted_outputs = {}
                    lseen.clear()
                    self.track_id_counter = 0
                    
            # Her takipçi için güncelleme yapma
            try:
                for i, tracker in enumerate(trackers):
                    tracker.update(frame)
                    pos = tracker.get_position()
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    roi_gray = gray[startY:endY, startX:endX]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    roi = extract_features(roi_gray)
                    preds = model.predict(roi)
                    prediction_label = labels[preds.argmax()]

                    # Yüzü dikdörtgenle çevreleme ve duygu etiketini yazma
                    color_map = {
                        'angry': (0, 0, 255),
                        'disgust': (0, 255, 0),
                        'fear': (128, 0, 128),
                        'happy': (0, 255, 255),
                        'neutral': (255, 255, 255),
                        'sad': (255, 0, 0),
                        'surprise': (0, 165, 255)
                    }
                    color = color_map[prediction_label]
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(frame, f"ID: {track_ids[i]} - {prediction_label}", (startX - 10, startY - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color)

                    # İzleyicinin en son görüldüğü zamanı güncelle
                    last_seen[track_ids[i]] = datetime.datetime.now()

                    # Duygu tahminlerini ve zaman bilgilerini güncelle
                    zaman_farki = last_seen[track_ids[i]] - last_s_re[track_ids[i]]
                    if zaman_farki.total_seconds() >= zaman_araligi:
                        emotion_counters[track_ids[i]][prediction_label] += 1
                        predicted_outputs[track_ids[i]].append(prediction_label)
                        last_s_re[track_ids[i]] = last_seen[track_ids[i]]
                    else:
                        predicted_outputs[track_ids[i]].append('')
                    self.tut_dongu = self.tut_dongu + 1

                # Sonuçları gösterme
                cv2.imshow('Video', frame)

            except Exception as e:
                print(f"Error: {e}")
                continue

            # 'q' tuşuna basıldığında döngüyü kır ve uygulamayı sonlandır
            # Çıkış için 'q' tuşuna basın
            if cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('q'):
                player.stop()
                break

            # Videoyu kontrol et
            if player.get_state() == vlc.State.Ended:
                break  # Video sona erdiğinde döngüyü kır ve uygulamayı sonlandır

        # Kaynakları serbest bırakma ve pencereleri kapatma
        webcam.release()
        cv2.destroyAllWindows()

        #print(emotion_counters)
        #print(predicted_outputs)
        #print(self.tut_dongu)
        
        # Grafik boyutunu ve bar rengini ayarlama
        for track_id, emotion_counter in emotion_counters.items():
            total_emotion_count = sum(emotion_counter.values())
            print(f"Takipçi ID: {track_id}, Duygu Sayıları: {emotion_counter}, Toplam Duygu Sayısı: {total_emotion_count}")

        # Her takipçi için duygu sayısını yazdırma
        for track_id, counter in emotion_counters.items():
            print(f"ID: {track_id}, Emotion Counter: {counter}")
        
            
        # Kaynakları serbest bırakma ve pencereleri kapatma

        simdiki_zaman = datetime.datetime.now()
        formatli_zaman = simdiki_zaman.strftime("%Y-%m-%d %H:%M:%S")

        yaz_texti = ""
        with open('yaz.txt', 'a') as dosya:
            for track_id, counter in emotion_counters.items():
                for emotion, count in counter.items():
                    yaz_texti += f"{emotion} {count}/"
                dosya.write(yaz_texti)
                dosya.write(formatli_zaman)
                dosya.write("\n")
                yaz_texti = ""
        dosya.close()

        # Grafik boyutunu ve bar rengini ayarlama
        for track_id, emotion_counter in emotion_counters.items():
            plt.figure(figsize=(8, 6))
            duygular = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            plt.bar(duygular, emotion_counter.values(), color='skyblue')
            # Grafik başlığını ve eksen etiketlerini ayarlama
            plt.title('Takipçi İçin Duygu Tahminlerinin Sütün Grafiği')
            plt.xlabel(f"Takipçi ID: {track_id} için Duygu Durumları")
            plt.ylabel('Her takipçi için saniyede tespit edilen duygu sayısı')
            # Yatay ızgarayı etkinleştirme
            plt.grid(axis='y')
            # Her sınıf için tespit edilen sayıları grafik üzerinde gösterme
            for i, (prediction_label, count) in enumerate(emotion_counter.items()):
                plt.text(i, count + 0.1, str(count), ha='center')
            # Grafiği gösterme
        plt.show()

        # Her takipçi için duygu tahminlerini ve zaman bilgilerini tutmak için sözlükler oluşturun
        algilanan_kare_sayilari = {}
        algilanan_saniyeler = {}
        algilanmayan_kare_sayilari = {}
        algilanmayan_saniyeler = {}

        # Her takipçi için verileri ayrı ayrı işleyin
        for track_id, predictions in predicted_outputs.items():
            zaman_grafigi_tut = 0
            algilanan_kare_sayilari[track_id] = []
            algilanan_saniyeler[track_id] = []
            algilanmayan_kare_sayilari[track_id] = []
            algilanmayan_saniyeler[track_id] = []

        #print(predicted_outputs)

        # Her takipçi için duygu tahminlerini ve zaman bilgilerini tutmak için sözlükler oluşturun
        algilanan_kare_sayilari = {}
        algilanan_saniyeler = {}
        algilanmayan_kare_sayilari = {}
        algilanmayan_saniyeler = {}

        # Her takipçi için verileri ayrı ayrı işleyin
        for track_id, predictions in predicted_outputs.items():
            zaman_grafigi_tut = 0
            algilanan_kare_sayilari[track_id] = []
            algilanan_saniyeler[track_id] = []
            algilanmayan_kare_sayilari[track_id] = []
            algilanmayan_saniyeler[track_id] = []

            for prediction in predictions:
                if prediction is None or prediction == '':
                    algilanmayan_saniyeler[track_id].append(zaman_grafigi_tut)
                    algilanmayan_kare_sayilari[track_id].append('')  # Algılanmayan saniyelerde kare sayısı yoktur, bu yüzden boş eklenir
                else:
                    algilanan_saniyeler[track_id].append(zaman_grafigi_tut)
                    algilanan_kare_sayilari[track_id].append(prediction)  # Algılanan saniyelerdeki kare sayısını ekleyin
                zaman_grafigi_tut += 1

            # Her takipçi için grafik oluşturun
            plt.figure()
            plt.plot(algilanan_saniyeler[track_id], algilanan_kare_sayilari[track_id], color='red', marker='o')
            plt.title(f'Takipçi {track_id} için Saniyede Tespit Edilen Çizgi Grafiği')
            plt.xlabel('Tespit edilen kareler')
            plt.ylabel('Duygu Tahmini')
            plt.grid(True)

            # Grafik dosyasını kaydedin
            plt.savefig(f'takipci_{track_id}_grafik.png')

            # Her takipçi için veri dosyasını yazın
            with open(f'takipci_{track_id}_algilanan_kare.txt', 'a') as dosya:
                dosya.write(f'Takipçi ID: {track_id}\n')
                dosya.write(str(algilanan_saniyeler[track_id]))
                dosya.write("\n")
                dosya.write(str(algilanan_kare_sayilari[track_id]))
                dosya.write("\n")
                formatli_zaman = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dosya.write(str(formatli_zaman))
                dosya.write("\n\n")

        # Plot'u göster
        plt.show()

        algilanan_kare_sayilari = {}
        algilanan_saniyeler = {}

        for track_id, predictions in predicted_outputs.items():
            zaman_grafigi_tut = 0
            algilanan_kare_sayilari[track_id] = []
            algilanan_saniyeler[track_id] = []

            for prediction in predictions:
                if prediction != '':
                    algilanan_saniyeler[track_id].append(zaman_grafigi_tut)
                    algilanan_kare_sayilari[track_id].append(prediction)
                zaman_grafigi_tut += 1

            # Her takipçi için boş olmayan duygu tahminlerini filtreleyin
            filtered_predictions = [x for x in algilanan_kare_sayilari[track_id] if x != '']

            # Her takipçi için grafik oluşturun
            plt.figure()
            plt.plot(filtered_predictions, color='red', marker='o', label='Algılanan Duygular')
            plt.xlabel('Tespit edilen kareler')
            plt.ylabel('Duygu Durumu')
            plt.title(f'Takipçi {track_id} için Tespit Edilen Çizgi Grafiği')
            plt.grid(True)
            plt.legend()

            # Grafik dosyasını kaydedin
            plt.savefig(f'takipci_{track_id}_filtered_grafik.png')

            # Her takipçi için veri dosyasını yazın
            with open(f'takipci_{track_id}_filtered_algilanan_kare.txt', 'a') as dosya:
                dosya.write(f'Takipçi ID: {track_id}\n')
                dosya.write(str(algilanan_saniyeler[track_id]))
                dosya.write("\n")
                dosya.write(str(filtered_predictions))
                dosya.write("\n")
                formatli_zaman = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dosya.write(str(formatli_zaman))
                dosya.write("\n\n")

        # Plot'u göster
        plt.show()

        player.stop()
        webcam.release()
        cv2.destroyAllWindows()
        

    def anket(self):
        self.survey_window = tk.Toplevel(self.master)
        self.survey_app = SurveyApp(self.survey_window)

    def select_video(self):
        video_path = filedialog.askopenfilename(
            title="Video Seç",
            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
        )
        if video_path:
            self.video_path = video_path  # Seçilen video yolunu güncelle
            messagebox.showinfo("Video Seçildi", f"Seçilen Video: {video_path}")
            print(f"Seçilen Video: {video_path}")
        else:
            messagebox.showinfo("Varsayılan Video", f"Varsayılan Video: {self.video_path}")

class SurveyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reklam Filmi Anketi")
        
        # Anket soruları ve seçenekleri
        self.questions = [
            "Reklam filmi sırasında en yoğun hissettiğiniz duygu hangisiydi?",
            "Reklam filmi genel olarak hoşunuza gitti mi?",
            "Reklam filmi sonrasında ürün hakkında düşünceniz olumlu mu?",
            "Reklamın sesi nasıl?",
            "Reklamın görselleri nasıl?",
            "Reklamın mesajı net miydi?",
            "Reklam filmi ilgini çekti mi?"
        ]
        
        self.options = [
            ["Mutlu", "Üzgün", "Kızgın", "Şaşkın", "Nötr", "İğrenme", "Korkmuş"],
            ["Evet", "Hayır"],
            ["Evet", "Hayır"],
            ["İyi", "Orta", "Kötü"],
            ["İyi", "Orta", "Kötü"],
            ["Evet", "Hayır"],
            ["Evet", "Hayır"]
        ]
        
        self.responses = []
        
        # Anket formunu oluşturma
        self.create_form()
        
        # Gönder butonu
        self.submit_button = tk.Button(self.root, text="Gönder", command=self.submit_survey)
        self.submit_button.pack(pady=10)
    
    def create_form(self):
        self.questions_labels = []
        self.options_vars = []
        
        for idx, question in enumerate(self.questions):
            label = tk.Label(self.root, text=question)
            label.pack(anchor='w')
            self.questions_labels.append(label)
            
            var = tk.StringVar()
            self.options_vars.append(var)
            
            for option in self.options[idx]:
                radio = tk.Radiobutton(self.root, text=option, variable=var, value=option)
                radio.pack(anchor='w')
    
    def submit_survey(self):
        responses = [var.get() for var in self.options_vars]
        
        if all(responses):
            self.save_responses(responses)
            messagebox.showinfo("Teşekkürler", "Anketi doldurduğunuz için teşekkürler!")
            self.root.destroy()  # Anket penceresini kapat
        else:
            messagebox.showwarning("Eksik Cevaplar", "Lütfen tüm soruları yanıtlayın.")
    
    def save_responses(self, responses):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("survey_responses.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp] + responses)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.mainloop()
