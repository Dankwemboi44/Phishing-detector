import pandas as pd 
import os 
 
os.makedirs('data/raw', exist_ok=True) 
 
data = [ 
    ['Bofya link hii upate 10000 KES free airtime', 'phishing'], 
    ['Account yako imefreeze thibitisha hapa', 'phishing'], 
    ['Umeshinda 50000 bob Tuma 500 kwa 07xxxxxxx upokee', 'phishing'], 
    ['Tuma password yako turestore account yako', 'phishing'], 
    ['Bonyeza link hii upate free bundle ya 20GB', 'phishing'], 
    ['Lipia 200 bob upate 5000 KES', 'phishing'], 
    ['Account yako itafungwa ndani ya masaa 24 tuma details', 'phishing'], 
    ['Free iPhone 15 Bonyeza hapa upate', 'phishing'], 
    ['Umeteuliwa kushinda 100000 KES Thibitisha sasa', 'phishing'], 
    ['Niaje bro tukutane stage 7 nikuje pesa yako', 'safe'], 
    ['Kazi iko leo Rongai wakenya wa ngori', 'safe'], 
    ['Lipisha nimelipa 500 Check M-PESA', 'safe'], 
    ['Viatu freshi Rongai leo asubuhi', 'safe'], 
    ['Sherehe kesho CBD tumeet', 'safe'], 
    ['Tupatane nikujie mzigo', 'safe'], 
    ['Nk mkuu umelala aje', 'safe'], 
    ['Bruda kesho nasema tumeet', 'safe'], 
    ['Kazi ya mjengo imeanza', 'safe'], 
    ['Watu wa Rongai mko fiti', 'safe'], 
    ['Niko stage tupatane', 'safe'], 
] 
 
df = pd.DataFrame(data, columns=['message', 'label']) 
df.to_csv('data/raw/messages.csv', index=False) 
 
print(f'Created {len(df)} messages') 
print(f'Phishing: {sum(df["label"] == "phishing")}') 
print(f'Safe: {sum(df["label"] == "safe")}') 
