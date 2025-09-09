import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pythainlp import word_tokenize
from pythainlp.util import normalize
import emoji
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000
SELECT_K_BEST = 2000

def load_data(filepath):
    try:
        df = pd.read_excel(filepath)
        if not all(col in df.columns for col in ['paragraph', 'label']):
            raise ValueError("Excel file must contain 'paragraph' and 'label' columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = normalize(text)
    text = emoji.replace_emoji(text, replace='')
    pattern = r'[^ก-๙a-zA-Z0-9%\$฿.,+\- ]'
    text = re.sub(pattern, ' ', text)
    text = ' '.join(text.split())
    return text.strip()

def tokenize_thai_text(text):
    money_pattern = r'\d+[.,]?\d*[%\$฿]'
    money_matches = re.findall(money_pattern, text)    
    tokens = word_tokenize(text, engine="newmm")
    final_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        matched = False
        for pattern in money_matches:
            if token in pattern and i + 1 < len(tokens):
                combined = token + tokens[i+1]
                if combined in pattern:
                    final_tokens.append(combined)
                    i += 1
                    matched = True
                    break
        if not matched:
            final_tokens.append(token)
        i += 1
    return final_tokens

def preprocess_data(df):
    df['cleaned_text'] = df['paragraph'].apply(clean_text)
    df = df[df['cleaned_text'].str.len() > 0]
    print("\nClass Distribution:")
    print(df['label'].value_counts())
    return df

def fit_feature_pipeline(X_train, y_train):
    tfidf = TfidfVectorizer(
        tokenizer=tokenize_thai_text, max_features=MAX_FEATURES, ngram_range=(1, 2), lowercase=False
    )
    ch2 = SelectKBest(chi2, k=min(SELECT_K_BEST, X_train.shape[0]))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_train_selected = ch2.fit_transform(X_train_tfidf, y_train)
    return X_train_selected, tfidf, ch2

def transform_features(X_data, tfidf, ch2):
    X_tfidf = tfidf.transform(X_data)
    X_selected = ch2.transform(X_tfidf)
    return X_selected

def train_model(X_train, y_train, model_type='lr'):
    if model_type == 'lr':
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced', multi_class='multinomial')
    elif model_type == 'nb':
        model = MultinomialNB()
    elif model_type == 'svm':
        model = SVC(kernel='linear', random_state=RANDOM_STATE, class_weight='balanced', probability=True)
    else:
        raise ValueError("Invalid model type. Choose 'lr', 'nb', or 'svm'")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['neg', 'pos', 'neu']))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['neg', 'pos', 'neu'], yticklabels=['neg', 'pos', 'neu'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    # return dictionary
    return {'accuracy': accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred, average='weighted'), 'recall': recall_score(y_test, y_pred, average='weighted'), 'f1': f1_score(y_test, y_pred, average='weighted'), 'confusion_matrix': cm}

def save_model(model, tfidf, feature_selector, filename):
    model_data = {'model': model, 'tfidf': tfidf, 'feature_selector': feature_selector}
    joblib.dump(model_data, filename)
    print(f"\nModel saved as {filename}")

def load_saved_model(filename):
    try:
        return joblib.load(filename)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_new_text(model_data, text):
    if not model_data:
        return "Model not loaded correctly"
    cleaned_text = clean_text(text)
    selected_text = transform_features([cleaned_text], model_data['tfidf'], model_data['feature_selector'])
    prediction = model_data['model'].predict(selected_text)
    probabilities = model_data['model'].predict_proba(selected_text)
    return {'prediction': prediction[0], 'probabilities': {'neg': probabilities[0][0], 'pos': probabilities[0][1], 'neu': probabilities[0][2]}}

def main():
    data_file = "D:\project\sentiment_tf-idf\dataset\single_mix_3label_6type.xlsx"
    df = load_data(data_file)
    if df is None: return
    df = preprocess_data(df)
    
    X_raw = df['cleaned_text']
    y = df['label']
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    print(f"\nData split into {len(X_train_raw)} training and {len(X_test_raw)} testing samples.")
    X_train, tfidf, ch2 = fit_feature_pipeline(X_train_raw, y_train)
    X_test = transform_features(X_test_raw, tfidf, ch2)
    
    print(f"Final training features shape: {X_train.shape}")
    print(f"Final testing features shape: {X_test.shape}")
    
    model_choice = input("\nSelect model to train (lr/nb/svm): ").strip().lower()
    
    while model_choice not in ['lr', 'nb', 'svm']:
        model_choice = input("Select model to train (lr/nb/svm): ").strip().lower()
    
    print(f"\nTraining '{model_choice}' model")
    model = train_model(X_train, y_train, model_choice)
    
    test_metrics = evaluate_model(model, X_test, y_test)
    print("\nEvaluation Metrics Summary:")
    print(test_metrics)
    
    model_save_file = "thai_sentiment_model.joblib"
    save_model(model, tfidf, ch2, model_save_file)

    # --- testing with unknown label dataset
    print("\n--- Prediction Demonstration ---")
    loaded_model_data = load_saved_model(model_save_file)
    
    if loaded_model_data:
        example_texts = [
            "SJWD แย้มไตรมาส 4 สดใส มั่นใจดันรายได้ปีนี้แตะ 2.5 หมื่นล้าน",
            "เช้าวันนี้SETESG ปิดที่ระดับ 755.66 จุด เพิ่มขึ้น 2.77 จุด หรือ 0.37 %",
            "VIH ทุ่ม 200 ล้านบาท ประกาศซื้อหุ้นคืน พร้อมจ่ายเงินปันผลงวดผลงานปี 67 ในอัตรา 0.25 บาท หลังโชว์กำไรสุทธิ 315.55 ล้านบาท เพิ่มขึ้น 10.77%",
            "PTG ปี 67 เติบโตแข็งแกร่ง มีกำไรสุทธิ 1,042 ล้านบาท เพิ่มขึ้น 7.9% YoY หลังธุรกิจ Oil ขยายตัว 12.3% -Non-Oil โตอย่างมีนัยสำคัญ พร้อมปันผลอัตรา 0.25 บาท",
            "WPH โชว์รายได้ปี67 ทะลุ 2,000 ลบ.โตขึ้น 39% กำไรแตะ 283 ลบ.เพิ่มขึ้น 198% บอร์ดมีมติอนุมัติปันผลในอัตรา 25 หุ้นเดิมต่อ 1 หุ้นปันผล ลุยขยายบริการ ดันผลงานปีนี้ออลไทม์ไฮ",
            "สรุป SET100 วันนี้ DELTA บวกสูงสุด 5%",
            "MINT เด้ง 1% รับข่าวเตรียมตั้ง “กองรีท” มูลค่า 1.2 พันล้านดอลลาร์ มั่นใจรายได้ปีนี้โต 10%",
            "MAGURO วิ่ง 4% ทำ “ออลไทม์ไฮ” มั่นใจรายได้ปีนี้โต 30% ลุยขยายสาขาเต็มสูบ",
            "DELTA ดีดบวก 3% หลังหลุด “แคชบาลานซ์” วันแรก",
            "“ทองคำ” เช้านี้ขึ้น 200 บ. “รูปพรรณ” ขายออก 42,800 บาท",
            "SET บ่ายรูด 20 จุด กังวลท่องเที่ยวชะงัก “ญี่ปุ่น” เตือนพลเมืองเสี่ยงเกิด “วินาศกรรม” ไทย",
            "“เงินบาท” เปิดเช้าอ่อนค่า 34.75 บ. หลังตลาดกังวล",
            "วันนี้SETHD ปิดที่ระดับ 1,085.29 จุด ลดลง 6.51 จุด หรือ 0.60 %",
            "KEX ปี67 แบกขาดทุน 5,911.3 ล้านบาท หลังปริมาณการส่งพัสดุจากกลุ่ม SKA วูบ- ปริมาณการจัดส่งลดลง 23%",
            "SET ปิดเช้าร่วง 10 จุด หลุด 1,200 เซ่นบาทอ่อน-งบ บจ.ต่ำคาด",
            "EA ร่วง 1% วิตกขายธุรกิจ “โซลาร์ฟาร์ม” พิษณุโลก 8 พันล้าน ส่อเพิ่มทุนต่ำ-เก็งหลุด SET50",
            "SET ปิดลบ 2 จุด นลท.เทขายลดเสี่ยง กังวลเทรดวอร์ จับตา ครม.สัญจร พรุ่งนี้",
            "S&P จัดอันดับเครดิต Boeing แนวโน้มลบ หลังพนักงาน 3.3 หมื่นคน หยุดงานประท้วงยืดเยื้อ",
            "WTI ดิ่ง 3% หลุด 73 เหรียญ กังวลเศรษฐกิจจีนซบเซา หวั่นกระทบดีมานด์น้ำมัน",
            "แนวโน้ม SET ซึมตัว เจอปัจจัยกดดัน บาทอ่อน-งบไตรมาส 3 ต่ำคาด",
            "SKR คว้า SET ESG Ratings 2024 ระดับ “BBB”",
            "WPH ลุยเปิด รพ.วัฒนแพทย์ ภูเก็ต Q4/69 ปักธงรองรับต่างชาติ",
            "“ดัชนีฮั่งเส็ง” ปิดทะลุ 20,000 จุด รับ “จีน” กระตุ้นเศรษฐกิจเชิงรุกปี 68",
            "MK ปรับโครงสร้างผู้ถือหุ้น หลัง “รวิพรรณ จารุทวี” เข้าถือ 20.92% ขึ้นเบอร์ 2",
            "SET สัปดาห์นี้ “ฟื้นตัว” จับตา “เงินเฟ้อ” ไทย ต.ค.- เลือกตั้ง ปธน.สหรัฐฯ",
            "แนวโน้ม SET รีบาวด์กรอบ 1,445-1,465 จุด",
            "คัด 12 หุ้น รับประโยชน์ “ต่างชาติ” ลงทุนไทย 9 เดือน ทะลุ 7.2 แสนล้าน",
            "KBANK โบรกอัพเป้า 165 บาท รับนโยบายเงินปันผลใหม่ 45%",
            "SCM กางแผนธุรกิจปี 68 ชู 5 กลยุทธ์ มุ่งสู่องค์กรชั้นนำ-การเติบโตยั่งยืน",
            "AGE เร่งปั้น ABM สู่เรือธง เดินเกมธุรกิจกรีน สร้างรายได้โตยั่งยืน",
            "การลงทุนหุ้นไทยที่ ซบเซาอย่างหนัก แม้จะมี ข่าวดีจากภาคการท่องเที่ยว แต่ก็ยังไม่สามารถพลิกฟื้น ความเชื่อมั่นนักลงทุนได้",
            "ความผันผวนของตลาดหุ้นไทยที่ น่ากังวล ถึงแม้จะ มีโอกาสทำกำไร ในหุ้นบางกลุ่ม แต่ก็ไม่คุ้มค่า กับความเสี่ยงที่ต้องเจอ",
            "มูลค่าการซื้อของนักลงทุนต่างชาติที่ ลดลงอย่างน่าใจหาย ทำให้ การลงทุนหุ้นไทย ไม่น่าสนใจ เลย แม้จะมี มาตรการกระตุ้นเศรษฐกิจ ก็ไม่เพียงพอที่จะเห็นผล",
            "ดัชนีราคาหลักทรัพย์ในไทย ปรับตัวลดลงต่อเนื่อง แม้จะมี กำไรของบริษัทจดทะเบียน ที่ยังดี แต่ก็ยังไม่ช่วย พยุงตลาดให้ดีขึ้น",
            "การลงทุนหุ้นต่างประเทศมี ความเสี่ยงจากอัตราแลกเปลี่ยนสูง ถึงแม้จะ ให้ผลตอบแทนดี ในตลาดเทคโนโลยี แต่ก็ไม่สามารถชดเชย ความผันผวนของค่าเงินบาทได้",
            "ตลาดหุ้นต่างประเทศที่ เผชิญกับภาวะเศรษฐกิจถดถอย ทำให้ การลงทุนหุ้นต่างประเทศ ดูไม่สดใส แม้จะมี บริษัทเติบโตสูง ก็ไม่ช่วย ให้พอร์ตโดยรวมเป็นบวก",
            "การลงทุนในพันธบัตรรัฐบาลให้ ผลตอบแทนต่ำติดดิน แม้จะ มีความมั่นคงสูง แต่ก็ไม่คุ้มค่า เมื่อเทียบกับอัตราเงินเฟ้อที่พุ่งขึ้น",
            "อัตราดอกเบี้ยประจำวันของธนาคารพาณิชย์ที่ ต่ำเตี้ยเรี่ยดิน ทำให้ การลงทุนในพันธบัตรรัฐบาล ไม่จูงใจ เลย ถึงแม้จะ ปลอดภัยไร้ความเสี่ยง ก็ไม่สามารถสร้างความมั่งคั่ง ได้จริง",
            "การลงทุนในตลาดเงินให้ ผลตอบแทนเพียงน้อยนิด ถึงแม้จะ มีสภาพคล่องสูง แต่ก็ไม่เพียงพอ ที่จะสู้กับค่าครองชีพที่เพิ่มขึ้น",
            "ราคาน้ำมันดิบในตลาดโลก พุ่งสูงขึ้นอย่างต่อเนื่อง ทำให้ ต้นทุนการผลิตของธุรกิจ เพิ่มขึ้นอย่างมาก แม้จะมี การค้นพบแหล่งน้ำมันใหม่ ก็ไม่สามารถบรรเทา ผลกระทบทั้งหมดได้",
            "การลงทุนหุ้นไทย เมื่อเปรียบเทียบกับความผันผวนแล้ว ข้อเสียเรื่องสภาพคล่องที่จำกัดถือว่าเล็กน้อย",
            "การลงทุนหุ้นต่างประเทศ ถ้าเทียบกับคู่แข่ง ปัญหาเรื่องความซับซ้อนของข้อมูลถือว่าไม่ใช่เรื่องใหญ่",
            "การลงทุนในพันธบัตรรัฐบาล เมื่อดูจากมุมมองโดยรวม ข้อจำกัดเรื่องผลตอบแทนที่ต่ำไม่ได้สำคัญมาก",
            "การลงทุนในตลาดเงิน เมื่อชั่งน้ำหนักดูแล้ว ข้อเสียเรื่องอัตราดอกเบี้ยต่ำไม่ได้มีน้ำหนักมาก",
            "การลงทุนในทองคำ ถ้าเทียบกับประโยชน์ที่ได้รับ ปัญหาเรื่องราคาผันผวนถือว่าเล็กน้อย",
            "การลงทุนในอสังหาริมทรัพย์ เมื่อพิจารณาจากภาพรวม ข้อบกพร่องเรื่องสภาพคล่องต่ำไม่ได้สำคัญมาก",
            "การลงทุนในกองทุนรวม ถ้าเปรียบเทียบกับทางเลือกอื่น ข้อเสียเรื่องค่าธรรมเนียมยังยอมรับได้",
            "การลงทุนตราสารหนี้ เมื่อดูจากมุมมองของผู้ใช้งาน ปัญหาเรื่องผลตอบแทนที่จำกัดไม่ได้เป็นอุปสรรคมาก",
            "การลงทุนตราสารอนุพันธ์ ถ้าเทียบกับมาตรฐานทั่วไป ข้อจำกัดเรื่องความเสี่ยงสูงถือว่าปกติ",
            "ราคาน้ำมันดิบในตลาดโลก เมื่อชั่งน้ำหนักข้อดีข้อเสียแล้ว ปัญหาเรื่องความผันผวนของราคาไม่ได้มีน้ำหนักมาก",
            "ราคาน้ำมันดิบฟื้นตัวหลังจากการปรับตัวลง",
            "น้ำมันเติบโตท่ามกลางความผันผวน",
            "ราคาน้ำมันขายปลีกแข็งแกร่งจากการลดลง",
            "อัตราแลกเปลี่ยนบาท-ดอลลาร์ปรับตัวดีจากความกังวล",
            "ราคาน้ำมันดิบเด่นหลังจากการขายออก",
            "อัตราแลกเปลี่ยนพุ่งแรงหลังจากการแก้ไข",
            "ราคาน้ำมันขายปลีกเติบโตท่ามกลางเงินเฟ้อ",
            "อัตราแลกเปลี่ยนบาท-ดอลลาร์แข็งแกร่งจากการขายทำกำไร",
            "ราคาน้ำมันดิบฟื้นตัวหลังจากความไม่แน่นอน",
            "อัตราแลกเปลี่ยนปรับตัวดีจากการชะลอตัว",
            "หุ้นไทยขาดทุนเพิ่มขึ้น",
            "หุ้นไทยตกต่ำฟื้นตัว",
            "หุ้นไทยวิกฤตกลับรุ่ง",
            "หุ้นไทยล้มเหลวสู่ความสำเร็จ",
            "หุ้นไทยสูญเสียได้คืน",
            "หุ้นไทยแย่ขึ้น",
            "หุ้นไทยพลิกขาดทุนเป็นกำไร",
            "หุ้นไทยวิกฤตเป็นโอกาส",
            "หุ้นไทยปัญหาแก้ไขได้",
            "หุ้นไทยอุปสรรคกลายเป็นบทเรียน",
        ]

        results = []
        
        for text in example_texts:
            result = predict_new_text(loaded_model_data, text)
            print(f"\nText: '{text}'")
            print(f"  -> Prediction: {result['prediction']}")

            row = {
                'text': text,
                'prediction': result['prediction']
            }

            for class_name, probability in result['probabilities'].items():
                row[f'prob_{class_name}'] = probability
        
            results.append(row)
        df = pd.DataFrame(results)
    
        csv_name = f"single_mix_3label_6type_{model_choice}.csv"
        df.to_csv(csv_name, index=False, encoding="utf-8-sig")
        
        print(f"บันทึกผลลัพธ์ {len(results)} รายการลงใน prediction_results.csv เรียบร้อยแล้ว")
        
        # แสดงตัวอย่างผลลัพธ์
        print("\nตัวอย่างผลลัพธ์ 5 รายการแรก:")
        print(df.head())
        
        # แสดงสถิติการทำนาย
        print("\nสถิติการทำนาย:")
        print(df['prediction'].value_counts())
if __name__ == "__main__":
    main()