

from flask import Flask, render_template, request, redirect, url_for
import pickle
import re
import numpy as np
from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import word_tokenize


arabic_stopwords = [
    'نيسان', 'هما', 'آه', 'بس', 'أخبر', 'لا', 'مه', 'كل', 'بَسْ', 'إنما', 'ترك', 'لعل', 'إلّا', 'قد', 'ألف', 
    'أنى', 'بك', 'وَيْ', 'لستما', 'خبَّر', 'ت', 'ذينك', 'أولالك', 'حيَّ', 'نا', 'هَجْ', 'ما', 'سرا', 'قلما', 
    'وما', 'اتخذ', 'عسى', 'يورو', 'عجبا', 'اللائي', 'إياها', 'جنيه', 'كلما', 'جيم', 'فاء', 'بهن', 'أولاء', 
    'سين', 'حادي', 'خمسين', 'ض', 'بماذا', 'حزيران', 'شباط', 'مع', 'غير', 'وا', 'كليكما', 'بات', 'ثلاثين', 
    'تِه', 'هَاتانِ', 'اثنان', 'ه', 'ثمة', 'ست', 'فو', 'لم', 'مائة', 'يفعلان', 'التي', 'زود', 'نعم', 'شمال', 
    'أسكن', 'ة', 'سحقا', 'ثان', 'ذين', 'تين', 'من', 'ا', 'رزق', 'أنتم', 'هؤلاء', 'تارة', 'عل', 'فمن', 'ثامن', 
    'كذلك', 'ثمانين', 'أوت', 'رُبَّ', 'إيه', 'لي', 'لبيك', 'ثاء', 'أمسى', 'زعم', 'دولار', 'ثلاثمائة', 'ذلك', 
    'هذان', 'ذات', 'بخ', 'لكنما', 'هنا', 'هناك', 'ح', 'لمّا', 'جميع', 'بعض', 'ستمائة', 'هَيْهات', 'لكي', 'خمسون', 
    'جويلية', 'لات', 'ريال', 'هذه', 'خ', 'لهما', 'عاد', 'لست', 'آمينَ', 'وإن', 'كأيّن', 'كلا', 'أقبل', 'إحدى', 
    'هل', 'اربعون', 'بئس', 'فيما', 'حجا', 'ثلاثون', 'كأنما', 'ثمانمئة', 'اثنا', 'الألى', 'أجمع', 'صبرا', 'كان', 
    'انقلب', 'مادام', 'لسن', 'تِي', 'عشر', 'نحو', 'فلان', 'بؤسا', 'ذهب', 'بطآن', 'أحد', 'عاشر', 'درى', 'الآن', 
    'لعمر', 'سوف', 'ّأيّان', 'ولكن', 'أول', 'أطعم', 'ألا', 'صاد', 'اللتيا', 'ء', 'مكانَك', 'كيت', 'صبر', 'ثلاثة', 
    'جوان', 'ياء', 'عن', 'عند', 'أينما', 'جانفي', 'لما', 'لئن', 'ذَيْنِ', 'فضلا', 'د', 'ءَ', 'كما', 'حيث', 
    'سادس', 'تانِ', 'سبعمئة', 'أصلا', 'ميم', 'مافتئ', 'أفريل', 'أوشك', 'أبدا', 'كيفما', 'إياه', 'إذن', 'ومن', 
    'أمام', 'واو', 'يوان', 'ع', 'تسعمائة', 'صهْ', 'مايو', 'لسنا', 'نوفمبر', 'ظاء', 'شَتَّانَ', 'مازال', 'خمسمئة', 
    'ين', 'إياكن', 'كى', 'أو', 'فإن', 'ن', 'ذانِ', 'ذِه', 'مكانكنّ', 'تلكم', 'أرى', 'ديسمبر', 'شبه', 'كثيرا', 
    'ثمنمئة', 'تعلَّم', 'غدا', 'غين', 'هَاتِه', 'يناير', 'كلاهما', 'نيف', 'جلل', 'ليسا', 'إياهن', 'اللتين', 
    'إليكم', 'دونك', 'كأنّ', 'عشرين', 'أيّ', 'ذلكم', 'أي', 'أربعاء', 'سابع', 'أل', 'إيهٍ', 'حتى', 'سبت', 
    'حبيب', 'خاء', 'هلّا', 'عامة', 'أيضا', 'كسا', 'أى', 'جمعة', 'هاتان', 'ب', 'لوما', 'اللتان', 'أغسطس', 
    'باء', 'إذما', 'وإذ', 'ص', 'عليه', 'تعسا', 'إمّا', 'ريث', 'قطّ', 'لو', 'أنت', 'ليست', 'ما برح', 'حين', 
    'ف', 'ضحوة', 'وراءَك', 'عما', 'كن', 'إلَيْكَ', 'لكنَّ', 'خلافا', 'عدا', 'لهن', 'بل', 'هيا', 'ارتدّ', 'أين', 
    'كرب', 'تسعة', 'نحن', 'تسعمئة', 'فيه', 'لن', 'أُفٍّ', 'إن', 'تفعلين', 'علق', 'هكذا', 'حدَث', 'هَذِه', 
    'هيت', 'كي', 'ك', 'صباح', 'وجد', 'حمٌ', 'كذا', 'أنتما', 'أنتِ', 'ستون', 'ستين', 'تلقاء', 'إياك', 'تموز', 
    'أهلا', 'حسب', 'إذ', 'عشرون', 'طَق', 'كانون', 'لكما', 'علم', 'اللذين', 'ثاني', 'ذواتا', 'أمد', 'رابع', 
    'س', 'لدن', 'شتان', 'عليك', 'كأين', 'أيلول', 'سبعمائة', 'فرادى', 'بغتة', 'قام', 'ؤ', 'أنًّ', 'بين', 'إنا', 
    'هاته', 'م', 'ضاد', 'تسعين', 'حاي', 'وهو', 'عَدَسْ', 'إليكنّ', 'طاق', 'مذ', 'بكم', 'همزة', 'ثم', 'بعدا', 
    'إنه', 'والذين', 'فبراير', 'سبعون', 'أيار', 'هنالك', 'آهٍ', 'منذ', 'آها', 'أبٌ', 'راح', 'أولئك', 'بلى', 
    'تبدّل', 'تسع', 'سبتمبر', 'لا سيما', 'ليرة', 'كلَّا', 'سبعة', 'ذيت', 'حرى', 'له', 'ثمانية', 'سبحان', 'مئة', 
    'اثني', 'هاكَ', 'كاد', 'أمامك', 'استحال', 'أعطى', 'هاء', 'خال', 'جير', 'أبريل', 'ذا', 'شيكل', 'قبل', 
    'كِخ', 'الذين', 'بمن', 'غ', 'تفعلون', 'ثالث', 'كم', 'مما', 'أربعمائة', 'ئ', 'تانِك', 'وإذا', 'ش', 'تلكما', 
    'آذار', 'لكيلا', 'هيّا', 'كيف', 'غالبا', 'لكم', 'إلى', 'خميس', 'هَذِي', 'ته', 'أما', 'في', 'كأي', 'إليكَ', 
    'هللة', 'خاصة', 'أخذ', 'ثلاثمئة', 'ذِي', 'خلا', 'إذا', 'خلف', 'صار', 'ما أفعله', 'يونيو', 'ولو', 'شين', 
    'ذي', 'آنفا', 'بنا', 'ثماني', 'لستم', 'تاء', 'بيد', 'إليك', 'ذلكما', 'كلتا', 'هاك', 'آ', 'مكانكما', 
    'آناء', 'أوّهْ', 'ظ', 'ماي', 'أنشأ', 'سمعا', 'اللاتي', 'نبَّا', 'لستن', 'أكثر', 'أن', 'بهما', 'أفٍّ', 
    'تجاه', 'اللذان', 'كاف', 'هَذَيْنِ', 'سنتيم', 'بما', 'ط', 'هبّ', 'آض', 'لها', 'أقل', 'ولا', 'لاسيما', 
    'لعلَّ', 'حمدا', 'عيانا', 'صهٍ', 'مارس', 'نون', 'قاف', 'مئتان', 'خمس', 'أخٌ', 'هَذانِ', 'فلا', 'وهب', 
    'مرّة', 'ى', 'فيم', 'ليت', 'خمسة', 'نَخْ', 'خامس', 'ستة', 'ذواتي', 'ثمَّ', 'أصبح', 'منه', 'الذي', 'إنَّ', 
    'ذانك', 'حَذارِ', 'أ', 'سبع', 'هَاتِي', 'هو', 'لولا', 'الألاء', 'ليستا', 'أربع', 'لنا', 'هذي', 'رجع', 
    'درهم', 'على', 'إما', 'شتانَ', 'تحوّل', 'حاء', 'أجل', 'آهاً', 'ج', 'كلّما', 'ممن', 'اربعين', 'تينك', 
    'إليكما', 'م', 'إذاً',"اذا", 'سرعان', 'سقى', 'تخذ', 'أبو', 'أمامكَ', 'هي', 'إيانا', 'هَؤلاء', 'بسّ', 'ذال', 
    'يفعلون', 'عدَّ', 'آهِ', 'ما انفك', 'عين', 'و', 'قاطبة', 'أنّى', 'أربعة', 'راء', 'دون', 'هاتي', 'ها', 
    'منها', 'ثمّ', 'أنتن', 'واهاً', 'بها', 'سوى', 'ر', 'ثلاثاء', 'طالما', 'ابتدأ', 'يوليو', 'مليم', 'رويدك', 
    'أيها', 'هلم', 'إياهم', 'أمّا', 'هاهنا', 'ذ', 'هيهات', 'هَاتَيْنِ', 'غداة', 'اللواتي', 'لدى', 'ق', 
    'ساء', 'ثمانون', 'ألفى', 'دينار', 'بكن', 'بَلْهَ', 'أعلم', 'تفعلان', 'أخو', 'صراحة', 'بكما', 'أنا', 
    'إياكما', 'تَيْنِ', 'هلا', 'أنبأ', 'واحد', 'دال', 'كأن', 'هاتين', 'تسعون', 'مساء', 'مهما', 'زاي', 'ليسوا', 
    'إياهما', 'يمين', 'اثنين', 'عوض', 'ظنَّ', 'حيثما', 'ذاك', 'أيا', 'علًّ', 'رأى', 'لام', 'طفق', 'بهم', 
    'ليس', 'كليهما', 'ستمئة', 'أمس', 'ظلّ', 'كأيّ', 'حمو', 'آي', 'أم', 'تاسع', 'صدقا', 'آب', 'انبرى', 
    'هذين', 'فيها', 'أيّان', 'ذه', 'متى', 'والذي', 'تي', 'هن', 'عشرة', 'طرا', 'حاشا', 'إياي', 'فلس', 'ورد', 
    'فيفري', 'أكتوبر', 'حار', 'أربعمئة', 'سبعين', 'مكانكم', 'مثل', 'قرش', 'تحت', 'به', 'لكن', 'غادر', 'ي', 
    'بعد', 'لهم', 'إياكم', 'إليكن', 'تلك', 'ز', 'ل', 'إى', 'نَّ', 'أف', 'طاء', 'هم', 'هَذا', 'ثلاث', 'ذلكن', 
    'إزاء', 'ذو', 'حبذا', 'ثمان', 'نفس', 'ثمّة', 'معاذ', 'حقا', 'لك', 'تشرين', 'دواليك', 'اخلولق', 'ذوا', 
    'بضع', 'فوق', 'فإذا', 'شرع', 'ث', 'إي', 'ذان', 'أوه', 'إلا', 'بي', 'أفعل به', 'يا', 'خمسمائة', 'وُشْكَانَ', 
    'جعل', 'بخٍ', 'أضحى', 'هذا'
]


stemmer = ISRIStemmer()


def remove_stopwords_arabic(text):
    words = text.split()
    filtered_words = [word for word in words if word not in arabic_stopwords]
    return ' '.join(filtered_words)


def normalize_arabic_text(text):
    text = text.replace("صلى الله عليه وسلم", "صلى_الله_عليه_وسلم")
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    text = re.sub(r'[0-9٠-٩]+', '', text)
    text = re.sub(r'[إأآءؤئ]', 'ا', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def apply_stemming(text):
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)


def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens


def extract_handcrafted_features(text):
    tokens = word_tokenize(text)
    economy_words = ["مال", "اقتصاد", "سوق", "تجارة"]
    num_economy = sum(1 for word in tokens if word in economy_words)
    num_unique_chars = len(set(text))
    return [num_economy, num_unique_chars]


with open('svm_model.pkl', 'rb') as file:
    svm_model, tfidf_vectorizer, scaler, selector = pickle.load(file)


app = Flask(__name__)


def predict_text_class(text):
    cleaned_text = remove_stopwords_arabic(text)
    normalized_text = normalize_arabic_text(cleaned_text)
    stemmed_text = apply_stemming(normalized_text)
    tokenized_text = ' '.join(tokenize_text(stemmed_text))
    
    text_tfidf = tfidf_vectorizer.transform([tokenized_text]).toarray()
    handcrafted_features = np.array(extract_handcrafted_features(cleaned_text)).reshape(1, -1)
    combined_features = np.hstack((text_tfidf, handcrafted_features))
    combined_features_selected = selector.transform(combined_features)
    combined_features_scaled = scaler.transform(combined_features_selected)
    
    predicted_class = svm_model.predict(combined_features_scaled)[0].strip()  # إزالة المسافات الزائدة
    
    
    print(f"Predicted class (before translation): '{predicted_class}'")

    if predicted_class == "Economy" or predicted_class == "articlesEconomy":
        predicted_class_arabic = "اقتصاد"
    elif predicted_class == "Sports" or predicted_class == "articlesSports":
        predicted_class_arabic = "رياضة"
    elif predicted_class == "Culture" or predicted_class == "articlesCulture":
        predicted_class_arabic = "ثقافة"
    elif predicted_class == "Religion" or predicted_class == "articlesReligion":
        predicted_class_arabic = "دين"
    elif predicted_class == "Local" or predicted_class == "articlesLocal":
        predicted_class_arabic = "محلي"
    elif predicted_class == "International" or predicted_class == "articlesInternational":
        predicted_class_arabic = "دولي"
    else:
        predicted_class_arabic = predicted_class
    
    print(f"Translated class to Arabic : '{predicted_class_arabic}'")

    return predicted_class_arabic


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    text = file.read().decode('utf-8')
    predicted_class = predict_text_class(text)
    
    return render_template('index.html', result=predicted_class)


if __name__ == '__main__':
    app.run(debug=True)
