# تشخیص زنده‌ی حالت چهره با وب‌کم (Facial Emotion Recognition)

یک CNN آموزش‌دیده **از صفر** (بدون هیچ مدل pretrained) روی دیتاست FER2013 که حالت
احساسی چهره را از فریم وب‌کم به‌صورت زنده تشخیص می‌دهد و درصد اطمینان را نمایش می‌دهد.

این پروژه با هدف یادگیری کامل فرآیند یک پروژه‌ی یادگیری عمیق ساخته شده: از خواندن
داده‌ی خام تا آموزش، دیباگ، ارزیابی صادقانه، و رساندن مدل به یک برنامه‌ی واقعی.

## نتایج نهایی

| معیار | مقدار |
|---|---|
| بهترین val accuracy (PublicTest) | **60.55%** |
| Test accuracy نهایی (PrivateTest، فقط یک‌بار ارزیابی شد) | **60.60%** |
| بهترین کلاس (F1) | Happy — 0.838 |
| ضعیف‌ترین کلاس (F1) | Fear — 0.273 (شباهت بصری زیاد با Sad/Surprise) |

جزئیات کامل، از جمله یک آزمایش که در ابتدا نتیجه‌ی بدتری داد و نحوه‌ی تشخیص و رفع آن،
در [docs/PHASE_3_REPORT.md](docs/PHASE_3_REPORT.md) و [docs/PHASE_4_REPORT.md](docs/PHASE_4_REPORT.md) آمده است.

> این عدد را عمداً بدون اغراق گزارش می‌کنیم. سقف واقع‌بینانه‌ی دقت روی FER2013 با یک
> CNN ساده حدود ۶۵-۷۰٪ است؛ رسیدن به اعداد بالاتر معمولاً نیازمند معماری‌های عمیق‌تر
> یا ensemble چند مدل است.

## معماری و دیتاست

- **دیتاست**: [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) — ۳۵,۸۸۷
  تصویر ۴۸×۴۸ خاکستری در ۷ کلاس احساسی (Angry, Disgust, Fear, Happy, Sad, Surprise,
  Neutral)، با تقسیم رسمی Training/PublicTest/PrivateTest.
- **مدل**: `EmotionCNNImproved` — سه بلوک Conv+BatchNorm+ReLU+MaxPool، به‌علاوه‌ی
  Dropout قبل از لایه‌های fully-connected. حدود ۶۸۴K پارامتر.
- **تکنیک‌های بهبود**: data augmentation (flip/rotation/translation)، class weighting
  با وزن ملایم‌شده (sqrt) برای عدم توازن کلاس‌ها، LR scheduler، و early stopping.
- **تشخیص چهره‌ی زنده**: OpenCV Haar Cascade (نه بخشی از مدل یادگیری‌شده — فقط برای
  پیدا کردن محل چهره در فریم وب‌کم قبل از دادن آن به مدل).

## ساختار پروژه

```
src/
  dataset.py          کلاس Dataset، DataLoader، augmentation، class weights
  model.py             معماری‌های EmotionCNN (baseline) و EmotionCNNImproved
  train.py             حلقه‌ی آموزش عمومی + لاگ + checkpoint
  experiments.py        آزمایش‌های فاز ۳ (BatchNorm+Dropout، augmentation، ...)
  final_v2.py            آزمایش نهایی و برنده‌ی فاز ۳
  evaluate.py            ارزیابی نهایی روی PrivateTest
  live_demo.py            برنامه‌ی زنده‌ی وب‌کم
  explore_data.py, check_gpu.py, check_samples.py, overfit_test.py,
  plot_training.py, compare_experiments.py     ابزارهای کمکی/دیباگ
data/fer2013.csv       دیتاست خام (در git نیست، باید خودتان دانلود کنید)
checkpoints/            وزن‌های ذخیره‌شده‌ی مدل (در git نیست)
results/                نمودارها، جدول‌ها، گزارش‌های ارزیابی
models/                 فایل Haar Cascade برای تشخیص چهره
docs/                   گزارش هر فاز + یادداشت‌های مصاحبه
```

## نصب

```bash
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

دیتاست `fer2013.csv` را دانلود کرده و در `data/fer2013.csv` قرار دهید.

> نکته: کنسول ویندوز پیش‌فرض از UTF-8 پشتیبانی نمی‌کند. همه‌ی اسکریپت‌ها را با
> `PYTHONIOENCODING=utf-8` اجرا کنید تا خروجی فارسی درست چاپ شود.

## اجرا

```bash
# بررسی GPU
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/check_gpu.py

# آموزش مدل بهبودیافته با بهترین تنظیمات (نتیجه‌ی فاز ۳)
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/final_v2.py

# ارزیابی نهایی روی PrivateTest
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/evaluate.py

# برنامه‌ی زنده‌ی وب‌کم (خروج با کلید q)
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/live_demo.py
```

## محدودیت‌های شناخته‌شده

- دقت روی کلاس Fear ضعیف است (شباهت بصری زیاد با Sad/Surprise/Neutral در تصاویر
  کوچک سیاه‌سفید — جزئیات در [docs/PHASE_4_REPORT.md](docs/PHASE_4_REPORT.md)).
- خود دیتاست FER2013 دارای مقداری نویز برچسب شناخته‌شده است (چند نمونه در تحلیل خطا
  دیده شد که برچسب‌شان با محتوای واقعی تصویر همخوانی نداشت).
- تشخیص چهره با Haar Cascade در نورپردازی ضعیف یا زوایای غیرجلویی صورت ضعیف‌تر عمل
  می‌کند نسبت به روش‌های مبتنی بر یادگیری عمیق.
- مدل روی یک چهره در هر لحظه کار می‌کند (بزرگ‌ترین چهره‌ی تشخیص‌داده‌شده).

## مستندات کامل هر فاز

- [docs/PHASE_0_REPORT.md](docs/PHASE_0_REPORT.md) — راه‌اندازی محیط
- [docs/PHASE_1_REPORT.md](docs/PHASE_1_REPORT.md) — داده
- [docs/PHASE_2_REPORT.md](docs/PHASE_2_REPORT.md) — مدل پایه
- [docs/PHASE_3_REPORT.md](docs/PHASE_3_REPORT.md) — بهبود مدل (شامل یک نتیجه‌ی
  غیرمنتظره و نحوه‌ی تشخیص/رفع آن)
- [docs/PHASE_4_REPORT.md](docs/PHASE_4_REPORT.md) — ارزیابی نهایی
- [docs/PHASE_5_REPORT.md](docs/PHASE_5_REPORT.md) — برنامه‌ی زنده
- [docs/INTERVIEW_NOTES.md](docs/INTERVIEW_NOTES.md) — خلاصه‌ی آماده‌سازی مصاحبه
- [docs/DEMO_RECORDING_GUIDE.md](docs/DEMO_RECORDING_GUIDE.md) — راهنمای ضبط دمو
- [ROADMAP.md](ROADMAP.md) — وضعیت پیشرفت پروژه فاز به فاز
