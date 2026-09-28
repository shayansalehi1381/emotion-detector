# ROADMAP — تشخیص زنده‌ی حالت چهره با وب‌کم

مرجع پیشرفت پروژه. هر آیتم تمام‌شده را `[x]` کن. کار باید فاز به فاز و خودکار پیش برود.

## فاز ۰ — راه‌اندازی
- [x] ساخت `.venv` با `py -3.11 -m venv .venv`
- [x] نصب PyTorch با پشتیبانی CUDA (torch==2.6.0+cu124، سازگار با درایور نصب‌شده)
- [x] نصب بقیه‌ی پکیج‌ها (numpy, pandas, matplotlib, scikit-learn, opencv-python, mediapipe, tqdm, seaborn)
- [x] تست GPU: `src/check_gpu.py` — تأیید شد: RTX 3060 Laptop GPU شناسایی شد
- [x] اسکلت پوشه‌ها: `src/`, `data/`, `checkpoints/`, `results/`, `docs/`
- [x] `.gitignore` (data/*.csv, .venv/, __pycache__/, checkpoints/*.pt, ...)
- [x] `requirements.txt`
- [x] `git init` مستقل برای این پروژه + کامیت اول
- [x] `docs/PHASE_0_REPORT.md`

## فاز ۱ — داده
- [x] بررسی فرمت CSV (ستون‌ها، تعداد ردیف‌ها، توزیع Usage)
- [x] نمودار توزیع کلاس‌ها (تعداد نمونه هر احساس) در `results/class_distribution.png`
- [x] `src/dataset.py`: بازسازی عکس ۴۸×۴۸ از رشته‌ی پیکسل‌ها + کلاس Dataset پای‌تورچ
- [x] DataLoader برای train/val/test طبق ستون Usage (`get_dataloaders`)
- [x] تست سلامت: نمایش چند نمونه از هر کلاس در `results/sample_grid.png`
- [x] `docs/PHASE_1_REPORT.md`

## فاز ۲ — مدل پایه
- [ ] `src/model.py`: یک CNN ساده از صفر
- [ ] تست overfit روی یک batch کوچک (اطمینان از صحت pipeline)
- [ ] `src/train.py` با لاگ‌گیری (loss/accuracy هر epoch)
- [ ] اجرای اول train کامل
- [ ] نمودار loss و accuracy (train/val) در `results/`
- [ ] `docs/PHASE_2_REPORT.md`

## فاز ۳ — بهبود مدل
- [ ] افزودن BatchNorm
- [ ] افزودن Dropout
- [ ] Data augmentation (flip, rotation, ...)
- [ ] LR scheduler
- [ ] وزن‌دهی کلاس‌ها (class weighting) به‌خاطر عدم توازن FER2013
- [ ] Early stopping
- [ ] جدول مقایسه‌ی آزمایش‌ها (baseline vs بهبودها) در `results/`
- [ ] ذخیره‌ی بهترین checkpoint در `checkpoints/`
- [ ] `docs/PHASE_3_REPORT.md`

## فاز ۴ — ارزیابی
- [ ] ارزیابی نهایی روی PrivateTest (test set)
- [ ] Confusion matrix در `results/`
- [ ] Precision/Recall/F1 به تفکیک کلاس
- [ ] تحلیل خطاها با نمونه عکس‌های اشتباه طبقه‌بندی‌شده
- [ ] `docs/PHASE_4_REPORT.md`

## فاز ۵ — برنامه‌ی زنده
- [ ] تشخیص چهره با MediaPipe یا OpenCV (Haar/DNN) روی فریم وب‌کم
- [ ] Preprocessing یکسان با زمان train (resize, grayscale, normalize)
- [ ] هموارسازی پیش‌بینی‌ها بین فریم‌ها (مثلاً میانگین متحرک)
- [ ] نمایش زنده: حالت احساسی + درصد اطمینان + FPS روی تصویر
- [ ] خروج تمیز با کلید `q`
- [ ] `docs/PHASE_5_REPORT.md`

## فاز ۶ — مستندسازی نهایی
- [ ] README.md حرفه‌ای (معرفی پروژه، نصب، اجرا، نتایج، محدودیت‌ها)
- [ ] `docs/INTERVIEW_NOTES.md` (خلاصه‌ی کل مفاهیم برای آمادگی مصاحبه)
- [ ] راهنمای ضبط دمو (چگونه یک ویدیو کوتاه از برنامه‌ی زنده ضبط شود)

## وضعیت فعلی
فاز ۱ کامل شد. در حال شروع فاز ۲ (مدل پایه).
