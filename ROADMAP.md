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
- [x] `src/model.py`: یک CNN ساده از صفر (۶۸۳,۵۲۷ پارامتر)
- [x] تست overfit روی یک batch کوچک (`src/overfit_test.py` — موفق، ۱۰۰٪ روی batch)
- [x] `src/train.py` با لاگ‌گیری (loss/accuracy هر epoch)
- [x] اجرای اول train کامل (۳۰ epoch، بهترین val_acc=56.84% در epoch 11)
- [x] نمودار loss و accuracy (train/val) در `results/baseline_curves.png`
- [x] `docs/PHASE_2_REPORT.md`

## فاز ۳ — بهبود مدل
- [x] افزودن BatchNorm
- [x] افزودن Dropout
- [x] Data augmentation (flip, rotation, translation)
- [x] LR scheduler (ReduceLROnPlateau)
- [x] وزن‌دهی کلاس‌ها (class weighting؛ نسخه‌ی sqrt ملایم‌تر از خام بهتر جواب داد)
- [x] Early stopping
- [x] جدول مقایسه‌ی آزمایش‌ها در `results/experiments_comparison.csv` + نمودار
- [x] ذخیره‌ی بهترین checkpoint: `checkpoints/best.pt` (val_acc=60.55%)
- [x] `docs/PHASE_3_REPORT.md`

## فاز ۴ — ارزیابی
- [x] ارزیابی نهایی روی PrivateTest (test set) — test accuracy=60.60%
- [x] Confusion matrix در `results/confusion_matrix.png`
- [x] Precision/Recall/F1 به تفکیک کلاس در `results/classification_report.txt`
- [x] تحلیل خطاها با نمونه عکس‌های اشتباه طبقه‌بندی‌شده در `results/error_analysis.png`
- [x] `docs/PHASE_4_REPORT.md`

## فاز ۵ — برنامه‌ی زنده
- [x] تشخیص چهره با OpenCV Haar Cascade روی فریم وب‌کم (mediapipe نسخه‌ی نصب‌شده API کلاسیک نداشت)
- [x] Preprocessing یکسان با زمان train (resize, grayscale, normalize)
- [x] هموارسازی پیش‌بینی‌ها بین فریم‌ها (میانگین متحرک روی ۸ فریم اخیر)
- [x] نمایش زنده: حالت احساسی + درصد اطمینان + FPS روی تصویر
- [x] خروج تمیز با کلید `q`
- [x] `docs/PHASE_5_REPORT.md`

## فاز ۶ — مستندسازی نهایی
- [ ] README.md حرفه‌ای (معرفی پروژه، نصب، اجرا، نتایج، محدودیت‌ها)
- [ ] `docs/INTERVIEW_NOTES.md` (خلاصه‌ی کل مفاهیم برای آمادگی مصاحبه)
- [ ] راهنمای ضبط دمو (چگونه یک ویدیو کوتاه از برنامه‌ی زنده ضبط شود)

## وضعیت فعلی
فاز ۵ کامل شد (برنامه‌ی زنده با Haar Cascade + هموارسازی). در حال شروع فاز ۶ (مستندسازی نهایی).
