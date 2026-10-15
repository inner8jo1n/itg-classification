| Method | CV macro F1 | CV std | Test macro F1 | Test accuracy | Negative recall | CV change | Test change |
|---|---|---|---|---|---|---|---|
| LinearSVC balanced, words + chars, tuned | 0.762 | 0.025 | 0.748 | 0.827 | 0.522 | +22.6% | +11.3% |
| LogReg balanced, words + chars | 0.759 | 0.019 | 0.748 | 0.816 | 0.638 | +22.2% | +11.3% |
| LogReg balanced, words + chars, tuned | 0.759 | 0.019 | 0.748 | 0.816 | 0.638 | +22.2% | +11.3% |
| LinearSVC balanced | 0.737 | 0.018 | 0.754 | 0.816 | 0.551 | +18.7% | +12.1% |
| LogReg balanced, char 2-5 grams | 0.734 | 0.014 | 0.742 | 0.806 | 0.696 | +18.1% | +10.5% |
| LogReg balanced, unigrams | 0.723 | 0.012 | 0.713 | 0.780 | 0.667 | +16.5% | +6.1% |
| LogReg, balanced class weights | 0.723 | 0.017 | 0.722 | 0.791 | 0.638 | +16.3% | +7.5% |
| LogReg balanced, negation tokens | 0.722 | 0.016 | 0.722 | 0.792 | 0.638 | +16.3% | +7.5% |
| LogReg balanced, 1-3 grams | 0.721 | 0.015 | 0.722 | 0.791 | 0.638 | +16.1% | +7.5% |
| LinearSVC | 0.713 | 0.009 | 0.740 | 0.814 | 0.464 | +14.8% | +10.1% |
| ComplementNB | 0.701 | 0.017 | 0.708 | 0.792 | 0.464 | +12.9% | +5.3% |
| LogReg, words + chars | 0.680 | 0.016 | 0.726 | 0.830 | 0.348 | +9.4% | +8.0% |
| RandomForest balanced | 0.674 | 0.013 | 0.683 | 0.772 | 0.464 | +8.5% | +1.6% |
| LogReg, unigrams | 0.643 | 0.022 | 0.708 | 0.801 | 0.348 | +3.5% | +5.4% |
| LogReg, char 2-5 grams | 0.636 | 0.021 | 0.679 | 0.810 | 0.261 | +2.5% | +1.0% |
| LogReg, negation tokens | 0.621 | 0.020 | 0.672 | 0.795 | 0.261 | +0.1% | +0.0% |
| Baseline TF-IDF + LogReg | 0.621 | 0.019 | 0.672 | 0.795 | 0.261 | +0.0% | +0.0% |
| RandomForest | 0.621 | 0.023 | 0.638 | 0.771 | 0.246 | -0.0% | -5.0% |
| LogReg, 1-3 grams | 0.615 | 0.022 | 0.667 | 0.796 | 0.246 | -1.0% | -0.8% |
