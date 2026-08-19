from data.text_normalization import normalize_text
def test_normalize(): assert normalize_text('* A  A !')=='a a'
