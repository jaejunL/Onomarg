from eval1.phonology import english_to_phones_batch,korean_to_phones_batch
def test_no_embedded_word_boundaries():
 for seq in english_to_phones_batch(['dring dring','chwa reu'])+korean_to_phones_batch(['빠 아 앙']):assert all('|' not in x for x in seq)
