from eval1.compare_models import paired_delta
def test_paired_delta_direction():
 a=[{'eval_audio_id':'x','class_key':'c','m':.2},{'eval_audio_id':'y','class_key':'c','m':.4}];b=[{'eval_audio_id':'x','class_key':'c','m':.3},{'eval_audio_id':'y','class_key':'c','m':.7}];assert paired_delta(a,b,'m')['mean_delta']<0
