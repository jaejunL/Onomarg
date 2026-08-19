import json
from pathlib import Path
from eval1.validate_predictions import validate
def test_prediction_validator_fixture(tmp_path):
 manifest=tmp_path/'m.jsonl';pred=tmp_path/'p.jsonl';manifest.write_text(json.dumps({'eval_audio_id':'x'})+'\n');pred.write_text(''.join(json.dumps({'eval_audio_id':'x','beam_rank':i})+'\n' for i in range(3)));assert validate(manifest,pred,1)['prediction_rows']==3
