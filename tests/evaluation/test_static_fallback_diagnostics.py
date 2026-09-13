import numpy as np

from scripts.evaluation.diagnose_static_fallback import diagnose_added


def test_added_duplicate_wrong_class_and_new_object_are_separate():
    gt = np.array([1000,1000,2000,2000,3000,3000])
    def prediction(mask, label):
        return {'mask': np.array(mask,dtype=bool), 'class_id': label, 'confidence': 1.}
    before = {1: prediction([1,1,0,0,0,0],1)}
    after = {**before, 2: prediction([1,1,0,0,0,0],1),
             3: prediction([0,0,1,1,0,0],2), 4: prediction([0,0,0,0,1,1],2)}
    result = diagnose_added(before, after, gt, [1,2,3], min_region=1)
    assert [x['thresholds']['0.5']['outcome'] for x in result] == ['DUPLICATE_FP','NOVEL_TP','FP']


def test_class_outside_released_instance_vocabulary_is_ignored():
    after = {1: {'mask': np.ones(2,dtype=bool), 'class_id': 99, 'confidence': 1.}}
    result = diagnose_added({}, after, np.array([1000,1000]), [1], min_region=1)
    assert result[0]['thresholds']['0.5']['outcome'] == 'IGNORED'
