import importlib.util
from pathlib import Path
import pytest
from incident_alignment.deepwide import DeepWideConfig

def test_config_rejects_invalid():
    with pytest.raises(ValueError):
        DeepWideConfig(epochs=0)
    with pytest.raises(ValueError):
        DeepWideConfig(dropout=1.0)

def test_modules_import_without_torch():
    from incident_alignment.deepwide import train_prepared_fold, predict_pairs
    assert callable(train_prepared_fold) and callable(predict_pairs)

@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='torch optional')
def test_checkpoint_roundtrip(tmp_path):
    from incident_alignment.deepwide import build_model, save_model_checkpoint, load_model_checkpoint
    import torch
    model, cfg = build_model(pair_dim=12, tab_input_dim=5, dropout=0.0)
    model.eval()
    a = torch.randn(4,5)
    b = torch.randn(4,12)
    with torch.no_grad():
        expect = model(a,b).clone()
    ckpt=tmp_path/'test.pt'
    save_model_checkpoint(ckpt,model,{'model_config':cfg})
    rebuilt,meta,_=load_model_checkpoint(ckpt)
    with torch.no_grad():
        actual=rebuilt(a,b)
    assert torch.equal(expect,actual)
    assert meta['framework']=='pairwise-current'

@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='torch optional')
def test_old_style_checkpoint_without_config(tmp_path):
    from incident_alignment.deepwide import build_model, load_model_checkpoint
    import torch
    model,_=build_model(pair_dim=12, tab_input_dim=5)
    p=tmp_path/'old.pt'
    torch.save({'framework':'pairwise-current','state_dict':model.state_dict()},p)
    rebuilt,meta,_=load_model_checkpoint(p)
    assert meta['model_config']['pair_dim']==12
    assert rebuilt.wide_linear.in_features==5
