import logging

from transformers.configuration_utils import PretrainedConfig
from transformers import BertModel, BertConfig, GPT2Config



logger = logging.getLogger(__name__)
## 512 넘는 것은 학습하지 않음
#KoBERT 환결설정
kobert_config = {
    'attention_probs_dropout_prob': 0.1,
    'hidden_act': 'gelu',
    'hidden_dropout_prob': 0.1,
    'hidden_size': 768,
    'initializer_range': 0.02,
    'intermediate_size': 3072, # transformer 내에 있는 feed-forward network의 dimension size
    'max_position_embeddings': 512, #최대 몇 토큰 까지 임베딩 할 것인가?
    'num_attention_heads': 12, # transformer attention head number
    'num_hidden_layers': 12,
    'type_vocab_size': 2, # segment A,B 두종류임
    'vocab_size': 8002
}
#KoGPT2
kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000,
    "activation_function": "gelu"
}

def get_kobert_config():
    return BertConfig.from_dict(kobert_config)

def get_kogpt2_config():
    return GPT2Config.from_dict(kogpt2_config)