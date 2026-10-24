from jinja2.optimizer import optimize
from transformers.generation.continuous_batching.utils import create_warmup_future_states

from app.agents.agent_geometric_approach.data.text_preprocess import count_trans_freq, extract_alphabet, normalize_text
from kbrd.transform_layout import create_positions
from kbrd.utils import QWERTY_ROW, QWERTY_ROW_POS, ALPHABET
from kbrd.transform_layout import encode_alphabet, decode_kbrd_layout
from kbrd.transform_layout import kbrd_layout_to_array
from math_model.geometry import create_dist_matrix
from data.text_preprocess import create_trans_matrix
from math_model.optimize import optimize_kbrd_layout

key_to_idx, idx_to_key = encode_alphabet(ALPHABET)
kbrd_layout = kbrd_layout_to_array(QWERTY_ROW, key_to_idx)
positions, pos_array = create_positions(QWERTY_ROW_POS, key_to_idx)

dist_matrix = create_dist_matrix(pos_array)
alphabet, pattern = extract_alphabet(ALPHABET)
cnt = count_trans_freq(normalize_text("akljsklgncxmnbvuioquoiruqjhgbnvm bm,ajhflikahuiohqwuioghqiojioghuhjgksdhgkjhu", pattern))
_, trans_matrix = create_trans_matrix(cnt, alphabet)
best_layout, best_score = optimize_kbrd_layout(kbrd_layout, dist_matrix, trans_matrix)
print(best_layout, best_score)
print(decode_kbrd_layout(best_layout, idx_to_key))