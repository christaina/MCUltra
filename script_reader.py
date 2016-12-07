from reader import load_data
from reader import get_vocab
from reader import vocab_transform
from reader import batch_iter


contexts, questions, choices, labels, choices_map, context_lens, qs_lens =\
    load_data(data_path="wdw/test")


# # 2. Fit vocabulary with questions and context.
vocab = get_vocab(contexts, questions)

# # 3. Transform context and questions
contexts = vocab_transform(contexts, vocab)
questions = vocab_transform(questions, vocab)

# 4. Give to batch_iter
readers = batch_iter(contexts, questions, choices, labels, choices_map,
           context_lens, qs_lens)

# for q, c, ch, lab, ch_map, c_lens, q_lens in readers:
#     print(c.shape)
#     break
