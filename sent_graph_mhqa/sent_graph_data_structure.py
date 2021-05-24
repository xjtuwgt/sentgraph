class Example(object):
    def __init__(self,
                 qas_id,
                 question_tokens,
                 doc_tokens,
                 sent_num,
                 sent_names,
                 para_names,
                 ques_entities_text,
                 ctx_entities_text,
                 para_start_end_position,
                 sent_start_end_position,
                 ques_entity_start_end_position,
                 ctx_entity_start_end_position,
                 question_text,
                 question_word_to_char_idx,
                 answer_candidates_in_ctx_entity_ids,
                 ctx_text,
                 ctx_word_to_char_idx,
                 edges=None):
        self.qas_id = qas_id
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.para_names = para_names
        self.ques_entities_text = ques_entities_text
        self.ctx_entities_text = ctx_entities_text
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.ques_entity_start_end_position = ques_entity_start_end_position
        self.ctx_entity_start_end_position = ctx_entity_start_end_position
        self.question_word_to_char_idx = question_word_to_char_idx
        self.ctx_text = ctx_text
        self.ctx_word_to_char_idx = ctx_word_to_char_idx
        self.answer_candidates_in_ctx_entity_ids = answer_candidates_in_ctx_entity_ids
        self.edges = edges