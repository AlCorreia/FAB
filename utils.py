import re

def get_word_span(context, spans, start, stop):
	search_index = 0
	word_id = []
	word_index=[]
	for word_idx, word in  enumerate(spans):
		word_init = context.find(word, search_index)
		assert word_init >= 0
		word_end = word_init + len(word)
		if  (stop > word_init and start < word_end):
			word_id.append(word_idx)
			word_index.append([word_init, word_end])
		search_index=word_end

	assert len(word_id) > 0, "{} {} {} {}".format(context, spans, start, stop)
	return word_id[0], (word_id[-1] + 1), word_index[0][0], word_index[-1][0] 


def process_tokens(temp_tokens):
	tokens = []
	for token in temp_tokens:
		flag = False
		l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
		# \u2013 is en-dash. Used for number to nubmer
		# l = ("-", "\u2212", "\u2014", "\u2013")
		# l = ("\u2013",)
		tokens.extend(re.split("([{}])".format("".join(l)), token))
	return tokens