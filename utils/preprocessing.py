import string
import itertools
import num2words
from nltk.corpus import words


def remove_header_and_footer(lines, start_barrier='*** START OF THIS PROJECT', end_barrier='*** END OF THIS PROJECT'):
    header_and_footer_positions = []
    start = 0
    end = len(lines)
    for pos, line in enumerate(lines):
        if line.startswith(start_barrier):
            start = pos
        elif line.startswith(end_barrier):
            end = pos
        else:
            continue
    lines = lines[start + 1:end]
    return lines


# http://www.gutenberg.org/files/60773/60773-0.txt
def find_chapter_words(lines,
                       minimal_word_length=4,
                       appearance_threshold=2,
                       words_in_sentance=4):
    class WordInfo:
        def __init__(self):
            self.count = 0
            self.positions = []

        def __repr__(self):
            return repr(self.positions)

    word_to_info = {}
    chapter_punctuation = string.punctuation
    for line_number, line in enumerate(lines):
        words = line.split()
        if not words or len(words) < 1:
            continue
        # First word should look like Chapter or CHAPTER or chapter.
        # We want to avoid connecting words such as OF and THE
        first_word = words[0].strip(chapter_punctuation)
        if len(first_word) < minimal_word_length:
            continue
        # last word should be the chapter id, e.g: 1, 11, X, I, A
        # so it is either a number or a capital text
        last_word = words[-1].strip(chapter_punctuation)
        # Should mostly be in the format of "Chapter _" so we expect at most two words
        is_last_word_chapter_index = last_word.isdigit() or last_word.isupper()
        is_chapter_candidate = first_word.isalpha() and is_last_word_chapter_index
        is_sentance = len(words) >= words_in_sentance
        if not is_chapter_candidate or is_sentance:
            continue
        if first_word not in word_to_info:
            word_to_info[first_word] = WordInfo()
        info = word_to_info[first_word]
        info.count += 1
        info.positions.append(line_number)
    meta_content_words = {k: v for (k, v) in word_to_info.items()
                          if v.count >= appearance_threshold
                          and (k.istitle() or k.isupper())}
    return meta_content_words.keys()


def remove_chapters(lines, chapter_expected_apperances=2):
    class LineInfo:
        def __init__(self):
            self.count = 0
            self.positions = []

        def __repr__(self):
            return repr(self.positions)

    chapter_lines = {}
    chapter_words = find_chapter_words(lines)
    for line_number, line in enumerate(lines):
        stripped_line = line.strip()
        for word in chapter_words:
            if stripped_line.startswith(word):
                if line not in chapter_lines:
                    chapter_lines[stripped_line] = LineInfo()
                chapter_line = chapter_lines[stripped_line]
                chapter_line.count += 1
                chapter_line.positions.append(line_number)
                break
    chapter_lines = {k: v for (k, v) in chapter_lines.items() if v.count == chapter_expected_apperances}
    largest_apperances = []
    for line_info in chapter_lines.values():
        # we iterate line by line so positions are already sorted
        largest_position = line_info.positions[1]
        largest_apperances.append(largest_position)
    # sorting in reverse so we can remove lines without affecting next line deletion
    largest_apperances = sorted(largest_apperances, reverse=True)
    first_chapter_second_apperance = largest_apperances[-1]
    last_chapter_second_apperance = largest_apperances[0]
    lines = lines[:last_chapter_second_apperance]
    for line_pos in largest_apperances[1:]:
        del lines[line_pos]
    lines = lines[first_chapter_second_apperance:]
    return lines


class SentanceTokenizer:
    def __init__(self,
                 keep_non_english_letters: bool,
                 keep_numbers: bool,
                 keep_spaces: bool,
                 stemming: bool):
        from nltk.corpus import stopwords
        from nltk.tokenize import RegexpTokenizer
        from nltk.stem import PorterStemmer

        # Tokenize text
        self.blacklisted_words = set(stopwords.words('english'))
        pattern = string.punctuation + r'\s'
        self.tokenizer = RegexpTokenizer(r'[{}]+'.format(pattern), gaps=True)
        self.keep_non_english_letters = keep_non_english_letters
        self.keep_spaces = keep_spaces
        self.keep_numbers = keep_numbers
        self.stemmer = PorterStemmer() if stemming else None
        self.english_words = set(words.words())

    def tokenize_sentance(self, line: str):
        # can be a string containing spaces with a punctuation inside
        def clean_space(token):
            return ' ' if token.count(' ') > 0 and self.keep_spaces else ''

        def clean_char(character):
            if not self.keep_non_english_letters and character not in string.ascii_lowercase:
                return ''
            else:
                return character

        def translate_token_to_tokens(token):
            if text_token.isalpha():
                return [token]
            elif text_token.isalnum():
                alpha_numeric_tokens = [''.join(x) for _, x in itertools.groupby(text_token, key=str.isdigit)]
                result = []
                for alpha_numeric_token in alpha_numeric_tokens:
                    if alpha_numeric_token.isalpha():
                        result.append(alpha_numeric_token)
                    elif self.keep_numbers:
                        number_sentance = num2words.num2words(alpha_numeric_token)
                        number_tokens = [word.lower().strip(string.punctuation) for word in number_sentance.split()]
                        result.extend(number_tokens)
                # if len(result) > 2:
                #     print(f'Alpha numeric: {result}')
                return result
            else:
                cleaned_text_token = ''.join([clean_char(c) for c in text_token])
                #if text_token != cleaned_text_token:
                #    print(f'{text_token} -> {cleaned_text_token}')
                return [cleaned_text_token]

        tokens = []

        def append_token(token):
            if not token or token in self.blacklisted_words:
                return
            if self.stemmer:
                token = self.stemmer.stem(token)
            tokens.append(token)

        prev_end = None
        words_span_in_line = self.tokenizer.span_tokenize(text=line)
        for span in words_span_in_line:
            start, end = span
            if prev_end is not None:
                space_token = line[prev_end:start]
                space_token = clean_space(space_token)
                if space_token:
                    tokens.append(space_token)
            prev_end = end
            text_token = line[start: end].lower()
            text_tokens = translate_token_to_tokens(text_token)
            for token in text_tokens:
                append_token(token)
        #         if text_token not in self.english_words and appended_token not in self.english_words:
        #             non_english_words.append(text_token)
        # if len(non_english_words) > 0:
        #     print(f'Tokens: {tokens}, non english words: {non_english_words}')
        return tokens

    def tokenize_sentances(self, lines):
        sentances = []
        for line in lines:
            tokens = self.tokenize_sentance(line)
            if len(tokens) > 0:
                sentances.append(tokens)
        return sentances


def tokenize_lines(lines, keep_non_english_letters, keep_spaces):
    tokenizer = SentanceTokenizer(keep_non_english_letters, keep_spaces)
    sentances = tokenizer.tokenize_sentances(lines, keep_non_english_letters, keep_spaces)
    return ''.join(sentances)