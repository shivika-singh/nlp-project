#!/usr/bin/env python3
"""
NLPBot v3 - Intent-based NLP chatbot (ONE FILE, pure Python, no installs)

NLP techniques used:
  1. Preprocessing      : lowercase, tokenization, stopword removal, stemming
  2. Spelling correction: Levenshtein edit distance against the known vocabulary
  3. TF-IDF vectors     : class IntentClassifier -> self.idf, _vector(), analyze()
  4. Cosine similarity  : user vector vs every training-pattern vector
  5. Sentiment analysis : lexicon + negation + emoji -> emotion-aware replies

How to run (Terminal):
  python3 chatbot.py          -> opens the chat window in your browser
  python3 chatbot.py --cli    -> chat inside the terminal
  python3 chatbot.py --demo   -> sample conversations
  python3 chatbot.py --eval   -> accuracy / precision / recall / F1 evaluation
"""
import json
import math
import random
import re
import sys
import threading
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer

VERSION = "v3 (spelling correction + visible TF-IDF)"

# ============================================================================
# 1 + 2. PREPROCESSING AND INTENT KNOWLEDGE BASE
# ============================================================================
STOPWORDS = set("""a an the is are was were be been am do does did to of in on at for with and or
but if so than then that this these those it its i me my you your we our they them he she
can could would should will shall may might please tell about explain give what whats how
which who whom whose when where why any some just also very really like want need let know
""".split())
# note: negations (not, no, never) are intentionally NOT removed here


def tokenize(text):
    """Lowercase, strip punctuation, split into word tokens."""
    return re.findall(r"[a-z0-9']+", text.lower())


def stem(word):
    """Very small suffix-stripping stemmer (Porter-style, simplified)."""
    for suf in ("ization", "ational", "ations", "ation", "ingly", "ities", "ness",
                "ment", "ing", "ies", "ied", "ers", "est", "ed", "ly", "es", "er", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)]
            if suf in ("ies", "ied"):
                word += "y"
            break
    return word


def preprocess(text):
    """Full pipeline: tokenize -> remove stopwords -> stem."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]


# ----------------------------------------------------------------------------
# 2. INTENT DEFINITIONS (training patterns + responses)
# ----------------------------------------------------------------------------
INTENTS = {
    "greeting": {
        "patterns": ["hello", "hi there", "hey", "good morning", "good evening", "greetings"],
        "responses": ["Hello! I'm NLPBot. Ask me about NLP concepts or AI applications.",
                      "Hi there! Want to learn about tokenization, TF-IDF, or AI in healthcare?"],
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "see you later", "quit", "exit", "talk to you later"],
        "responses": ["Goodbye! Keep exploring AI. 👋", "See you later! Happy learning."],
    },
    "thanks": {
        "patterns": ["thanks", "thank you", "thanks a lot", "that was helpful", "great help"],
        "responses": ["You're welcome! Ask me anything else about NLP or AI.",
                      "Glad I could help!"],
    },
    "capabilities": {
        "patterns": ["what can you do", "what topics do you know", "who are you",
                     "what do you know", "your abilities"],
        "responses": ["I'm an intent-based chatbot. Try: 'what is NLP', 'tokenization', "
                      "'stemming vs lemmatization', 'TF-IDF', 'named entity recognition', "
                      "'transformers', 'AI in healthcare', 'AI in finance', 'self-driving cars'."],
    },
    # ---------------- NLP concepts ----------------
    "what_is_nlp": {
        "patterns": ["what is nlp", "define natural language processing", "meaning of nlp",
                     "explain nlp", "natural language processing introduction"],
        "responses": ["NLP (Natural Language Processing) is a branch of AI that enables computers "
                      "to read, understand, interpret and generate human language. Typical tasks: "
                      "translation, sentiment analysis, chatbots, summarization, speech recognition."],
    },
    "tokenization": {
        "patterns": ["what is tokenization", "explain tokenizer", "split text into words",
                     "word tokenization sentence tokenization", "tokens"],
        "responses": ["Tokenization splits raw text into smaller units (tokens) such as words, "
                      "sub-words or sentences. Example: 'I love NLP' -> ['I', 'love', 'NLP']. "
                      "It is the first step of almost every NLP pipeline."],
    },
    "stopwords": {
        "patterns": ["what are stop words", "stopword removal", "remove common words",
                     "stopwords in nlp"],
        "responses": ["Stop words are very common words (is, the, and, of) that carry little "
                      "meaning. Removing them shrinks the vocabulary and focuses the model on "
                      "informative words. Caution: negations like 'not' should usually be kept."],
    },
    "stemming_lemmatization": {
        "patterns": ["stemming", "lemmatization", "difference between stemming and lemmatization",
                     "what is a stemmer", "what is lemma", "root form of word"],
        "responses": ["Stemming chops suffixes using rules ('studies' -> 'studi'), it is fast but crude. "
                      "Lemmatization uses vocabulary and grammar to return a real dictionary form "
                      "('studies' -> 'study', 'better' -> 'good'). Lemmatization is more accurate, "
                      "stemming is faster."],
    },
    "bow_tfidf": {
        "patterns": ["what is tf-idf", "tfidf", "term frequency inverse document frequency",
                     "bag of words", "text vectorization", "convert text to numbers"],
        "responses": ["Bag-of-Words counts word occurrences per document. TF-IDF improves on it: "
                      "TF = how often a term appears in a document, IDF = log(N / docs containing term). "
                      "Words frequent in one document but rare overall get a high weight. "
                      "I use TF-IDF + cosine similarity to understand your questions!"],
    },
    "embeddings": {
        "patterns": ["word embeddings", "word2vec", "glove", "what is embedding",
                     "vector representation of words", "semantic similarity"],
        "responses": ["Word embeddings (Word2Vec, GloVe, FastText) map words to dense vectors so that "
                      "similar words lie close together, e.g. king - man + woman ~ queen. "
                      "Modern models (BERT) give context-dependent embeddings."],
    },
    "pos_tagging": {
        "patterns": ["pos tagging", "part of speech tagging", "what are noun verb adjective tags",
                     "grammar tagging"],
        "responses": ["POS tagging labels each word with its grammatical role (noun, verb, adjective...). "
                      "Example: 'The cat sleeps' -> The/DET cat/NOUN sleeps/VERB. "
                      "It helps in parsing, NER and information extraction."],
    },
    "ner": {
        "patterns": ["named entity recognition", "ner", "extract names places organizations",
                     "entity extraction"],
        "responses": ["NER finds and classifies entities in text: persons, organizations, locations, "
                      "dates, money. Example: 'Sundar Pichai works at Google in California' -> "
                      "PERSON, ORG, LOCATION."],
    },
    "sentiment_analysis": {
        "patterns": ["what is sentiment analysis", "opinion mining", "classify positive negative text",
                     "emotion detection", "sentiment of reviews"],
        "responses": ["Sentiment analysis identifies the opinion in text: positive, negative or neutral "
                      "(emotion detection goes further: happy, sad, angry, fear...). Used for movie "
                      "reviews, tweets, product feedback and stock news. Approaches: lexicon-based "
                      "(VADER), classical ML (Naive Bayes, SVM) and deep learning (BERT)."],
    },
    "transformers": {
        "patterns": ["what are transformers", "bert", "gpt", "attention mechanism",
                     "large language models", "llm", "self attention"],
        "responses": ["Transformers are neural networks built on self-attention, which lets every word "
                      "look at every other word in a sentence in parallel. They power BERT, GPT and "
                      "today's large language models (LLMs) used for chatbots, translation and coding."],
    },
    "machine_translation": {
        "patterns": ["machine translation", "language translation", "google translate",
                     "translate languages automatically"],
        "responses": ["Machine translation automatically converts text between languages. It moved from "
                      "rule-based to statistical to neural (seq2seq + attention, Transformers). "
                      "Example: Google Translate."],
    },
    # ---------------- AI applications ----------------
    "what_is_ai": {
        "patterns": ["what is ai", "define artificial intelligence", "explain ai",
                     "meaning of artificial intelligence"],
        "responses": ["Artificial Intelligence is the field of building machines that perform tasks "
                      "needing human intelligence: learning, reasoning, perception and language "
                      "understanding. Machine Learning and Deep Learning are subsets of AI."],
    },
    "ml_vs_dl": {
        "patterns": ["difference between machine learning and deep learning", "what is machine learning",
                     "what is deep learning", "ml vs dl", "neural network"],
        "responses": ["Machine Learning lets systems learn patterns from data instead of explicit rules. "
                      "Deep Learning is a subset of ML using multi-layer neural networks, great for "
                      "images, speech and text but needing more data and compute."],
    },
    "ai_healthcare": {
        "patterns": ["ai in healthcare", "medical applications of ai", "disease detection using ai",
                     "ai doctors hospitals", "medical imaging ai"],
        "responses": ["AI in healthcare: early disease detection from X-rays/MRI, drug discovery, "
                      "personalised treatment, virtual health assistants, and NLP for mining patient "
                      "records and clinical notes."],
    },
    "ai_finance": {
        "patterns": ["ai in finance", "ai in banking", "banking applications of ai", "fraud detection",
                     "stock market prediction ai", "credit scoring"],
        "responses": ["AI in finance: fraud detection, credit scoring, algorithmic trading, risk "
                      "management, robo-advisors, and sentiment analysis of financial news and tweets "
                      "to gauge market mood."],
    },
    "ai_education": {
        "patterns": ["ai in education", "personalized learning", "intelligent tutoring system",
                     "ai for students teachers", "online course analysis"],
        "responses": ["AI in education: personalised learning paths, intelligent tutors, automatic "
                      "grading, plagiarism detection, and analysing student feedback on online courses."],
    },
    "ai_agriculture": {
        "patterns": ["ai in agriculture", "farming using ai", "crop disease detection",
                     "smart farming drones"],
        "responses": ["AI in agriculture: crop-disease detection from leaf images, yield prediction, "
                      "soil and weather analysis, drone monitoring and precision irrigation."],
    },
    "ai_transport": {
        "patterns": ["self driving cars", "autonomous vehicles", "ai in transportation",
                     "traffic prediction", "route optimization"],
        "responses": ["AI in transport: self-driving cars (computer vision + sensor fusion), traffic "
                      "prediction, route optimisation, and predictive maintenance."],
    },
    "ai_ecommerce": {
        "patterns": ["ai in ecommerce", "recommendation system", "product recommendations",
                     "netflix amazon recommendation", "customer service chatbots"],
        "responses": ["AI in e-commerce/media: recommendation engines (Amazon, Netflix), dynamic pricing, "
                      "visual search, demand forecasting, review analysis and customer-support chatbots."],
    },
}

POSITIVE = set("good great awesome love excellent amazing happy nice helpful wonderful cool fantastic "
               "perfect brilliant enjoy fun interesting thanks".split())
NEGATIVE = set("bad terrible awful hate worst sad angry boring useless confusing difficult hard "
               "stupid annoying frustrated confused poor horrible".split())
NEGATIONS = {"not", "no", "never", "dont", "don't", "isn't", "wasn't", "cannot", "can't"}
EMOJI = {"😊": 1, "😀": 1, "👍": 1, "❤️": 1, "😢": -1, "😡": -1, "👎": -1, "😞": -1}


def detect_sentiment(text):
    score, toks = 0, tokenize(text)
    for i, t in enumerate(toks):
        val = 1 if t in POSITIVE else -1 if t in NEGATIVE else 0
        if val and i > 0 and toks[i - 1] in NEGATIONS:
            val = -val
        score += val
    score += sum(v for e, v in EMOJI.items() if e in text)
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"



# ============================================================================
# 3. SPELLING CORRECTION + TF-IDF INTENT CLASSIFIER
# ============================================================================
def edit_distance(a, b):
    """Damerau-Levenshtein distance: insert / delete / substitute / swap-adjacent each cost 1."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)      # transposition (teh -> the)
    return d[len(a)][len(b)]


# Ordinary English words that must never be "corrected" into domain words
COMMON_WORDS = set("""work works worked working word words use used using make makes made help helps helped
learn learns learning teach does doing done come comes goes going take takes check write written read reads
build built apply applied model models data code text speech voice image images video search find found
look looks think thinks start started begin begins first second third other another there their these those
about after before under over every each many much more most less least same different simple basic best
better good great small large long short high important example examples used bot bots chatbot chatbots""".split())


class IntentClassifier:
    """
    mode : "bow" (raw counts) | "tfidf" (default) | "tfidf_bigram" (unigrams + word pairs)
    spell: True -> fix misspelled words with edit distance before classification
    """

    def __init__(self, intents, mode="tfidf", spell=True):
        self.mode, self.spell = mode, spell
        self.vocab = Counter()                       # known words -> frequency
        for data in intents.values():
            for p in data["patterns"]:
                self.vocab.update(t for t in tokenize(p) if t not in STOPWORDS)
        self.docs = []                               # (intent, original pattern, features)
        for name, data in intents.items():
            for p in data["patterns"]:
                self.docs.append((name, p, self._features(preprocess(p))))
        n = len(self.docs)
        df = Counter(t for _, _, f in self.docs for t in set(f))   # document frequency
        if mode == "bow":
            self.idf = {t: 1.0 for t in df}
        else:                                        # <<< TF-IDF: inverse document frequency
            self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.vecs = [(name, p, self._vector(f)) for name, p, f in self.docs]

    def _features(self, stems):
        feats = list(stems)
        if self.mode == "tfidf_bigram":
            feats += [a + "_" + b for a, b in zip(stems, stems[1:])]
        return feats

    PRIORITY = {"what", "how", "why", "who", "when", "where", "which", "tell", "explain", "define"}

    def correct(self, text):
        """Returns (corrected token list, [(wrong, fixed), ...]).
        - words of 5+ letters are matched against the domain vocabulary AND everyday stopwords
        - words of 3-4 letters are matched against stopwords only (wat -> what, hw -> skipped)
        - tolerance grows with word length (1, 2 or 3 edits); swapped letters count as 1 edit"""
        tokens = tokenize(text)
        if not self.spell:
            return tokens, []
        known = set(self.vocab) | STOPWORDS | POSITIVE | NEGATIVE | NEGATIONS | COMMON_WORDS
        out, changes = [], []
        for t in tokens:
            if t in known or len(t) < 3 or t.isdigit() or not t.isalpha():
                out.append(t)
                continue
            if len(t) <= 4:
                pool, max_d = [w for w in STOPWORDS if len(w) >= 3], 1
            else:
                pool, max_d = list(self.vocab) + [w for w in STOPWORDS if len(w) >= 4], \
                    (1 if len(t) <= 6 else 2 if len(t) <= 9 else 3)
            cands = []
            for v in pool:
                if abs(len(v) - len(t)) > max_d:
                    continue
                d = edit_distance(t, v)
                if d <= max_d:
                    cands.append((d, v[0] != t[0], v not in self.PRIORITY, v not in self.vocab, -self.vocab[v], v))
            if cands:
                best = min(cands)[-1]
                out.append(best)
                changes.append((t, best))
            else:
                out.append(t)
        return out, changes

    def _vector(self, tokens):
        """TF-IDF vector, length-normalised: weight = TF x IDF."""
        tf = Counter(tokens)
        vec = {t: (c / max(len(tokens), 1)) * self.idf.get(t, 0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    @staticmethod
    def _cosine(a, b):
        return sum(v * b.get(t, 0.0) for t, v in a.items())

    def analyze(self, text):
        """Full pipeline result (used by the chat window to show every step)."""
        tokens = tokenize(text)
        corrected, changes = self.correct(text)
        stems = [stem(t) for t in corrected if t not in STOPWORDS]
        res = dict(tokens=tokens, corrected=corrected, changes=changes, cleaned=stems,
                   weights=[], pattern="", intent=None, conf=0.0)
        if not stems:
            return res
        q = self._vector(self._features(stems))
        res["weights"] = [[t, round(w, 3)] for t, w in sorted(q.items(), key=lambda kv: -kv[1]) if w > 0][:6]
        best_intent, best_pat, best = None, "", 0.0
        for name, pat, vec in self.vecs:             # cosine similarity with every pattern
            s = self._cosine(q, vec)
            if s > best:
                best_intent, best_pat, best = name, pat, s
        res.update(intent=best_intent, pattern=best_pat, conf=best)
        return res

    def predict(self, text):
        a = self.analyze(text)
        return a["intent"], a["conf"]



# ============================================================================
# 4. THE CHATBOT
# ============================================================================
FALLBACK = ("Sorry, I didn't understand that. I can answer about NLP concepts "
            "(tokenization, TF-IDF, NER, transformers...) and AI applications "
            "(healthcare, finance, education...). Type 'help' for ideas.")


class NLPBot:
    THRESHOLD = 0.30

    def __init__(self, mode="tfidf", spell=True):
        self.clf = IntentClassifier(INTENTS, mode=mode, spell=spell)

    def reply(self, text):
        sentiment = detect_sentiment(text)
        a = self.clf.analyze(text)
        intent, conf = a["intent"], a["conf"]
        if text.strip().lower() == "help":
            intent, conf = "capabilities", 1.0
        if intent is None or conf < self.THRESHOLD:
            answer, intent = FALLBACK, "fallback"
        else:
            answer = random.choice(INTENTS[intent]["responses"])
        if sentiment == "negative":
            answer = "I'm sorry this feels frustrating - let's take it step by step. " + answer
        elif sentiment == "positive" and intent not in ("greeting", "thanks", "goodbye"):
            answer = "Glad you're enjoying this! " + answer
        a.update(reply=answer, intent=intent, conf=conf, sentiment=sentiment)
        return a


def describe(r):
    lines = []
    if r["changes"]:
        lines.append("   [spelling corrected: " + ", ".join(f"{a} -> {b}" for a, b in r["changes"]) + "]")
    lines.append(f"   [intent={r['intent']}, confidence={r['conf']:.2f}, sentiment={r['sentiment']}]")
    if r["weights"]:
        lines.append("   [TF-IDF weights: " + ", ".join(f"{t}={w:.2f}" for t, w in r["weights"]) + "]")
    return "\n".join(lines)


def run_cli():
    bot = NLPBot()
    print(f"NLPBot {VERSION}\nNLPBot: Hi! Ask me about NLP or AI applications (type 'quit' to exit).")
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        r = bot.reply(msg)
        print("NLPBot:", r["reply"])
        print(describe(r))
        if r["intent"] == "goodbye":
            break


def run_demo():
    random.seed(1)
    bot = NLPBot()
    s = "The students are studying Natural Language Processing!"
    print(f"NLPBot {VERSION}\nPreprocessing example\n  Input   : {s}\n  Tokens  : {tokenize(s)}\n  Cleaned : {preprocess(s)}\n")
    convs = [
        ["hello", "what is NLP?", "explain tokenization", "difference between stemming and lemmatization", "thanks", "bye"],
        ["Tell me about TF-IDF", "how is AI used in healthcare", "I hate fraud, how does AI in banking help?",
         "AI in education is awesome", "what is the weather today?"],
        ["wat is tokenisation", "explan stemming", "helthcare ai", "how do transfomers work", "named entitiy recognition", "quit"],
    ]
    for i, conv in enumerate(convs, 1):
        print("=" * 70 + f"\nSample Conversation {i}" + (" (with spelling mistakes)" if i == 3 else "") + "\n" + "=" * 70)
        for msg in conv:
            r = bot.reply(msg)
            print(f"You   : {msg}\nBot   : {r['reply']}\n{describe(r)}\n")


# ============================================================================
# 5. CHAT WINDOW (browser) - small local web server, standard library only
# ============================================================================
PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NLPBot - Intent-Based NLP Chatbot</title>
<style>
  *{box-sizing:border-box} body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:#eef1f7;color:#1c2333}
  header{background:#3b4cca;color:#fff;padding:14px 22px} header h1{margin:0;font-size:20px} header p{margin:3px 0 0;font-size:13px;opacity:.85}
  .wrap{display:flex;gap:16px;padding:16px;max-width:1200px;margin:auto;height:calc(100vh - 78px)}
  .card{background:#fff;border-radius:14px;box-shadow:0 2px 10px rgba(0,0,0,.08);display:flex;flex-direction:column;min-height:0}
  .chat{flex:3} .info{flex:2;padding:16px;overflow:auto}
  #log{flex:1;overflow:auto;padding:18px;display:flex;flex-direction:column;gap:10px}
  .msg{max-width:82%;padding:10px 14px;border-radius:14px;line-height:1.45;font-size:15px;white-space:pre-wrap}
  .user{align-self:flex-end;background:#3b4cca;color:#fff;border-bottom-right-radius:4px}
  .bot{align-self:flex-start;background:#eef1f7;border-bottom-left-radius:4px}
  .fix{margin-top:6px;font-size:12px;color:#b45309;background:#fff4e0;border-radius:8px;padding:4px 8px}
  .tags{margin-top:6px;font-size:11px;color:#5b6478} .tag{display:inline-block;background:#dde3f5;border-radius:8px;padding:2px 7px;margin-right:4px}
  .neg{background:#fde0e0}.pos{background:#dcf3df}
  .chips{padding:0 14px 8px;display:flex;flex-wrap:wrap;gap:6px}
  .chip{border:1px solid #c5cdea;background:#fff;color:#3b4cca;border-radius:16px;padding:4px 11px;font-size:12px;cursor:pointer}
  .chip:hover{background:#eef1ff}
  form{display:flex;gap:8px;padding:12px;border-top:1px solid #e4e8f2}
  input{flex:1;padding:11px 14px;border:1px solid #c5cdea;border-radius:22px;font-size:15px;outline:none} input:focus{border-color:#3b4cca}
  button.send{background:#3b4cca;color:#fff;border:0;border-radius:22px;padding:0 22px;font-size:15px;cursor:pointer}
  .info h3{margin:0 0 4px;font-size:15px} .info small{color:#5b6478}
  .step{margin-top:10px} .step b{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#3b4cca}
  .box{margin-top:4px;background:#f5f7fc;border-radius:8px;padding:8px 10px;font-family:Menlo,Consolas,monospace;font-size:12.5px;word-break:break-word;white-space:pre-wrap}
  .bar{height:10px;background:#e4e8f2;border-radius:6px;overflow:hidden;margin-top:6px} .bar i{display:block;height:100%;background:#3b4cca;width:0}
  @media(max-width:800px){.wrap{flex-direction:column;height:auto}.chat{height:70vh}}
</style></head><body>
<header><h1>NLPBot - Intent-Based NLP Chatbot</h1>
<p>__VERSION__ &nbsp;|&nbsp; Ask about NLP concepts or AI applications. Try a spelling mistake: "wat is tokenisation"</p></header>
<div class="wrap">
  <div class="card chat">
    <div id="log"></div>
    <div class="chips" id="chips"></div>
    <form id="f"><input id="q" autocomplete="off" placeholder="Type your question, e.g. what is tokenization?" autofocus>
    <button class="send">Send</button></form>
  </div>
  <div class="card info">
    <h3>Under the hood</h3><small>NLP pipeline for your last message</small>
    <div class="step"><b>1. Original text</b><div class="box" id="s1">-</div></div>
    <div class="step"><b>2. Tokens</b><div class="box" id="s2">-</div></div>
    <div class="step"><b>3. Spelling correction (edit distance)</b><div class="box" id="s3">-</div></div>
    <div class="step"><b>4. Stopword removal + stemming</b><div class="box" id="s4">-</div></div>
    <div class="step"><b>5. TF-IDF weights of your words</b><div class="box" id="s5">-</div></div>
    <div class="step"><b>6. Closest training pattern (cosine)</b><div class="box" id="s6">-</div><div class="bar"><i id="bar"></i></div></div>
    <div class="step"><b>7. Intent / confidence</b><div class="box" id="s7">-</div></div>
    <div class="step"><b>8. User sentiment</b><div class="box" id="s8">-</div></div>
  </div>
</div>
<script>
const log=document.getElementById('log'), q=document.getElementById('q');
const samples=["hello","what is NLP?","explain tokenization","wat is tkenzation","explan stemming","helthcare ai","how do transfomers work","named entitiy recognition","stemming vs lemmatization","what is TF-IDF","I hate fraud, how does AI in banking help?","AI in education is awesome","what is the weather today?"];
function add(text,cls,m){
  const d=document.createElement('div'); d.className='msg '+cls; d.textContent=text;
  if(m){
    if(m.changes.length){const x=document.createElement('div'); x.className='fix';
      x.textContent='Spelling corrected: '+m.changes.map(c=>c[0]+' → '+c[1]).join(', '); d.appendChild(x);}
    const t=document.createElement('div'); t.className='tags';
    t.innerHTML='<span class="tag">intent: '+m.intent+'</span><span class="tag">confidence: '+m.conf.toFixed(2)+'</span><span class="tag '+(m.sentiment=='negative'?'neg':m.sentiment=='positive'?'pos':'')+'">sentiment: '+m.sentiment+'</span>';
    d.appendChild(t);}
  log.appendChild(d); log.scrollTop=log.scrollHeight;
}
function panel(m,text){
  s1.textContent=text; s2.textContent=JSON.stringify(m.tokens);
  s3.textContent=m.changes.length? m.changes.map(c=>c[0]+'  ->  '+c[1]).join('\n') : 'no spelling mistakes found';
  s4.textContent=JSON.stringify(m.cleaned);
  s5.textContent=m.weights.length? m.weights.map(w=>w[0]+' : '+w[1].toFixed(3)).join('\n') : '-';
  s6.textContent=m.pattern? '"'+m.pattern+'"  ->  cosine = '+m.conf.toFixed(2) : 'no match';
  bar.style.width=(m.conf*100)+'%';
  s7.textContent=m.intent+'  ('+m.conf.toFixed(2)+(m.intent=='fallback'?', below 0.30 threshold':'')+')';
  s8.textContent=m.sentiment;
}
async function send(text){
  if(!text.trim())return; add(text,'user'); q.value='';
  const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});
  const m=await r.json(); add(m.reply,'bot',m); panel(m,text);
}
document.getElementById('f').onsubmit=e=>{e.preventDefault();send(q.value)};
samples.forEach(s=>{const c=document.createElement('span');c.className='chip';c.textContent=s;c.onclick=()=>send(s);chips.appendChild(c)});
add("Hello! I'm NLPBot. Ask me about NLP concepts or AI applications - type below or click a suggestion.",'bot');
</script></body></html>""".replace("__VERSION__", VERSION)

_bot = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        text = json.loads(self.rfile.read(length) or b"{}").get("message", "")
        self._send(200, json.dumps(_bot.reply(text)).encode("utf-8"), "application/json")

    def log_message(self, *args):
        pass


def run_web():
    global _bot
    _bot = NLPBot()
    server, port = None, 8000
    while server is None and port < 8020:
        try:
            server = HTTPServer(("127.0.0.1", port), Handler)
        except OSError:
            port += 1
    url = f"http://localhost:{port}"
    print(f"NLPBot {VERSION}\nRunning at {url}  (press Ctrl+C to stop)")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


# ============================================================================
# 6. EVALUATION (accuracy, precision, recall, F1) - python3 chatbot.py --eval
# ============================================================================
THRESHOLD = NLPBot.THRESHOLD

# (question, true intent). "fallback" = out-of-scope question the bot should refuse.
CLEAN = [
    ("hello bot", "greeting"), ("good morning bot", "greeting"), ("hello there", "greeting"),
    ("goodbye for now", "goodbye"), ("ok bye", "goodbye"), ("see you", "goodbye"),
    ("thank you so much", "thanks"), ("thanks for helping", "thanks"),
    ("tell me what you can do", "capabilities"), ("what topics can you answer", "capabilities"),
    ("can you explain natural language processing", "what_is_nlp"), ("what does nlp mean", "what_is_nlp"),
    ("define nlp", "what_is_nlp"),
    ("how does a tokenizer work", "tokenization"), ("what are tokens in nlp", "tokenization"),
    ("how to split sentences into words", "tokenization"),
    ("why remove stop words", "stopwords"), ("what are stopwords", "stopwords"),
    ("removing common words from text", "stopwords"),
    ("what is lemmatization", "stemming_lemmatization"), ("explain stemming", "stemming_lemmatization"),
    ("how does a stemmer reduce words to the root", "stemming_lemmatization"),
    ("explain tf idf", "bow_tfidf"), ("how does bag of words work", "bow_tfidf"),
    ("how do we convert text to numeric vectors", "bow_tfidf"),
    ("explain word2vec", "embeddings"), ("what are word embeddings", "embeddings"),
    ("how are words represented as vectors", "embeddings"),
    ("what is part of speech tagging", "pos_tagging"), ("how do taggers label nouns and verbs", "pos_tagging"),
    ("what is named entity recognition", "ner"), ("how to extract names of people and organizations", "ner"),
    ("what is sentiment analysis", "sentiment_analysis"), ("how to detect opinion in reviews", "sentiment_analysis"),
    ("emotion detection from text", "sentiment_analysis"),
    ("how does the attention mechanism work", "transformers"), ("what is bert", "transformers"),
    ("explain large language models", "transformers"),
    ("how does google translate work", "machine_translation"), ("automatic translation of languages", "machine_translation"),
    ("what is artificial intelligence", "what_is_ai"), ("define ai", "what_is_ai"),
    ("what is deep learning", "ml_vs_dl"), ("machine learning versus deep learning", "ml_vs_dl"),
    ("how do neural networks work", "ml_vs_dl"),
    ("how does ai help in medicine", "ai_healthcare"), ("medical imaging using ai", "ai_healthcare"),
    ("ai for disease detection", "ai_healthcare"),
    ("how is ai used in banks", "ai_finance"), ("fraud detection with ai", "ai_finance"),
    ("ai in stock trading", "ai_finance"),
    ("how can ai help students", "ai_education"), ("personalized learning with ai", "ai_education"),
    ("how do farmers use ai", "ai_agriculture"), ("crop disease detection", "ai_agriculture"),
    ("drones in smart farming", "ai_agriculture"),
    ("how do self driving cars work", "ai_transport"), ("ai for traffic prediction", "ai_transport"),
    ("autonomous vehicles", "ai_transport"),
    ("how does netflix recommend movies", "ai_ecommerce"), ("recommendation system for shopping", "ai_ecommerce"),
    ("chatbots for customer service", "ai_ecommerce"),
    ("what is the weather today", "fallback"), ("who won the cricket match", "fallback"),
    ("tell me a joke", "fallback"), ("how to cook pasta", "fallback"),
    ("what is the capital of france", "fallback"), ("book a flight ticket", "fallback"),
]

MISSPELLED = [
    ("what is tokenisation", "tokenization"), ("explan stemming", "stemming_lemmatization"),
    ("what is sentimnt analysis", "sentiment_analysis"), ("helthcare ai", "ai_healthcare"),
    ("how do transfomers work", "transformers"), ("named entitiy recognition", "ner"),
    ("what is lemmatisation", "stemming_lemmatization"), ("ai in educaton", "ai_education"),
    ("agricultre and ai", "ai_agriculture"), ("self drivng cars", "ai_transport"),
    ("machin translation", "machine_translation"), ("word embedings", "embeddings"),
]

CONFIGS = [
    ("Bag-of-Words", dict(mode="bow", spell=False)),
    ("TF-IDF", dict(mode="tfidf", spell=False)),
    ("TF-IDF + spelling correction", dict(mode="tfidf", spell=True)),
    ("TF-IDF + bigrams + spelling", dict(mode="tfidf_bigram", spell=True)),
]


def predict(clf, text):
    intent, conf = clf.predict(text)
    return intent if intent is not None and conf >= THRESHOLD else "fallback"


def accuracy(clf, data):
    return sum(predict(clf, q) == y for q, y in data) / len(data)


def per_class_report(clf, data):
    tp, fp, fn = Counter(), Counter(), Counter()
    support = Counter(y for _, y in data)
    for q, y in data:
        p = predict(clf, q)
        if p == y:
            tp[y] += 1
        else:
            fp[p] += 1
            fn[y] += 1
    rows = []
    for c in sorted(set(support) | set(fp)):
        pr = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rc = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        rows.append((c, pr, rc, f1, support[c]))
    return rows


def main():
    full = CLEAN + MISSPELLED
    print(f"Test set: {len(CLEAN)} clean + {len(MISSPELLED)} misspelled questions "
          f"= {len(full)} (none are training patterns)\n")
    print(f"{'Configuration':34s} {'Clean':>8s} {'Misspelled':>11s} {'Overall':>9s}")
    print("-" * 66)
    for name, kw in CONFIGS:
        clf = IntentClassifier(INTENTS, **kw)
        print(f"{name:34s} {accuracy(clf, CLEAN):8.1%} {accuracy(clf, MISSPELLED):11.1%} {accuracy(clf, full):9.1%}")

    name, kw = CONFIGS[2]          # report details for the configuration used by the chatbot
    clf = IntentClassifier(INTENTS, **kw)
    print(f"\nPer-intent report for: {name}")
    print(f"{'Intent':26s} {'Precision':>9s} {'Recall':>7s} {'F1':>6s} {'Support':>8s}")
    print("-" * 60)
    rows = per_class_report(clf, full)
    for c, pr, rc, f1, sup in rows:
        print(f"{c:26s} {pr:9.2f} {rc:7.2f} {f1:6.2f} {sup:8d}")
    n = len(rows)
    print("-" * 60)
    print(f"{'Macro average':26s} {sum(r[1] for r in rows)/n:9.2f} {sum(r[2] for r in rows)/n:7.2f} "
          f"{sum(r[3] for r in rows)/n:6.2f} {len(full):8d}")

    errors = [(q, y, predict(clf, q)) for q, y in full if predict(clf, q) != y]
    print(f"\nMisclassified questions ({len(errors)}):")
    for q, y, p in errors:
        print(f"  '{q}'  true={y}  predicted={p}")




if __name__ == "__main__":
    if "--eval" in sys.argv:
        main()
    elif "--demo" in sys.argv:
        run_demo()
    elif "--cli" in sys.argv:
        run_cli()
    else:
        run_web()