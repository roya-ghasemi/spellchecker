#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install SpellChecker')


# In[4]:


get_ipython().system('pip install textblob')


# In[5]:


from spellchecker import SpellChecker

def spell_check(text):
    spell = SpellChecker()

    # Split the text into words
    words = text.split()

    # Find misspelled words
    misspelled = spell.unknown(words)

    corrected_text = []
    for word in words:
        if word in misspelled:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)

    return ' '.join(corrected_text)

input_text = "این یک نمونه متن با اشتباهات املایی است."
corrected_text = spell_check(input_text)

print("متن اصلاح شده:", corrected_text)


# In[6]:


from textblob import TextBlob

def spell_check(text):
    blob = TextBlob(text)
    corrected_text = blob.correct()
    return str(corrected_text)

input_text = "این یک نمونه متن با اشتباهات املایی است."
corrected_text = spell_check(input_text)

print("متن اصلاح شده:", corrected_text)


# In[7]:


from textblob import TextBlob

def spell_check(text):
    blob = TextBlob(text)
    corrected_text = blob.correct()
    return str(corrected_text)

def main():
    user_input = input("لطفاً متن خود را وارد کنید: ")
    corrected_text = spell_check(user_input)

    print("\nمتن اصلاح شده:")
    print(corrected_text)

if __name__ == "__main__":
    main()


# In[9]:


from transformers import pipeline

def spell_check(text):
    spell_check_pipeline = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

    corrected_text = spell_check_pipeline(text, max_length=150, num_return_sequences=1)[0]['generated_text']
    return corrected_text

def main():
    user_input = input("لطفاً متن خود را وارد کنید: ")
    corrected_text = spell_check(user_input)

    print("\nمتن اصلاح شده:")
    print(corrected_text)

if __name__ == "__main__":
    main()


# In[12]:


get_ipython().system('pip install transformers')


# In[14]:


from transformers import TFAutoModelForTokenClassification, AutoTokenizer

# مدل و توکنایزر مورد استفاده (می‌توانید از مدل‌های مشابه خود استفاده کنید)
model_name = "bert-base-cased"
model = TFAutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# تابع برای تشخیص و تصحیح اشتباهات املایی
def spell_check(text):
    # توکن‌بندی متن
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    
    # اجرای مدل بر روی متن
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
    outputs = model(inputs)
    predictions = tf.argmax(outputs.logits, axis=-1).numpy()
    
    # تصحیح اشتباهات
    corrected_text = []
    for token, label_id in zip(tokens, predictions[0]):
        label = model.config.id2label[label_id]
        if label == "B-ORG":
            corrected_text.append("[ORG]")
        elif label == "B-PER":
            corrected_text.append("[PER]")
        else:
            corrected_text.append(token)
    
    return " ".join(corrected_text)

# ورودی از کاربر گرفته شده و نمونه اشتباهات را تصحیح می‌کنیم
user_input = input("متن خود را وارد کنید: ")
corrected_output = spell_check(user_input)
print("متن اصلاح شده:", corrected_output)


# In[15]:


import tensorflow as tf


# In[16]:


import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# In[18]:


from transformers import TFAutoModelForTokenClassification, AutoTokenizer
import tensorflow as tf

# مدل و توکنایزر مورد استفاده (می‌توانید از مدل‌های مشابه خود استفاده کنید)
model_name = "bert-base-cased"
model = TFAutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# تابع برای تشخیص و تصحیح اشتباهات املایی
def spell_check(text):
    # توکن‌بندی متن
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    
    # اجرای مدل بر روی متن
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
    outputs = model(inputs)
    predictions = tf.argmax(outputs.logits, axis=-1).numpy()
    
    # تصحیح اشتباهات
    corrected_text = []
    for token, label_id in zip(tokens, predictions[0]):
        label = model.config.id2label[label_id]
        if label == "B-ORG":
            corrected_text.append("[ORG]")
        elif label == "B-PER":
            corrected_text.append("[PER]")
        else:
            corrected_text.append(token)
    
    return " ".join(corrected_text)

# ورودی از کاربر گرفته شده و نمونه اشتباهات را تصحیح می‌کنیم
user_input = input("متن خود را وارد کنید: ")
corrected_output = spell_check(user_input)
print("متن اصلاح شده:", corrected_output)


# In[19]:


get_ipython().system('pip install hazm')


# In[21]:


pip install hazm --user


# In[26]:


get_ipython().system('pip install pyspellchecker')


# In[32]:


from difflib import get_close_matches
from hazm import Normalizer, word_tokenize

# ایجاد یک نمونه از کلاس Normalizer
normalizer = Normalizer()

# ورودی از کاربر گرفته شده
user_input = input("متن خود را وارد کنید: ")

# نرمال‌سازی و توکن‌بندی متن
normalized_text = normalizer.normalize(user_input)
tokens = word_tokenize(normalized_text)

# لغت‌نامه‌ی ساده برای مثال
dictionary = ["تست", "فلور", "کامپیوتر", "موبایل", "رستوران"]

# تصحیح اشتباهات املایی
corrected_tokens = []
for token in tokens:
    closest_match = get_close_matches(token, dictionary, n=1, cutoff=0.7)
    corrected_token = closest_match[0] if closest_match else token
    corrected_tokens.append(corrected_token)

# تبدیل لیست توکن‌های تصحیح شده به متن
corrected_output = " ".join(corrected_tokens)

print("متن اصلاح شده:", corrected_output)


# In[33]:


from difflib import get_close_matches
from hazm import Normalizer, word_tokenize

# ایجاد یک نمونه از کلاس Normalizer
normalizer = Normalizer()

# ورودی از کاربر گرفته شده
user_input = input("متن خود را وارد کنید: ")

# نرمال‌سازی و توکن‌بندی متن
normalized_text = normalizer.normalize(user_input)
tokens = word_tokenize(normalized_text)

# لغت‌نامه‌ی ساده برای مثال
dictionary = ["تست", "فلور", "کامپیوتر", "موبایل", "رستوران"]

# تصحیح اشتباهات املایی
corrected_tokens = []
for token in tokens:
    closest_match = get_close_matches(token, dictionary, n=1, cutoff=0.7)
    corrected_token = closest_match[0] if closest_match else token
    corrected_tokens.append(corrected_token)

# تبدیل لیست توکن‌های تصحیح شده به متن
corrected_output = " ".join(corrected_tokens)

print("متن اصلاح شده:", corrected_output)


# In[34]:


from difflib import get_close_matches
from hazm import Normalizer, word_tokenize

# ایجاد یک نمونه از کلاس Normalizer
normalizer = Normalizer()

# ورودی از کاربر گرفته شده
user_input = input("متن خود را وارد کنید: ")

# نرمال‌سازی و توکن‌بندی متن
normalized_text = normalizer.normalize(user_input)
tokens = word_tokenize(normalized_text)

# لغت‌نامه‌ی ساده برای مثال
dictionary = ["تست", "فلور", "کامپیوتر", "موبایل", "رستوران"]

# تصحیح اشتباهات املایی
corrected_tokens = []
for token in tokens:
    closest_match = get_close_matches(token, dictionary, n=1, cutoff=0.7)
    corrected_token = closest_match[0] if closest_match else token
    corrected_tokens.append(corrected_token)

# تبدیل لیست توکن‌های تصحیح شده به متن
corrected_output = " ".join(corrected_tokens)

print("متن اصلاح شده:", corrected_output)


# In[40]:


# خواندن فایل واژه‌نامه دهخدا
with open("farsi_words.txt", "r", encoding="utf-8") as f:
    dictionary = [line.strip() for line in f]

# ورودی از کاربر گرفته شده
user_input = input("متن خود را وارد کنید: ")

# نرمال‌سازی و توکن‌بندی متن
normalized_text = normalizer.normalize(user_input)
tokens = word_tokenize(normalized_text)

# تصحیح اشتباهات املایی
corrected_tokens = []
for token in tokens:
    closest_match = get_close_matches(token, dictionary, n=1, cutoff=0.7)
    corrected_token = closest_match[0] if closest_match else token
    corrected_tokens.append(corrected_token)

# تبدیل لیست توکن‌های تصحیح شده به متن
corrected_output = " ".join(corrected_tokens)

print("متن اصلاح شده:", corrected_output)


# In[38]:


get_ipython().system('pip install indexer')


# In[39]:


from spellchecker import SpellChecker
from hazm import Normalizer, word_tokenize

# ایجاد یک نمونه از کلاسهای Normalizer و SpellChecker
normalizer = Normalizer()
spell = SpellChecker()

# ورودی از کاربر گرفته شده
user_input = input("متن خود را وارد کنید: ")

# نرمال‌سازی و توکن‌بندی متن
normalized_text = normalizer.normalize(user_input)
tokens = word_tokenize(normalized_text)

# تصحیح اشتباهات املایی
corrected_tokens = []
for token in tokens:
    corrected_token = spell.correction(token)
    corrected_tokens.append(corrected_token)

# تبدیل لیست توکن‌های تصحیح شده به متن
corrected_output = " ".join(corrected_tokens)

print("متن اصلاح شده:", corrected_output)


# In[50]:


from difflib import get_close_matches
from hazm import Normalizer, word_tokenize

# ایجاد یک نمونه از کلاس Normalizer
normalizer = Normalizer()

# ورودی از کاربر گرفته شده
user_input = input("متن خود را وارد کنید: ")

# نرمال‌سازی و توکن‌بندی متن
normalized_text = normalizer.normalize(user_input)
tokens = word_tokenize(normalized_text)

# لغت‌نامه‌ی ساده برای مثال
dictionary = ["تست", "فلور", "کامپیوتر", "موبایل", "رستوران"]

# متغیرهای برای محاسبه دقت
total_words = len(tokens)
corrected_words = 0

# تصحیح اشتباهات املایی
corrected_tokens = []
for token in tokens:
    closest_match = get_close_matches(token, dictionary, n=1, cutoff=0.7)
    corrected_token = closest_match[0] if closest_match else token
    if corrected_token == token:
        corrected_words += 1
    corrected_tokens.append(corrected_token)

# تبدیل لیست توکن‌های تصحیح شده به متن
corrected_output = " ".join(corrected_tokens)

# محاسبه دقت
accuracy = corrected_words / total_words * 100

print("متن اصلاح شده:", corrected_output)
print("دقت تصحیح اشتباهات:", accuracy, "%")


# In[ ]:




