from pathlib import Path

p = Path("src/chatbot.py")
text = p.read_text()

count_obj = text.count('"type": "object"')
count_str = text.count('"type": "string"')

text = text.replace('"type": "object"', '"type": "OBJECT"')
text = text.replace('"type": "string"', '"type": "STRING"')

p.write_text(text)
print(f"Fixed {count_obj} 'object' and {count_str} 'string' type declarations.")