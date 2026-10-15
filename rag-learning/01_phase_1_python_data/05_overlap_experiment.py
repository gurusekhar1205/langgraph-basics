
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
)

text = (
    "This is the first sentence. "
    "This is the second sentence. "
    "This is the third sentence. "
    "This is the fourth sentence."
)

chunks = splitter.split_text(text)

print("Original text length:", len(text))
print("Number of chunks:", len(chunks))
print()

for i, chunk in enumerate(chunks):
    print("Chunk index:", i)
    print("Chunk text:", repr(chunk))
    print("Character count:", len(chunk))
    print("-" * 50)
